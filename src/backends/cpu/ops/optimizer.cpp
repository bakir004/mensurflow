// CPU SGD, Adam and AdamW: one loop over the elements, parallelized by OpenMP above the cutoff.
// Every element is independent, so the result is the same on any number of threads.
//
// Each update has a portable loop (the compiler vectorizes it for the baseline SSE2 build) and an
// AVX2 version for x86-64, selected at run time (see optimizer.hpp). The AVX2 version does exactly the
// operations of the portable one in the same order, with separate multiply and add (no FMA), so the
// two give identical bits.
#include "backends/cpu/ops/optimizer.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"

#include <atomic>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#define MENSURFLOW_OPTIMIZER_AVX2 1
#include <immintrin.h>
#endif

// AVX2 without FMA on purpose. With "avx2,fma" GCC and Clang fuse a * b + c into one rounding by default
// (-ffp-contract=fast), including in the scalar tail of these functions, and the results would no longer
// match the portable loop bit for bit. Without the fma target the compiler cannot emit FMA here.
// That is not enough when the whole file is compiled with FMA (-march=native, -mfma): the fusing is then
// possible in both versions and happens differently in each, so CMakeLists.txt also compiles this file
// with -ffp-contract=off.
#if defined(_MSC_VER)
#define MENSURFLOW_OPTIMIZER_TARGET
#else
#define MENSURFLOW_OPTIMIZER_TARGET __attribute__((target("avx2")))
#endif

namespace mensurflow::cpu {

namespace {

std::atomic<bool> avx2_switch{true};

// ---- SGD ---------------------------------------------------------------------------------------

template <bool Momentum>
void sgd_range(float* p, const float* g, float* velocity, std::int64_t begin, std::int64_t end,
               float lr, float momentum, float weight_decay) {
    for (std::int64_t i = begin; i < end; ++i) {
        if (Momentum) {
            const float v = momentum * velocity[i] + (g[i] + weight_decay * p[i]);
            velocity[i] = v;
            p[i] -= lr * v;
        } else {
            p[i] -= lr * (g[i] + weight_decay * p[i]);
        }
    }
}

template <bool Momentum>
void sgd_portable(float* p, const float* g, float* velocity, std::int64_t n, float lr,
                  float momentum, float weight_decay) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        if (Momentum) {
            const float v = momentum * velocity[i] + (g[i] + weight_decay * p[i]);
            velocity[i] = v;
            p[i] -= lr * v;
        } else {
            p[i] -= lr * (g[i] + weight_decay * p[i]);
        }
    }
}

// ---- Adam / AdamW ------------------------------------------------------------------------------

struct AdamCoefficients {
    float lr, beta1, beta2, eps, weight_decay;
    float step_size;    // lr / (1 - beta1^step)
    float inv_sqrt_bc2; // 1 / sqrt(1 - beta2^step)
    float keep;         // AdamW's decoupled shrink, 1 - lr * weight_decay
    float one_minus_beta1, one_minus_beta2;
};

AdamCoefficients adam_coefficients(float lr, float beta1, float beta2, float eps,
                                   float weight_decay, std::int64_t step) {
    // Bias corrections in double: beta^step for a large step is a tiny difference from 1.
    const double t = static_cast<double>(step);
    AdamCoefficients c{};
    c.lr = lr;
    c.beta1 = beta1;
    c.beta2 = beta2;
    c.eps = eps;
    c.weight_decay = weight_decay;
    c.step_size = static_cast<float>(lr / (1.0 - std::pow(static_cast<double>(beta1), t)));
    c.inv_sqrt_bc2 =
        static_cast<float>(1.0 / std::sqrt(1.0 - std::pow(static_cast<double>(beta2), t)));
    c.keep = 1.0f - lr * weight_decay;
    c.one_minus_beta1 = 1.0f - beta1;
    c.one_minus_beta2 = 1.0f - beta2;
    return c;
}

template <bool Decoupled>
inline void adam_element(float* p, const float* g, float* m, float* v, std::int64_t i,
                         const AdamCoefficients& c) {
    const float grad = Decoupled ? g[i] : g[i] + c.weight_decay * p[i];
    const float mi = c.beta1 * m[i] + c.one_minus_beta1 * grad;
    const float vi = c.beta2 * v[i] + c.one_minus_beta2 * grad * grad;
    m[i] = mi;
    v[i] = vi;
    const float denom = std::sqrt(vi) * c.inv_sqrt_bc2 + c.eps;
    const float base = Decoupled ? p[i] * c.keep : p[i];
    p[i] = base - c.step_size * mi / denom;
}

template <bool Decoupled>
void adam_portable(float* p, const float* g, float* m, float* v, std::int64_t n,
                   const AdamCoefficients& c) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        adam_element<Decoupled>(p, g, m, v, i, c);
    }
}

// ---- AVX2 --------------------------------------------------------------------------------------
#ifdef MENSURFLOW_OPTIMIZER_AVX2

template <bool Momentum>
MENSURFLOW_OPTIMIZER_TARGET void sgd_avx2(float* p, const float* g, float* velocity, std::int64_t n,
                                          float lr, float momentum, float weight_decay) {
    const std::int64_t blocks = n / 8;
    const __m256 vlr = _mm256_set1_ps(lr);
    const __m256 vmomentum = _mm256_set1_ps(momentum);
    const __m256 vdecay = _mm256_set1_ps(weight_decay);
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t b = 0; b < blocks; ++b) {
        const std::int64_t i = b * 8;
        const __m256 pv = _mm256_loadu_ps(p + i);
        const __m256 update = _mm256_add_ps(_mm256_loadu_ps(g + i), _mm256_mul_ps(vdecay, pv));
        if (Momentum) {
            const __m256 v =
                _mm256_add_ps(_mm256_mul_ps(vmomentum, _mm256_loadu_ps(velocity + i)), update);
            _mm256_storeu_ps(velocity + i, v);
            _mm256_storeu_ps(p + i, _mm256_sub_ps(pv, _mm256_mul_ps(vlr, v)));
        } else {
            _mm256_storeu_ps(p + i, _mm256_sub_ps(pv, _mm256_mul_ps(vlr, update)));
        }
    }
    sgd_range<Momentum>(p, g, velocity, blocks * 8, n, lr, momentum, weight_decay);
}

template <bool Decoupled>
MENSURFLOW_OPTIMIZER_TARGET void adam_avx2(float* p, const float* g, float* m, float* v,
                                           std::int64_t n, const AdamCoefficients& c) {
    const std::int64_t blocks = n / 8;
    const __m256 vbeta1 = _mm256_set1_ps(c.beta1);
    const __m256 vbeta2 = _mm256_set1_ps(c.beta2);
    const __m256 vomb1 = _mm256_set1_ps(c.one_minus_beta1);
    const __m256 vomb2 = _mm256_set1_ps(c.one_minus_beta2);
    const __m256 vdecay = _mm256_set1_ps(c.weight_decay);
    const __m256 vkeep = _mm256_set1_ps(c.keep);
    const __m256 vstep = _mm256_set1_ps(c.step_size);
    const __m256 vinv = _mm256_set1_ps(c.inv_sqrt_bc2);
    const __m256 veps = _mm256_set1_ps(c.eps);
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t b = 0; b < blocks; ++b) {
        const std::int64_t i = b * 8;
        const __m256 pv = _mm256_loadu_ps(p + i);
        const __m256 gv = _mm256_loadu_ps(g + i);
        const __m256 grad = Decoupled ? gv : _mm256_add_ps(gv, _mm256_mul_ps(vdecay, pv));
        const __m256 mi = _mm256_add_ps(_mm256_mul_ps(vbeta1, _mm256_loadu_ps(m + i)),
                                        _mm256_mul_ps(vomb1, grad));
        const __m256 vi = _mm256_add_ps(_mm256_mul_ps(vbeta2, _mm256_loadu_ps(v + i)),
                                        _mm256_mul_ps(_mm256_mul_ps(vomb2, grad), grad));
        _mm256_storeu_ps(m + i, mi);
        _mm256_storeu_ps(v + i, vi);
        const __m256 denom = _mm256_add_ps(_mm256_mul_ps(_mm256_sqrt_ps(vi), vinv), veps);
        const __m256 base = Decoupled ? _mm256_mul_ps(pv, vkeep) : pv;
        _mm256_storeu_ps(p + i,
                         _mm256_sub_ps(base, _mm256_div_ps(_mm256_mul_ps(vstep, mi), denom)));
    }
    for (std::int64_t i = blocks * 8; i < n; ++i) {
        adam_element<Decoupled>(p, g, m, v, i, c);
    }
}

#endif // MENSURFLOW_OPTIMIZER_AVX2

} // namespace

bool optimizer_avx2_available() {
#ifdef MENSURFLOW_OPTIMIZER_AVX2
    return packed_gemm_supported();
#else
    return false;
#endif
}

void set_optimizer_avx2(bool enabled) { avx2_switch.store(enabled); }

bool optimizer_avx2_enabled() { return optimizer_avx2_available() && avx2_switch.load(); }

void sgd_f32(float* p, const float* g, float* velocity, std::int64_t n, float lr, float momentum,
             float weight_decay) {
#ifdef MENSURFLOW_OPTIMIZER_AVX2
    if (optimizer_avx2_enabled()) {
        if (velocity != nullptr) {
            sgd_avx2<true>(p, g, velocity, n, lr, momentum, weight_decay);
        } else {
            sgd_avx2<false>(p, g, velocity, n, lr, momentum, weight_decay);
        }
        return;
    }
#endif
    if (velocity != nullptr) {
        sgd_portable<true>(p, g, velocity, n, lr, momentum, weight_decay);
    } else {
        sgd_portable<false>(p, g, velocity, n, lr, momentum, weight_decay);
    }
}

void adam_f32(float* p, const float* g, float* m, float* v, std::int64_t n, float lr, float beta1,
              float beta2, float eps, float weight_decay, std::int64_t step, bool decoupled) {
    const AdamCoefficients c = adam_coefficients(lr, beta1, beta2, eps, weight_decay, step);
#ifdef MENSURFLOW_OPTIMIZER_AVX2
    if (optimizer_avx2_enabled()) {
        if (decoupled) {
            adam_avx2<true>(p, g, m, v, n, c);
        } else {
            adam_avx2<false>(p, g, m, v, n, c);
        }
        return;
    }
#endif
    if (decoupled) {
        adam_portable<true>(p, g, m, v, n, c);
    } else {
        adam_portable<false>(p, g, m, v, n, c);
    }
}

} // namespace mensurflow::cpu
