// CPU SGD, Adam and AdamW: one loop over the elements, parallelized by OpenMP above the cutoff.
// Every element is independent, so the result is the same on any number of threads.
#include "backends/cpu/ops/optimizer.hpp"

#include "backends/cpu/threads.hpp"

#include <cmath>

namespace mensurflow::cpu {

void sgd_f32(float* p, const float* g, float* velocity, std::int64_t n, float lr, float momentum,
             float weight_decay) {
    if (velocity != nullptr) {
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
        for (std::int64_t i = 0; i < n; ++i) {
            const float v = momentum * velocity[i] + (g[i] + weight_decay * p[i]);
            velocity[i] = v;
            p[i] -= lr * v;
        }
    } else {
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] -= lr * (g[i] + weight_decay * p[i]);
        }
    }
}

namespace {

template <bool Decoupled>
void adam_loop(float* p, const float* g, float* m, float* v, std::int64_t n, float lr, float beta1,
               float beta2, float eps, float weight_decay, std::int64_t step) {
    // Bias corrections in double: beta^step for a large step is a tiny difference from 1.
    const double t = static_cast<double>(step);
    const float step_size =
        static_cast<float>(lr / (1.0 - std::pow(static_cast<double>(beta1), t)));
    const float inv_sqrt_bc2 =
        static_cast<float>(1.0 / std::sqrt(1.0 - std::pow(static_cast<double>(beta2), t)));
    const float keep = 1.0f - lr * weight_decay; // AdamW's decoupled shrink
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        const float grad = Decoupled ? g[i] : g[i] + weight_decay * p[i];
        const float mi = beta1 * m[i] + one_minus_beta1 * grad;
        const float vi = beta2 * v[i] + one_minus_beta2 * grad * grad;
        m[i] = mi;
        v[i] = vi;
        const float denom = std::sqrt(vi) * inv_sqrt_bc2 + eps;
        const float base = Decoupled ? p[i] * keep : p[i];
        p[i] = base - step_size * mi / denom;
    }
}

} // namespace

void adam_f32(float* p, const float* g, float* m, float* v, std::int64_t n, float lr, float beta1,
              float beta2, float eps, float weight_decay, std::int64_t step, bool decoupled) {
    if (decoupled) {
        adam_loop<true>(p, g, m, v, n, lr, beta1, beta2, eps, weight_decay, step);
    } else {
        adam_loop<false>(p, g, m, v, n, lr, beta1, beta2, eps, weight_decay, step);
    }
}

} // namespace mensurflow::cpu
