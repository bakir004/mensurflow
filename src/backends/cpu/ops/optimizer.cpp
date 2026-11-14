#include "backends/cpu/ops/optimizer.hpp"
#include "backends/cpu/threads.hpp"

#include <cmath>

namespace mensurflow::cpu {

void sgd_f32(const float* p, const float* g, float* out, float* velocity, std::int64_t n,
             float lr, float momentum, float decay) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        const float update = g[i] + decay * p[i];
        if (velocity != nullptr) {
            velocity[i] = momentum * velocity[i] + update;
            out[i] = p[i] - lr * velocity[i];
        } else {
            out[i] = p[i] - lr * update;
        }
    }
}

void adam_f32(const float* p, const float* g, float* out, float* m, float* v, std::int64_t n,
              float lr, float beta1, float beta2, float eps, float decay, std::int64_t step,
              bool decoupled) {
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    const float scale = lr * std::sqrt(bc2) / bc1;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        const float gradient = decoupled ? g[i] : g[i] + decay * p[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * gradient;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gradient * gradient;
        out[i] = decoupled ? p[i] * (1.0f - lr * decay) - scale * m[i] / (std::sqrt(v[i]) + eps)
                           : p[i] - scale * m[i] / (std::sqrt(v[i]) + eps);
    }
}
} // namespace mensurflow::cpu
