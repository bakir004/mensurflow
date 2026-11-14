// CPU optimizer updates: one fused pass over the parameter and its state, in place.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// p -= lr * v, where v = momentum * v + (g + weight_decay * p). Without a velocity buffer
// (velocity == nullptr) the update is p -= lr * (g + weight_decay * p). The buffer starts at zero,
// so the first step is the plain gradient step, as in PyTorch.
void sgd_f32(float* p, const float* g, float* velocity, std::int64_t n, float lr, float momentum,
             float weight_decay);

// One Adam step (`step` counts from 1) for PyTorch's formulation: bias-corrected moments,
//   p -= lr / (1 - beta1^step) * m / (sqrt(v) / sqrt(1 - beta2^step) + eps).
// decoupled == false (Adam) adds weight_decay * p to the gradient; decoupled == true (AdamW)
// first shrinks the parameter by (1 - lr * weight_decay) and leaves the gradient alone.
void adam_f32(float* p, const float* g, float* m, float* v, std::int64_t n, float lr, float beta1,
              float beta2, float eps, float weight_decay, std::int64_t step, bool decoupled);

} // namespace mensurflow::cpu
