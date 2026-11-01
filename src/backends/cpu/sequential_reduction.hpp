// Sequential CPU reduction kernels used as correctness and performance baselines.
#pragma once

#include <cstdint>

namespace mensurflow::cpu::sequential {

float sum_f32(const float* x, std::int64_t n);
float mean_f32(const float* x, std::int64_t n);
float max_f32(const float* x, std::int64_t n);

} // namespace mensurflow::cpu::sequential
