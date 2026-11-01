// CPU full-tensor reduction kernels over contiguous float32 buffers.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

float sum_f32(const float* x, std::int64_t n);
float mean_f32(const float* x, std::int64_t n);
float max_f32(const float* x, std::int64_t n);

} // namespace mensurflow::cpu
