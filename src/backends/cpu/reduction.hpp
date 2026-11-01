// CPU full-tensor reduction kernels over contiguous float32 buffers.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

float sum_f32(const float* x, std::int64_t n);
float mean_f32(const float* x, std::int64_t n);
float max_f32(const float* x, std::int64_t n);
void sum_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis);
void mean_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                   std::int64_t axis);
void max_axis_f32(const float* x, float* out, const std::int64_t* shape, std::int64_t rank,
                  std::int64_t axis);

} // namespace mensurflow::cpu
