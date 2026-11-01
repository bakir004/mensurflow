// CPU softmax kernels over contiguous float32 row-major buffers.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

void softmax_f32(const float* x, float* out, std::int64_t rows, std::int64_t width);

} // namespace mensurflow::cpu
