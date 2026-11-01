// CPU last-axis argmax kernels over contiguous float32 inputs.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

void argmax_f32(const float* x, std::int64_t* out, std::int64_t rows, std::int64_t width);

} // namespace mensurflow::cpu
