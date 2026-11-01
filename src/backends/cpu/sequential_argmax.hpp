// Sequential CPU last-axis argmax baseline over contiguous float32 inputs.
#pragma once

#include <cstdint>

namespace mensurflow::cpu::sequential {

void argmax_f32(const float* x, std::int64_t* out, std::int64_t rows, std::int64_t width);

} // namespace mensurflow::cpu::sequential
