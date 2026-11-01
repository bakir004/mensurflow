// Sequential CPU softmax baseline over contiguous float32 buffers.
#pragma once

#include <cstdint>

namespace mensurflow::cpu::sequential {

void softmax_f32(const float* x, float* out, std::int64_t rows, std::int64_t width);

} // namespace mensurflow::cpu::sequential
