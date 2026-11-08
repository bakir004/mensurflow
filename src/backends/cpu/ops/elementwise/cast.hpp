// Contiguous CPU casts. F16 and BF16 use uint16_t storage, with IEEE binary16/bfloat16 bits.
#pragma once

#include "mensurflow/dtype.hpp"

#include <cstdint>

namespace mensurflow::cpu {

// Source/destination must be F32, I32, F16 or BF16; otherwise throws invalid_argument.
// n >= 0; buffers are valid/aligned for their element types when n > 0. Buffers must not
// overlap, except exact aliasing for a same-dtype cast. Null buffers are permitted for n == 0.
void cast_buffer(const void* input, DType source, void* output, DType destination, std::int64_t n);

} // namespace mensurflow::cpu
