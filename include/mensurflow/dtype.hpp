// Element types a Tensor can hold. Kernels are registered per dtype, so most
// backends will only support a subset (F32 first, F16/BF16 later).
#pragma once

#include <cstddef>

namespace mensurflow {

enum class DType { F32, F16, BF16, I32, I64, U8, Bool };

// Bytes per element.
std::size_t dtype_size(DType dtype);

// Stable lowercase name ("f32", "bf16", ...); an invalid value throws invalid_argument.
const char* dtype_name(DType dtype);

bool is_floating_point(DType dtype);

} // namespace mensurflow
