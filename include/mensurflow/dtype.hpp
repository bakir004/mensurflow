// Element types a Tensor can hold. Kernels are registered per dtype, so most
// backends will only support a subset (F32 first, F16/BF16 later).
#pragma once

#include <cstddef>
#include <cstdint>

namespace mensurflow {

enum class DType { F32, F16, BF16, I32, I64, U8, Bool };

// Bytes per element.
std::size_t dtype_size(DType dtype);

// Stable lowercase name ("f32", "bf16", ...); an invalid value throws invalid_argument.
const char* dtype_name(DType dtype);

bool is_floating_point(DType dtype);

// Maps a C++ element type to its DType, e.g. dtype_of<float> == DType::F32.
// F16/BF16 have no standard C++ type; bool is left out because std::vector<bool> has no data().
template <typename T> struct DTypeOf;

template <> struct DTypeOf<float> {
    static constexpr DType value = DType::F32;
};

template <> struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::I32;
};

template <> struct DTypeOf<std::int64_t> {
    static constexpr DType value = DType::I64;
};

template <> struct DTypeOf<std::uint8_t> {
    static constexpr DType value = DType::U8;
};

template <typename T> inline constexpr DType dtype_of = DTypeOf<T>::value;

} // namespace mensurflow
