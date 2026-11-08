// Mathematical reference: long-double values, frexp/ldexp and explicit nearest-even rounding.
// No conversion implementation or rounding helper is shared with the production kernels.
#pragma once

#include "mensurflow/dtype.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace reference::casts {

using mensurflow::DType;

inline std::uint32_t load(const void* input, DType type, std::size_t i) {
    const auto* bytes = static_cast<const unsigned char*>(input);
    if (mensurflow::dtype_size(type) == 2) {
        std::uint16_t value;
        std::memcpy(&value, bytes + i * 2, 2);
        return value;
    }
    std::uint32_t value;
    std::memcpy(&value, bytes + i * 4, 4);
    return value;
}

inline void store(void* output, DType type, std::size_t i, std::uint32_t value) {
    auto* bytes = static_cast<unsigned char*>(output);
    if (mensurflow::dtype_size(type) == 2) {
        const auto small = static_cast<std::uint16_t>(value);
        std::memcpy(bytes + i * 2, &small, 2);
    } else {
        std::memcpy(bytes + i * 4, &value, 4);
    }
}

inline int fraction_bits(DType type) {
    return type == DType::F32 ? 23 : (type == DType::F16 ? 10 : 7);
}

inline long double decode(std::uint32_t raw, DType type) {
    if (type == DType::I32) {
        std::int32_t value;
        std::memcpy(&value, &raw, 4);
        return value;
    }
    const int mantissa = fraction_bits(type);
    const int exponent_bits = type == DType::F16 ? 5 : 8;
    const int bias = type == DType::F16 ? 15 : 127;
    const auto fraction = raw & ((1U << mantissa) - 1);
    const auto exponent = (raw >> mantissa) & ((1U << exponent_bits) - 1);
    const bool negative = ((raw >> (mantissa + exponent_bits)) & 1) != 0;
    long double value;
    if (exponent == (1U << exponent_bits) - 1) {
        value = fraction ? std::numeric_limits<long double>::quiet_NaN()
                         : std::numeric_limits<long double>::infinity();
    } else if (exponent == 0) {
        value = std::ldexp(static_cast<long double>(fraction), 1 - bias - mantissa);
    } else {
        value = std::ldexp(1.0L + std::ldexp(static_cast<long double>(fraction), -mantissa),
                           static_cast<int>(exponent) - bias);
    }
    return std::copysign(value, negative ? -1.0L : 1.0L);
}

inline std::uint32_t nearest_even(long double value) {
    const auto lower = std::floor(value);
    const auto base = static_cast<std::uint32_t>(lower);
    const auto remainder = value - lower;
    return base + (remainder > 0.5L || (remainder == 0.5L && (base & 1)));
}

inline std::uint32_t encode(long double value, DType type) {
    if (type == DType::I32) {
        const auto truncated = std::trunc(value);
        const auto result = !std::isfinite(value) || truncated < -2147483648.0L ||
                            truncated > 2147483647.0L
            ? std::numeric_limits<std::int32_t>::min() : static_cast<std::int32_t>(truncated);
        return static_cast<std::uint32_t>(result);
    }
    const int mantissa = fraction_bits(type);
    const int exponent_bits = type == DType::F16 ? 5 : 8;
    const int bias = type == DType::F16 ? 15 : 127;
    const auto sign = std::signbit(value) ? 1U << (mantissa + exponent_bits) : 0;
    const auto infinity = ((1U << exponent_bits) - 1) << mantissa;
    if (std::isnan(value)) {
        return sign | infinity | (1U << (mantissa - 1));
    }
    const auto magnitude = std::fabs(value);
    if (std::isinf(value)) {
        return sign | infinity;
    }
    if (magnitude == 0) {
        return sign;
    }
    int exponent;
    std::frexp(magnitude, &exponent);
    --exponent;
    if (exponent < 1 - bias) {
        return sign | nearest_even(std::ldexp(magnitude, mantissa + bias - 1));
    }
    auto significand = nearest_even(std::ldexp(magnitude, mantissa - exponent));
    if (significand == (1U << (mantissa + 1))) {
        significand >>= 1;
        ++exponent;
    }
    if (exponent + bias >= (1 << exponent_bits) - 1) {
        return sign | infinity;
    }
    return sign | (static_cast<std::uint32_t>(exponent + bias) << mantissa) |
           (significand - (1U << mantissa));
}

inline bool is_nan(std::uint32_t raw, DType type) {
    return type != DType::I32 && std::isnan(decode(raw, type));
}

inline void cast(const void* input, DType source, void* output, DType destination, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const auto raw = load(input, source, i);
        store(output, destination, i,
              source == destination ? raw : encode(decode(raw, source), destination));
    }
}

inline bool equal(const void* expected, const void* actual, DType type, std::size_t n,
                  bool exact_nan_bits = false) {
    for (std::size_t i = 0; i < n; ++i) {
        const auto want = load(expected, type, i), got = load(actual, type, i);
        if (want != got && (exact_nan_bits || !is_nan(want, type) || !is_nan(got, type))) {
            return false;
        }
    }
    return true;
}

} // namespace reference::casts
