#include "backends/cpu/ops/elementwise/cast.hpp"

#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"
#include "backends/cpu/threads.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace mensurflow::cpu {
namespace {

static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24, "cast requires IEEE binary32 float");

std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// All callers use shifts in [1, 24], so neither the mask nor the halfway bit can overflow.
std::uint32_t round_even_shift(std::uint32_t value, int shift) {
    const auto result = value >> shift;
    const auto remainder = value & ((std::uint32_t{1} << shift) - 1);
    const auto halfway = std::uint32_t{1} << (shift - 1);
    return result + (remainder > halfway || (remainder == halfway && (result & 1)));
}

std::uint16_t to_bf16(float value) {
    const auto raw = bits(value);
    if ((raw & 0x7fffffffU) > 0x7f800000U) {
        // Keep NaNs as NaNs even when their payload only occupies discarded bits.
        return static_cast<std::uint16_t>((raw >> 16) | 0x0040U);
    }
    return static_cast<std::uint16_t>((raw + 0x7fffU + ((raw >> 16) & 1U)) >> 16);
}

std::uint16_t to_f16(float value) {
    const auto raw = bits(value);
    const auto sign = (raw >> 16) & 0x8000U;
    const auto exponent = (raw >> 23) & 0xffU;
    const auto fraction = raw & 0x007fffffU;
    if (exponent == 255) {
        return static_cast<std::uint16_t>(sign | 0x7c00U |
                                         (fraction ? (fraction >> 13) | 0x0200U : 0));
    }
    const int target_exponent = static_cast<int>(exponent) - 127 + 15;
    if (target_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (target_exponent <= 0) {
        if (target_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        return static_cast<std::uint16_t>(sign |
            round_even_shift(fraction | 0x00800000U, 14 - target_exponent));
    }
    // Addition lets rounding carry from the fraction into the exponent, including infinity.
    return static_cast<std::uint16_t>(sign |
        ((static_cast<std::uint32_t>(target_exponent) << 10) + round_even_shift(fraction, 13)));
}

float from_f16(std::uint16_t value) {
    const auto sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
    int exponent = (value >> 10) & 31;
    std::uint32_t fraction = value & 0x03ffU;
    if (exponent == 0) {
        if (fraction == 0) {
            return from_bits(sign);
        }
        // Normalize a half subnormal before widening it exactly to binary32.
        exponent = -14;
        while ((fraction & 0x0400U) == 0) {
            fraction <<= 1;
            --exponent;
        }
        return from_bits(sign | (static_cast<std::uint32_t>(exponent + 127) << 23) |
                         ((fraction & 0x03ffU) << 13));
    }
    if (exponent == 31) {
        return from_bits(sign | 0x7f800000U | (fraction << 13) |
                         (fraction ? 0x00400000U : 0));
    }
    return from_bits(sign | (static_cast<std::uint32_t>(exponent - 15 + 127) << 23) |
                     (fraction << 13));
}

// Convert integers directly, avoiding a double rounding through F32 for BF16/F16 destinations.
std::uint32_t integer_float_bits(std::int32_t value, int mantissa, int exponent_bits, int bias) {
    const auto sign = value < 0 ? std::uint32_t{1} << (mantissa + exponent_bits) : 0;
    const auto unsigned_value = static_cast<std::uint32_t>(value);
    const auto magnitude = value < 0 ? 0U - unsigned_value : unsigned_value;
    if (magnitude == 0) {
        return 0;
    }
    int exponent = 0;
    for (auto remaining = magnitude; (remaining >>= 1) != 0;) {
        ++exponent;
    }
    auto significand = exponent <= mantissa ? magnitude << (mantissa - exponent)
                                           : round_even_shift(magnitude, exponent - mantissa);
    if (significand == (std::uint32_t{1} << (mantissa + 1))) {
        significand >>= 1;
        ++exponent;
    }
    const auto max_exponent = (std::uint32_t{1} << exponent_bits) - 1;
    if (exponent + bias >= static_cast<int>(max_exponent)) {
        return sign | (max_exponent << mantissa);
    }
    return sign | (static_cast<std::uint32_t>(exponent + bias) << mantissa) |
           (significand & ((std::uint32_t{1} << mantissa) - 1));
}

template <DType Type>
using Storage = std::conditional_t<Type == DType::F32, float,
                std::conditional_t<Type == DType::I32, std::int32_t, std::uint16_t>>;

template <DType Source, DType Destination>
Storage<Destination> convert(Storage<Source> value) {
    if constexpr (Source == DType::I32) {
        if constexpr (Destination == DType::F32) {
            return from_bits(integer_float_bits(value, 23, 8, 127));
        } else if constexpr (Destination == DType::F16) {
            return static_cast<std::uint16_t>(integer_float_bits(value, 10, 5, 15));
        } else if constexpr (Destination == DType::BF16) {
            return static_cast<std::uint16_t>(integer_float_bits(value, 7, 8, 127));
        }
    } else {
        float decoded;
        if constexpr (Source == DType::F32) {
            decoded = value;
        } else if constexpr (Source == DType::F16) {
            decoded = from_f16(value);
        } else {
            decoded = from_bits(static_cast<std::uint32_t>(value) << 16);
        }
        if constexpr (Destination == DType::F32) {
            return decoded;
        } else if constexpr (Destination == DType::F16) {
            return to_f16(decoded);
        } else if constexpr (Destination == DType::BF16) {
            return to_bf16(decoded);
        } else {
            // Guard before the C++ conversion: casting NaN/out-of-range floats is undefined.
            if (!(decoded >= -2147483648.0f && decoded < 2147483648.0f)) {
                return std::numeric_limits<std::int32_t>::min();
            }
            return static_cast<std::int32_t>(decoded);
        }
    }
}

template <DType Source, DType Destination>
void cast_loop(const void* input, void* output, std::int64_t n) {
    const auto* x = static_cast<const Storage<Source>*>(input);
    auto* out = static_cast<Storage<Destination>*>(output);
    if constexpr (Source == Destination) {
        if (input == output) {
            return;
        }
    }
    MENSURFLOW_PARALLEL_FOR(if (n > elementwise_parallel_cutoff(ElementwiseOp::cast)))
    for (std::int64_t i = 0; i < n; ++i) {
        if constexpr (Source == Destination) {
            std::memcpy(out + i, x + i, sizeof(Storage<Source>));
        } else {
            out[i] = convert<Source, Destination>(x[i]);
        }
    }
}

template <DType Source>
void cast_to(const void* input, void* output, DType destination, std::int64_t n) {
    switch (destination) {
    case DType::F32:
        return cast_loop<Source, DType::F32>(input, output, n);
    case DType::I32:
        return cast_loop<Source, DType::I32>(input, output, n);
    case DType::F16:
        return cast_loop<Source, DType::F16>(input, output, n);
    case DType::BF16:
        return cast_loop<Source, DType::BF16>(input, output, n);
    default:
        throw std::invalid_argument("cast_buffer: unsupported destination dtype");
    }
}

} // namespace

void cast_buffer(const void* input, DType source, void* output, DType destination, std::int64_t n) {
    switch (source) {
    case DType::F32:
        return cast_to<DType::F32>(input, output, destination, n);
    case DType::I32:
        return cast_to<DType::I32>(input, output, destination, n);
    case DType::F16:
        return cast_to<DType::F16>(input, output, destination, n);
    case DType::BF16:
        return cast_to<DType::BF16>(input, output, destination, n);
    default:
        throw std::invalid_argument("cast_buffer: unsupported source dtype");
    }
}

} // namespace mensurflow::cpu
