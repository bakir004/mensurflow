// Simple scalar references, deliberately independent of the production Philox/block loops.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace reference {

inline std::uint32_t philox_sample(std::uint64_t seed, std::uint64_t position) {
    const std::uint64_t block = position / 4;
    std::uint32_t a = static_cast<std::uint32_t>(block);
    std::uint32_t b = static_cast<std::uint32_t>(block >> 32);
    std::uint32_t c = 0, d = 0;
    const auto key0 = static_cast<std::uint32_t>(seed);
    const auto key1 = static_cast<std::uint32_t>(seed >> 32);
    for (std::uint32_t r = 0; r != 10; ++r) {
        const auto product0 = static_cast<std::uint64_t>(a) * 3528531795u;
        const auto product1 = static_cast<std::uint64_t>(c) * 3449720151u;
        const auto next_a = static_cast<std::uint32_t>(product1 >> 32) ^ b ^
                            (key0 + r * 2654435769u);
        const auto next_c = static_cast<std::uint32_t>(product0 >> 32) ^ d ^
                            (key1 + r * 3144134277u);
        a = next_a;
        b = static_cast<std::uint32_t>(product1);
        c = next_c;
        d = static_cast<std::uint32_t>(product0);
    }
    const std::uint32_t values[] = {a, b, c, d};
    return values[position % 4];
}

inline std::vector<float> uniform(std::int64_t n, float low, float high, std::uint64_t seed,
                                   std::uint64_t offset) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const double fraction = (philox_sample(seed, offset + i) >> 8) / 16777216.0;
        const double value = low + (static_cast<double>(high) - low) * fraction;
        const float last = high == low ? low : std::nextafter(high, low);
        values[static_cast<std::size_t>(i)] = std::max(low, std::min(last, static_cast<float>(value)));
    }
    return values;
}

inline std::vector<float> normal(std::int64_t n, float mean, float stddev, std::uint64_t seed,
                                  std::uint64_t offset) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const auto position = offset + static_cast<std::uint64_t>(i);
        const auto pair = position - position % 2;
        const double u = (philox_sample(seed, pair) + 0.5) / 4294967296.0;
        const double v = (philox_sample(seed, pair + 1) + 0.5) / 4294967296.0;
        const double radius = std::sqrt(-2.0 * std::log(u));
        const double angle = 2.0 * std::acos(-1.0) * v;
        const double z = radius * (position % 2 == 0 ? std::cos(angle) : std::sin(angle));
        values[static_cast<std::size_t>(i)] = static_cast<float>(mean + stddev * z);
    }
    return values;
}

inline std::vector<float> bernoulli(std::int64_t n, float probability, std::uint64_t seed,
                                     std::uint64_t offset) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] =
            philox_sample(seed, offset + i) / 4294967296.0 < probability ? 1.0f : 0.0f;
    }
    return values;
}

inline std::vector<std::uint8_t> dropout_mask(std::int64_t n, float probability, bool training,
                                               std::uint64_t seed, std::uint64_t offset) {
    std::vector<std::uint8_t> values(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] =
            !training || philox_sample(seed, offset + i) / 4294967296.0 >= probability;
    }
    return values;
}

inline std::vector<float> dropout(const std::vector<float>& input,
                                   const std::vector<std::uint8_t>& mask, float probability,
                                   bool training) {
    std::vector<float> values(input.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!training || probability == 0.0f) {
            values[i] = input[i];
        } else {
            values[i] = probability < 1.0f && mask[i]
                            ? input[i] * (1.0f / (1.0f - probability))
                            : 0.0f;
        }
    }
    return values;
}

} // namespace reference
