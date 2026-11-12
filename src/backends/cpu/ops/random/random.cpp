#include "backends/cpu/ops/random/random.hpp"

#include "backends/cpu/ops/random/philox.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mensurflow::cpu {

namespace {

// Generate each Philox block once. The first/last blocks may be partial, so nonmultiples
// of four and offsets in the middle of a block produce the same stream as a larger call.
template <typename Write>
void random_blocks(std::int64_t n, std::uint64_t seed, std::uint64_t offset, Write write) {
    if (n == 0) {
        return;
    }
    const auto skip = offset % 4;
    const auto count = static_cast<std::uint64_t>(n);
    const auto blocks = static_cast<std::int64_t>(count / 4 + (count % 4 + skip + 3) / 4);
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t b = 0; b < blocks; ++b) {
        const auto words = philox_block(seed, offset / 4 + static_cast<std::uint64_t>(b));
        for (std::uint64_t lane = 0; lane < 4; ++lane) {
            const auto position = static_cast<std::uint64_t>(b) * 4 + lane;
            if (position >= skip && position - skip < count) {
                write(words, lane, static_cast<std::int64_t>(position - skip));
            }
        }
    }
}

inline double unit(std::uint32_t word) { return static_cast<double>(word) * 0x1p-32; }

void fill(float* out, std::int64_t n, float value) {
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = value;
    }
}

} // namespace

void uniform_f32(float* out, std::int64_t n, float low, float high, std::uint64_t seed,
                 std::uint64_t offset) {
    if (low == high) {
        fill(out, n, low);
        return;
    }
    const double width = static_cast<double>(high) - low;
    const float last = std::nextafter(high, low);
    random_blocks(n, seed, offset,
                  [=](const PhiloxCounter& words, std::uint64_t lane, std::int64_t i) {
                      // An exactly representable 24-bit fraction in [0,1). Double scaling
                      // avoids overflow when finite F32 endpoints straddle the full range.
                      const double u = static_cast<double>(words[lane] >> 8) * 0x1p-24;
                      const float value = static_cast<float>(static_cast<double>(low) + width * u);
                      out[i] = std::min(last, std::max(low, value));
                  });
}

void normal_f32(float* out, std::int64_t n, float mean, float stddev, std::uint64_t seed,
                std::uint64_t offset) {
    if (stddev == 0.0f) {
        fill(out, n, mean);
        return;
    }
    if (n == 0) {
        return;
    }
    const auto skip = offset % 4;
    const auto count = static_cast<std::uint64_t>(n);
    const auto blocks = static_cast<std::int64_t>(count / 4 + (count % 4 + skip + 3) / 4);
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t b = 0; b < blocks; ++b) {
        const auto words = philox_block(seed, offset / 4 + static_cast<std::uint64_t>(b));
        for (std::uint64_t pair = 0; pair < 2; ++pair) {
            // Midpoints exclude zero and one; log never sees zero. One pair of uniforms
            // yields both the cosine and sine outputs of the Box-Muller transform.
            const double u1 = (static_cast<double>(words[2 * pair]) + 0.5) * 0x1p-32;
            const double u2 = (static_cast<double>(words[2 * pair + 1]) + 0.5) * 0x1p-32;
            const double radius = std::sqrt(-2.0 * std::log(u1));
            const double angle = 6.283185307179586476925286766559 * u2;
            const double z[2] = {radius * std::cos(angle), radius * std::sin(angle)};
            for (std::uint64_t lane = 0; lane < 2; ++lane) {
                const auto position = static_cast<std::uint64_t>(b) * 4 + 2 * pair + lane;
                if (position >= skip && position - skip < count) {
                    out[position - skip] =
                        static_cast<float>(static_cast<double>(mean) + stddev * z[lane]);
                }
            }
        }
    }
}

void bernoulli_f32(float* out, std::int64_t n, float probability, std::uint64_t seed,
                   std::uint64_t offset) {
    if (probability == 0.0f || probability == 1.0f) {
        fill(out, n, probability == 0.0f ? 0.0f : 1.0f);
        return;
    }
    random_blocks(n, seed, offset,
                  [=](const PhiloxCounter& words, std::uint64_t lane, std::int64_t i) {
                      out[i] = unit(words[lane]) < probability ? 1.0f : 0.0f;
                  });
}

void dropout_f32(const float* x, float* out, std::uint8_t* mask, std::int64_t n,
                 float probability, bool training, std::uint64_t seed, std::uint64_t offset) {
    if (!training || probability == 0.0f || probability == 1.0f) {
        const bool keep = !training || probability == 0.0f;
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
        for (std::int64_t i = 0; i < n; ++i) {
            mask[i] = static_cast<std::uint8_t>(keep);
            out[i] = keep ? x[i] : 0.0f;
        }
        return;
    }
    const float scale = 1.0f / (1.0f - probability);
    random_blocks(n, seed, offset,
                  [=](const PhiloxCounter& words, std::uint64_t lane, std::int64_t i) {
                      const bool keep = unit(words[lane]) >= probability;
                      mask[i] = static_cast<std::uint8_t>(keep);
                      out[i] = keep ? x[i] * scale : 0.0f;
                  });
}

void dropout_backward_f32(const float* gradient, const std::uint8_t* mask, float* out,
                          std::int64_t n, float probability, bool training) {
    const float scale = training && probability < 1.0f ? 1.0f / (1.0f - probability) : 1.0f;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        if (!training || probability == 0.0f) {
            out[i] = gradient[i];
        } else {
            out[i] = probability < 1.0f && mask[i] != 0 ? gradient[i] * scale : 0.0f;
        }
    }
}

} // namespace mensurflow::cpu
