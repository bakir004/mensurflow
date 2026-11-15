// CPU softmax and log_softmax, forward and backward: one parallel loop over the lines each.
// A line is contiguous when inner == 1 (softmax over the last axis, the common case); otherwise
// its elements are `inner` apart.
#include "backends/cpu/ops/softmax/softmax.hpp"

#include "backends/cpu/threads.hpp"

#include <cmath>
#include <limits>

namespace mensurflow::cpu {

void softmax_f32(const float* x, float* out, std::int64_t outer, std::int64_t length,
                 std::int64_t inner) {
    const std::int64_t lines = outer * inner;
    MENSURFLOW_PARALLEL_FOR(if (lines * length > parallel_threshold))
    for (std::int64_t line = 0; line < lines; ++line) {
        const std::int64_t base = (line / inner) * length * inner + line % inner;
        const float* in = x + base;
        float* dst = out + base;
        // The maximum keeps a NaN once it has seen one, so a NaN poisons the whole line.
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = 0; j < length; ++j) {
            const float value = in[j * inner];
            if (std::isnan(value) || value > maximum) {
                maximum = value;
                if (std::isnan(value)) {
                    break;
                }
            }
        }
        if (maximum == -std::numeric_limits<float>::infinity()) {
            for (std::int64_t j = 0; j < length; ++j) {
                dst[j * inner] = 0.0f; // every element masked out
            }
            continue;
        }
        double sum = 0.0;
        for (std::int64_t j = 0; j < length; ++j) {
            const float e = std::exp(in[j * inner] - maximum);
            dst[j * inner] = e;
            sum += static_cast<double>(e);
        }
        const float scale = static_cast<float>(1.0 / sum);
        for (std::int64_t j = 0; j < length; ++j) {
            dst[j * inner] *= scale;
        }
    }
}

void log_softmax_f32(const float* x, float* out, std::int64_t outer, std::int64_t length,
                     std::int64_t inner) {
    const std::int64_t lines = outer * inner;
    MENSURFLOW_PARALLEL_FOR(if (lines * length > parallel_threshold))
    for (std::int64_t line = 0; line < lines; ++line) {
        const std::int64_t base = (line / inner) * length * inner + line % inner;
        const float* in = x + base;
        float* dst = out + base;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = 0; j < length; ++j) {
            const float value = in[j * inner];
            if (std::isnan(value) || value > maximum) {
                maximum = value;
                if (std::isnan(value)) {
                    break;
                }
            }
        }
        if (maximum == -std::numeric_limits<float>::infinity()) {
            for (std::int64_t j = 0; j < length; ++j) {
                dst[j * inner] = -std::numeric_limits<float>::infinity();
            }
            continue;
        }
        double sum = 0.0;
        for (std::int64_t j = 0; j < length; ++j) {
            sum += static_cast<double>(std::exp(in[j * inner] - maximum));
        }
        // out = x - logsumexp = (x - max) - log(sum). Subtracting the maximum first matters: x - max
        // is exact for nearby values, while max + log(sum) rounded to float loses about 3e-5 when
        // the logits are near 1000.
        const float log_sum = static_cast<float>(std::log(sum));
        for (std::int64_t j = 0; j < length; ++j) {
            dst[j * inner] = (in[j * inner] - maximum) - log_sum;
        }
    }
}

void softmax_backward_f32(const float* y, const float* gradient, float* out, std::int64_t outer,
                          std::int64_t length, std::int64_t inner) {
    const std::int64_t lines = outer * inner;
    MENSURFLOW_PARALLEL_FOR(if (lines * length > parallel_threshold))
    for (std::int64_t line = 0; line < lines; ++line) {
        const std::int64_t base = (line / inner) * length * inner + line % inner;
        double dot = 0.0;
        for (std::int64_t j = 0; j < length; ++j) {
            const std::int64_t at = base + j * inner;
            dot += static_cast<double>(gradient[at]) * static_cast<double>(y[at]);
        }
        const float shift = static_cast<float>(dot);
        for (std::int64_t j = 0; j < length; ++j) {
            const std::int64_t at = base + j * inner;
            out[at] = y[at] * (gradient[at] - shift);
        }
    }
}

void log_softmax_backward_f32(const float* y, const float* gradient, float* out, std::int64_t outer,
                              std::int64_t length, std::int64_t inner) {
    const std::int64_t lines = outer * inner;
    MENSURFLOW_PARALLEL_FOR(if (lines * length > parallel_threshold))
    for (std::int64_t line = 0; line < lines; ++line) {
        const std::int64_t base = (line / inner) * length * inner + line % inner;
        double total = 0.0;
        for (std::int64_t j = 0; j < length; ++j) {
            total += static_cast<double>(gradient[base + j * inner]);
        }
        const float sum = static_cast<float>(total);
        for (std::int64_t j = 0; j < length; ++j) {
            const std::int64_t at = base + j * inner;
            out[at] = gradient[at] - std::exp(y[at]) * sum;
        }
    }
}

} // namespace mensurflow::cpu
