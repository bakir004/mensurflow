// CPU losses: one parallel loop per kernel. With no reduction each thread writes its own rows or
// elements. With sum or mean the loop runs over fixed chunks, each chunk's sum goes to its own
// slot, and the slots are added in order afterwards: the same additions in the same order at
// every thread count, so the loss is bit-identical whether it ran on 1 or 11 threads.
#include "backends/cpu/ops/loss/loss.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mensurflow::cpu {

namespace {

constexpr int reduce_none = 0;
constexpr int reduce_mean = 2;

// Elements per chunk of a reduced loss: big enough that the chunk sums are a small serial tail.
constexpr std::int64_t chunk_elements = 1 << 14;

} // namespace

void cross_entropy_f32(const float* logits, const std::int64_t* labels, float* out,
                       std::int64_t rows, std::int64_t classes, int reduction) {
    // Rows per chunk so a chunk holds about chunk_elements logits; at least one row.
    const std::int64_t rows_per_chunk = std::max<std::int64_t>(1, chunk_elements / classes);
    const std::int64_t chunks = (rows + rows_per_chunk - 1) / rows_per_chunk;
    std::vector<double> partial(reduction == reduce_none ? 0 : static_cast<std::size_t>(chunks));
    MENSURFLOW_PARALLEL_FOR(if (rows * classes > parallel_threshold))
    for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        const std::int64_t end = std::min(rows, (chunk + 1) * rows_per_chunk);
        double chunk_sum = 0.0;
        for (std::int64_t row = chunk * rows_per_chunk; row < end; ++row) {
            const float* x = logits + row * classes;
            float maximum = x[0];
            for (std::int64_t j = 1; j < classes; ++j) {
                maximum = x[j] > maximum || std::isnan(x[j]) ? x[j] : maximum;
            }
            double sum = 0.0;
            for (std::int64_t j = 0; j < classes; ++j) {
                sum += static_cast<double>(std::exp(x[j] - maximum));
            }
            // logsumexp - x[label], with the maximum taken out of both terms.
            const double loss = std::log(sum) - static_cast<double>(x[labels[row]] - maximum);
            if (reduction == reduce_none) {
                out[row] = static_cast<float>(loss);
            } else {
                chunk_sum += loss;
            }
        }
        if (reduction != reduce_none) {
            partial[static_cast<std::size_t>(chunk)] = chunk_sum;
        }
    }
    if (reduction != reduce_none) {
        double total = 0.0;
        for (double value : partial) {
            total += value;
        }
        out[0] = static_cast<float>(reduction == reduce_mean ? total / static_cast<double>(rows)
                                                             : total);
    }
}

void cross_entropy_backward_f32(const float* logits, const std::int64_t* labels,
                                const float* gradient, float* out, std::int64_t rows,
                                std::int64_t classes, int reduction) {
    // One upstream value for sum and mean. Read only then: with no reduction an empty batch has
    // no gradient values at all.
    const float shared = reduction == reduce_none   ? 0.0f
                         : reduction == reduce_mean ? gradient[0] / static_cast<float>(rows)
                                                    : gradient[0];
    MENSURFLOW_PARALLEL_FOR(if (rows * classes > parallel_threshold))
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* x = logits + row * classes;
        float* dx = out + row * classes;
        const float upstream = reduction == reduce_none ? gradient[row] : shared;
        float maximum = x[0];
        for (std::int64_t j = 1; j < classes; ++j) {
            maximum = x[j] > maximum || std::isnan(x[j]) ? x[j] : maximum;
        }
        double sum = 0.0;
        for (std::int64_t j = 0; j < classes; ++j) {
            const float e = std::exp(x[j] - maximum);
            dx[j] = e;
            sum += static_cast<double>(e);
        }
        const float scale = static_cast<float>(1.0 / sum) * upstream;
        for (std::int64_t j = 0; j < classes; ++j) {
            dx[j] *= scale; // softmax * upstream
        }
        dx[labels[row]] -= upstream; // minus the one-hot label
    }
}

void mse_loss_f32(const float* x, const float* target, float* out, std::int64_t n, int reduction) {
    if (reduction == reduce_none) {
        // Memory-bound like a copy: threads only pay off from the larger cutoff (see below).
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
        for (std::int64_t i = 0; i < n; ++i) {
            const float d = x[i] - target[i];
            out[i] = d * d;
        }
        return;
    }
    const std::int64_t chunks = (n + chunk_elements - 1) / chunk_elements;
    std::vector<double> partial(static_cast<std::size_t>(chunks));
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        const std::int64_t end = std::min(n, (chunk + 1) * chunk_elements);
        double sum = 0.0;
        for (std::int64_t i = chunk * chunk_elements; i < end; ++i) {
            const double d = static_cast<double>(x[i]) - static_cast<double>(target[i]);
            sum += d * d;
        }
        partial[static_cast<std::size_t>(chunk)] = sum;
    }
    double total = 0.0;
    for (double value : partial) {
        total += value;
    }
    out[0] = static_cast<float>(reduction == reduce_mean ? total / static_cast<double>(n) : total);
}

// One multiply-subtract per element, so it is limited by memory like a copy. With the 32,768
// cutoff, 2 to 11 threads were 1.6 to 4 times slower than one thread at 65,536 elements on an
// M3 Pro (0.0067 ms on one thread), so it uses the 262,144 cutoff of the matrix kernels, as concat
// and pad do.
void mse_loss_backward_f32(const float* x, const float* target, const float* gradient, float* out,
                           std::int64_t n, int reduction) {
    if (reduction == reduce_none) {
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
        for (std::int64_t i = 0; i < n; ++i) {
            out[i] = 2.0f * (x[i] - target[i]) * gradient[i];
        }
        return;
    }
    const float scale =
        2.0f * (reduction == reduce_mean ? gradient[0] / static_cast<float>(n) : gradient[0]);
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = (x[i] - target[i]) * scale;
    }
}

void bce_with_logits_f32(const float* x, const float* target, float* out, std::int64_t n,
                         int reduction) {
    if (reduction == reduce_none) {
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
        for (std::int64_t i = 0; i < n; ++i) {
            const float v = x[i];
            out[i] = std::max(v, 0.0f) - v * target[i] + std::log1p(std::exp(-std::abs(v)));
        }
        return;
    }
    const std::int64_t chunks = (n + chunk_elements - 1) / chunk_elements;
    std::vector<double> partial(static_cast<std::size_t>(chunks));
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        const std::int64_t end = std::min(n, (chunk + 1) * chunk_elements);
        double sum = 0.0;
        for (std::int64_t i = chunk * chunk_elements; i < end; ++i) {
            const float v = x[i];
            sum += static_cast<double>(std::max(v, 0.0f) - v * target[i] +
                                       std::log1p(std::exp(-std::abs(v))));
        }
        partial[static_cast<std::size_t>(chunk)] = sum;
    }
    double total = 0.0;
    for (double value : partial) {
        total += value;
    }
    out[0] = static_cast<float>(reduction == reduce_mean ? total / static_cast<double>(n) : total);
}

void bce_with_logits_backward_f32(const float* x, const float* target, const float* gradient,
                                  float* out, std::int64_t n, int reduction) {
    const float shared = reduction == reduce_none   ? 0.0f // see cross_entropy_backward_f32
                         : reduction == reduce_mean ? gradient[0] / static_cast<float>(n)
                                                    : gradient[0];
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold))
    for (std::int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        // sigmoid(v) - t without cancellation. With e = exp(-|v|) <= 1 (never overflows) and
        // p = e / (1 + e) < 0.5: for v >= 0, sigmoid = 1 - p, so sigmoid - t = (1 - t) - p, exact
        // for t = 1 where the plain form would subtract two numbers near 1; for v < 0, sigmoid = p.
        const float e = std::exp(-std::abs(v));
        const float p = e / (1.0f + e);
        const float difference = v >= 0.0f ? (1.0f - target[i]) - p : p - target[i];
        out[i] = difference * (reduction == reduce_none ? gradient[i] : shared);
    }
}

} // namespace mensurflow::cpu
