// CPU data-movement kernels: concat and constant pad. One parallel loop each.
#include "backends/cpu/ops/movement/movement.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace mensurflow::cpu {

// Both kernels only copy, so a thread finishes a small job faster than OpenMP can wake the others.
// With elementwise's cutoff of 32768 elements they ran 6 to 9 times slower than one thread just
// above it on an M3 Pro (a 64 x 1024 concat, a 181 x 181 pad) and were clearly faster from about a
// million, so they use the larger cutoff of the matrix kernels (262144 elements).

// The output is cut into jobs of at most `piece` floats, each lying inside one input row, so a row
// longer than a piece (concat along axis 0 has a single row per input) is still spread over the
// threads. Job j is row j / pieces_per_row; within the row, first_piece[t] is input t's first piece.
void concat_f32(const float* const* inputs, const std::int64_t* widths, std::int64_t count,
                float* out, std::int64_t outer) {
    constexpr std::int64_t piece = 1 << 14; // 64 KB of floats
    std::vector<std::int64_t> column(
        static_cast<std::size_t>(count)); // where input t starts in a row
    std::vector<std::int64_t> first_piece(static_cast<std::size_t>(count) +
                                          1); // prefix sum of pieces
    std::int64_t out_width = 0;
    for (std::int64_t t = 0; t < count; ++t) {
        column[static_cast<std::size_t>(t)] = out_width;
        out_width += widths[t];
        first_piece[static_cast<std::size_t>(t) + 1] =
            first_piece[static_cast<std::size_t>(t)] + (widths[t] + piece - 1) / piece;
    }
    const std::int64_t pieces_per_row = first_piece[static_cast<std::size_t>(count)];
    const std::int64_t jobs = outer * pieces_per_row;
    [[maybe_unused]] const std::int64_t total = outer * out_width; // read by the pragma only

    MENSURFLOW_PARALLEL_FOR(if (total > parallel_work_threshold))
    for (std::int64_t job = 0; job < jobs; ++job) {
        const std::int64_t row = job / pieces_per_row;
        const std::int64_t r = job % pieces_per_row;
        std::int64_t t = 0; // the input this piece belongs to; count is small, so a scan is enough
        while (first_piece[static_cast<std::size_t>(t) + 1] <= r) {
            ++t;
        }
        const std::int64_t width = widths[t];
        const std::int64_t begin = (r - first_piece[static_cast<std::size_t>(t)]) * piece;
        const std::int64_t length = std::min(piece, width - begin);
        std::memcpy(out + row * out_width + column[static_cast<std::size_t>(t)] + begin,
                    inputs[t] + row * width + begin,
                    static_cast<std::size_t>(length) * sizeof(float));
    }
}

// One output row (the last dimension) per iteration. The row's position in the other dimensions
// says whether it lies inside the input; if so it is [before fill | input row | after fill],
// otherwise all fill.
void pad_f32(const float* x, float* out, const std::int64_t* shape, const std::int64_t* before,
             const std::int64_t* after, int ndim, float value) {
    const int last = ndim - 1;
    std::vector<std::int64_t> out_shape(static_cast<std::size_t>(ndim));
    std::int64_t rows = 1;
    bool empty_input = false;
    for (int d = 0; d < ndim; ++d) {
        out_shape[static_cast<std::size_t>(d)] = shape[d] + before[d] + after[d];
        if (d < last) {
            rows *= out_shape[static_cast<std::size_t>(d)];
        }
        empty_input = empty_input || shape[d] == 0;
    }
    const std::int64_t width = out_shape[static_cast<std::size_t>(last)];
    const std::int64_t in_width = shape[last];
    const std::int64_t left = before[last];

    MENSURFLOW_PARALLEL_FOR(if (rows * width > parallel_work_threshold))
    for (std::int64_t row = 0; row < rows; ++row) {
        float* dst = out + row * width;
        // Walk the row index from the innermost outer dimension outwards, building the input row
        // index as we go; stop as soon as one coordinate falls in the padding.
        bool inside = !empty_input;
        std::int64_t rest = row;
        std::int64_t source_row = 0;
        std::int64_t scale = 1;
        for (int d = last - 1; d >= 0 && inside; --d) {
            const std::int64_t size = out_shape[static_cast<std::size_t>(d)];
            const std::int64_t coordinate = rest % size - before[d];
            rest /= size;
            inside = coordinate >= 0 && coordinate < shape[d];
            source_row += coordinate * scale;
            scale *= shape[d];
        }
        if (!inside) {
            for (std::int64_t j = 0; j < width; ++j) {
                dst[j] = value;
            }
            continue;
        }
        for (std::int64_t j = 0; j < left; ++j) {
            dst[j] = value;
        }
        std::memcpy(dst + left, x + source_row * in_width,
                    static_cast<std::size_t>(in_width) * sizeof(float));
        for (std::int64_t j = left + in_width; j < width; ++j) {
            dst[j] = value;
        }
    }
}

} // namespace mensurflow::cpu
