// CPU data-movement kernels: concat and constant pad. One parallel loop each.
#include "backends/cpu/ops/movement/movement.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace mensurflow::cpu {

void strided_copy_f32(const float* x, float* out, const std::int64_t* shape,
                      const std::int64_t* strides, int ndim) {
    std::int64_t n = 1;
    for (int d = 0; d < ndim; ++d) {
        n *= shape[d];
    }
    if (n == 0) {
        return;
    }

    const bool contiguous_rows = ndim > 0 && strides[ndim - 1] == 1;
    const std::int64_t row = contiguous_rows ? shape[ndim - 1] : 1;
    const std::int64_t rows = n / row;
    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t linear = 0; linear < rows; ++linear) {
        std::int64_t source = 0;
        std::int64_t index = linear;
        const int walked = contiguous_rows ? ndim - 1 : ndim;
        for (int d = walked - 1; d >= 0; --d) {
            const std::int64_t coordinate = index % shape[d];
            index /= shape[d];
            source += coordinate * strides[d];
        }
        if (contiguous_rows) {
            std::memcpy(out + linear * row, x + source,
                        static_cast<std::size_t>(row) * sizeof(float));
        } else {
            out[linear] = x[source];
        }
    }
}

void transpose_f32(const float* x, float* out, std::int64_t rows, std::int64_t cols) {
    constexpr std::int64_t tile = 32;
    const std::int64_t tile_rows = (rows + tile - 1) / tile;
    const std::int64_t tile_cols = (cols + tile - 1) / tile;
    const std::int64_t tiles = tile_rows * tile_cols;
    MENSURFLOW_PARALLEL_FOR(if (rows * cols > parallel_work_threshold))
    for (std::int64_t block = 0; block < tiles; ++block) {
        const std::int64_t row0 = (block / tile_cols) * tile;
        const std::int64_t col0 = (block % tile_cols) * tile;
        const std::int64_t row_end = std::min(row0 + tile, rows);
        const std::int64_t col_end = std::min(col0 + tile, cols);
        float values[tile][tile];
        for (std::int64_t i = row0; i < row_end; ++i) {
            for (std::int64_t j = col0; j < col_end; ++j) {
                values[i - row0][j - col0] = x[i * cols + j];
            }
        }
        for (std::int64_t j = col0; j < col_end; ++j) {
            for (std::int64_t i = row0; i < row_end; ++i) {
                out[j * rows + i] = values[i - row0][j - col0];
            }
        }
    }
}

void gather_f32(const float* x, const std::int64_t* indices, float* out,
                const std::int64_t* input_shape, const std::int64_t* input_strides,
                const std::int64_t* output_shape, int ndim, int axis) {
    std::int64_t n = 1;
    for (int d = 0; d < ndim; ++d) {
        n *= output_shape[d];
    }

    // For the common contiguous case, all dimensions except the selected one match. Each
    // output row then copies a contiguous run, avoiding per-element coordinate reconstruction.
    bool matching_rows = axis == ndim - 1 && input_strides[ndim - 1] == 1;
    for (int d = 0; d < ndim; ++d) {
        if (d != axis && input_shape[d] != output_shape[d]) {
            matching_rows = false;
        }
    }
    if (matching_rows) {
        std::int64_t outer = 1;
        std::int64_t inner = 1;
        for (int d = 0; d < axis; ++d) {
            outer *= input_shape[d];
        }
        for (int d = axis + 1; d < ndim; ++d) {
            inner *= input_shape[d];
        }
        const std::int64_t input_axis = input_shape[axis];
        const std::int64_t output_axis = output_shape[axis];
        const std::int64_t jobs = outer * output_axis;
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
        for (std::int64_t job = 0; job < jobs; ++job) {
            const std::int64_t outer_index = job / output_axis;
            const std::int64_t source_index = indices[job * inner];
            const std::int64_t source =
                (outer_index * input_axis + source_index) * inner;
            std::memcpy(out + job * inner, x + source,
                        static_cast<std::size_t>(inner) * sizeof(float));
        }
        return;
    }

    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t linear = 0; linear < n; ++linear) {
        std::int64_t rest = linear;
        std::int64_t source = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const std::int64_t coordinate = rest % output_shape[d];
            rest /= output_shape[d];
            source += (d == axis ? indices[linear] : coordinate) * input_strides[d];
        }
        out[linear] = x[source];
    }
}

void index_select_f32(const float* x, const std::int64_t* indices, float* out,
                      const std::int64_t* input_shape, const std::int64_t* input_strides,
                      int ndim, int axis, std::int64_t index_count) {
    std::int64_t n = 1;
    for (int d = 0; d < ndim; ++d) {
        n *= d == axis ? index_count : input_shape[d];
    }

    // A contiguous tensor can be viewed as outer rows of axis-sized blocks followed by inner
    // elements. Copying whole inner runs removes the division/modulo work from the hot loop.
    if (input_strides[ndim - 1] == 1) {
        std::int64_t outer = 1;
        std::int64_t inner = 1;
        for (int d = 0; d < axis; ++d) {
            outer *= input_shape[d];
        }
        for (int d = axis + 1; d < ndim; ++d) {
            inner *= input_shape[d];
        }
        const std::int64_t jobs = outer * index_count;
        MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
        for (std::int64_t job = 0; job < jobs; ++job) {
            const std::int64_t outer_index = job / index_count;
            const std::int64_t selected = job % index_count;
            const std::int64_t source =
                (outer_index * input_shape[axis] + indices[selected]) * inner;
            std::memcpy(out + job * inner, x + source,
                        static_cast<std::size_t>(inner) * sizeof(float));
        }
        return;
    }

    MENSURFLOW_PARALLEL_FOR(if (n > parallel_work_threshold))
    for (std::int64_t linear = 0; linear < n; ++linear) {
        std::int64_t rest = linear;
        std::int64_t source = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const std::int64_t size = d == axis ? index_count : input_shape[d];
            const std::int64_t coordinate = rest % size;
            rest /= size;
            source += (d == axis ? indices[coordinate] : coordinate) * input_strides[d];
        }
        out[linear] = x[source];
    }
}

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
