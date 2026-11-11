#include "backends/cpu/ops/indexing/indexing.hpp"

#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace mensurflow::cpu {

void one_hot_i64(const std::int64_t* indices, std::int64_t* out, std::int64_t count,
                  std::int64_t classes) {
    // Fill contiguous chunks, then set their selected class. Division happens per chunk rather
    // than per output element, and even one row with a large class count can use several threads.
    constexpr std::int64_t block = 1024;
    const std::int64_t blocks = classes / block + (classes % block != 0);
    const std::int64_t tasks = count * blocks;
    MENSURFLOW_PARALLEL_FOR(if (count * classes > parallel_threshold))
    for (std::int64_t task = 0; task < tasks; ++task) {
        const std::int64_t row = task / blocks;
        const std::int64_t begin = (task % blocks) * block;
        const std::int64_t end = begin + std::min(block, classes - begin);
        std::int64_t* output_row = out + row * classes;
        std::fill(output_row + begin, output_row + end, std::int64_t{0});
        const std::int64_t selected = indices[row];
        if (selected >= begin && selected < end) {
            output_row[selected] = 1;
        }
    }
}

void embedding_f32(const float* weight, const std::int64_t* indices, float* out,
                   std::int64_t count, std::int64_t width) {
    if (count == 0 || width == 0) {
        return;
    }
    // Copy long rows in pieces so a single very wide lookup can still use multiple threads.
    constexpr std::int64_t block = 1024;
    const std::int64_t blocks = width / block + (width % block != 0);
    const std::int64_t tasks = count * blocks;
    MENSURFLOW_PARALLEL_FOR(if (count * width > parallel_threshold))
    for (std::int64_t task = 0; task < tasks; ++task) {
        const std::int64_t row = task / blocks;
        const std::int64_t begin = (task % blocks) * block;
        const std::int64_t length = std::min(block, width - begin);
        std::memcpy(out + row * width + begin, weight + indices[row] * width + begin,
                    static_cast<std::size_t>(length) * sizeof(float));
    }
}

void embedding_backward_f32(const float* gradient, const std::int64_t* indices, float* out,
                            std::int64_t count, std::int64_t rows, std::int64_t width) {
    if (rows == 0 || width == 0) {
        return;
    }
    // Count and prefix-sum the IDs, then stably bucket their input positions by output row.
    std::vector<std::int64_t> starts(static_cast<std::size_t>(rows) + 1, 0);
    for (std::int64_t i = 0; i < count; ++i) {
        ++starts[static_cast<std::size_t>(indices[i]) + 1];
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        starts[static_cast<std::size_t>(row) + 1] += starts[static_cast<std::size_t>(row)];
    }
    std::vector<std::int64_t> cursor(starts.begin(), starts.end() - 1);
    std::vector<std::int64_t> positions(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        positions[static_cast<std::size_t>(cursor[static_cast<std::size_t>(indices[i])]++)] = i;
    }

    constexpr std::int64_t block = 64;
    const std::int64_t blocks = width / block + (width % block != 0);
    const std::int64_t tasks = rows * blocks;
    MENSURFLOW_PARALLEL_FOR(if (count * width > parallel_threshold ||
                               rows * width > parallel_threshold))
    for (std::int64_t task = 0; task < tasks; ++task) {
        const std::int64_t row = task / blocks;
        const std::int64_t begin = (task % blocks) * block;
        const std::int64_t end = begin + std::min(block, width - begin);
        float* output_row = out + row * width;
        std::fill(output_row + begin, output_row + end, 0.0f);
        for (std::int64_t j = starts[static_cast<std::size_t>(row)];
             j < starts[static_cast<std::size_t>(row) + 1]; ++j) {
            const float* input_row = gradient + positions[static_cast<std::size_t>(j)] * width;
            for (std::int64_t column = begin; column < end; ++column) {
                output_row[column] += input_row[column];
            }
        }
    }
}

} // namespace mensurflow::cpu
