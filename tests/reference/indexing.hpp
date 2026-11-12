// Small scalar references: direct lookup and scatter-add in input order, without grouping.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

inline std::vector<std::int64_t> one_hot(const std::vector<std::int64_t>& indices,
                                       std::int64_t classes) {
    std::vector<std::int64_t> out(indices.size() * static_cast<std::size_t>(classes), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        out[i * static_cast<std::size_t>(classes) + static_cast<std::size_t>(indices[i])] = 1;
    }
    return out;
}

inline std::vector<float> embedding(const std::vector<float>& weight,
                                    const std::vector<std::int64_t>& indices, std::int64_t width) {
    std::vector<float> out;
    for (const std::int64_t index : indices) {
        for (std::int64_t column = 0; column < width; ++column) {
            out.push_back(weight[static_cast<std::size_t>(index * width + column)]);
        }
    }
    return out;
}

inline std::vector<float> embedding_backward(const std::vector<float>& gradient,
                                             const std::vector<std::int64_t>& indices,
                                             std::int64_t rows, std::int64_t width) {
    std::vector<float> out(static_cast<std::size_t>(rows * width), 0.0f);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        for (std::int64_t column = 0; column < width; ++column) {
            out[static_cast<std::size_t>(indices[i] * width + column)] +=
                gradient[i * static_cast<std::size_t>(width) + static_cast<std::size_t>(column)];
        }
    }
    return out;
}

} // namespace reference
