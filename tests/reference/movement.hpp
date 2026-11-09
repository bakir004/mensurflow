// Naive references for data movement: element by element through full multi-indices, slow and
// obviously correct. The kernels are compared against these bit for bit, since they only copy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

using Dims = std::vector<std::int64_t>;

inline std::int64_t count(const Dims& shape) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        n *= d;
    }
    return n;
}

// Row-major multi-index of flat position `flat` in `shape`.
inline Dims unravel(std::int64_t flat, const Dims& shape) {
    Dims index(shape.size());
    for (std::size_t d = shape.size(); d-- > 0;) {
        index[d] = shape[d] == 0 ? 0 : flat % shape[d];
        flat = shape[d] == 0 ? 0 : flat / shape[d];
    }
    return index;
}

inline std::int64_t ravel(const Dims& index, const Dims& shape) {
    std::int64_t flat = 0;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        flat = flat * shape[d] + index[d];
    }
    return flat;
}

// concat along `axis`: each output element is looked up in the input whose range along the axis
// holds its coordinate.
inline std::vector<float> concat(const std::vector<std::vector<float>>& inputs,
                                 const std::vector<Dims>& shapes, int axis) {
    Dims out_shape = shapes[0];
    out_shape[static_cast<std::size_t>(axis)] = 0;
    for (const Dims& s : shapes) {
        out_shape[static_cast<std::size_t>(axis)] += s[static_cast<std::size_t>(axis)];
    }
    std::vector<float> out(static_cast<std::size_t>(count(out_shape)));
    for (std::int64_t flat = 0; flat < count(out_shape); ++flat) {
        Dims index = unravel(flat, out_shape);
        std::size_t t = 0;
        while (index[static_cast<std::size_t>(axis)] >= shapes[t][static_cast<std::size_t>(axis)]) {
            index[static_cast<std::size_t>(axis)] -= shapes[t][static_cast<std::size_t>(axis)];
            ++t;
        }
        out[static_cast<std::size_t>(flat)] =
            inputs[t][static_cast<std::size_t>(ravel(index, shapes[t]))];
    }
    return out;
}

// Constant pad with (before, after) per dimension, in dimension order.
inline std::vector<float> pad(const std::vector<float>& x, const Dims& shape, const Dims& before,
                              const Dims& after, float value) {
    Dims out_shape = shape;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        out_shape[d] += before[d] + after[d];
    }
    std::vector<float> out(static_cast<std::size_t>(count(out_shape)), value);
    for (std::int64_t flat = 0; flat < count(shape); ++flat) {
        Dims index = unravel(flat, shape);
        for (std::size_t d = 0; d < shape.size(); ++d) {
            index[d] += before[d];
        }
        out[static_cast<std::size_t>(ravel(index, out_shape))] = x[static_cast<std::size_t>(flat)];
    }
    return out;
}

} // namespace reference
