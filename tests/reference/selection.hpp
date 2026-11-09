// Independent comparison and selection references. Broadcasting uses logical indices rather
// than Tensor::expand/contiguous or the library's shape helpers.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace reference {

template <typename Predicate>
void comparison(const float* a, const float* b, std::uint8_t* out, std::int64_t n,
                Predicate predicate) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = predicate(a[i], b[i]) ? std::uint8_t{1} : std::uint8_t{0};
    }
}

inline void eq(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison(a, b, out, n, [](float x, float y) { return x == y; });
}
inline void ne(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison(a, b, out, n, [](float x, float y) { return x != y; });
}
inline void lt(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison(a, b, out, n, [](float x, float y) { return x < y; });
}
inline void le(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison(a, b, out, n, [](float x, float y) { return x <= y; });
}
inline void gt(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison(a, b, out, n, [](float x, float y) { return x > y; });
}
inline void ge(const float* a, const float* b, std::uint8_t* out, std::int64_t n) {
    comparison(a, b, out, n, [](float x, float y) { return x >= y; });
}

inline void where(const std::uint8_t* condition, const float* x, const float* y, float* out,
                  std::int64_t n) {
    for (std::int64_t i = 0; i < n; ++i) {
        // Selecting a value preserves its bits; arithmetic masking would corrupt NaNs and zeros.
        const float* chosen = condition[i] != 0 ? x + i : y + i;
        std::memcpy(out + i, chosen, sizeof(float));
    }
}

namespace selection {

using Shape = std::vector<std::int64_t>;

inline std::size_t count(const Shape& shape) {
    std::size_t result = 1;
    for (auto dimension : shape) {
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

inline Shape broadcast(std::initializer_list<Shape> shapes) {
    std::size_t rank = 0;
    for (const auto& shape : shapes) {
        rank = std::max(rank, shape.size());
    }
    Shape output(rank, 1);
    for (const auto& shape : shapes) {
        const std::size_t leading = rank - shape.size();
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
            auto& target = output[leading + dimension];
            const auto source = shape[dimension];
            if (target == 1) {
                target = source;
            } else if (source != 1 && source != target) {
                throw std::invalid_argument("reference: incompatible broadcast shapes");
            }
        }
    }
    return output;
}

inline std::size_t input_index(std::size_t flat, const Shape& output, const Shape& input) {
    std::size_t index = 0, stride = 1;
    const std::size_t leading = output.size() - input.size();
    for (std::size_t dimension = output.size(); dimension-- > 0;) {
        const auto coordinate = flat % static_cast<std::size_t>(output[dimension]);
        flat /= static_cast<std::size_t>(output[dimension]);
        if (dimension >= leading) {
            const auto size = static_cast<std::size_t>(input[dimension - leading]);
            index += (size == 1 ? 0 : coordinate) * stride;
            stride *= size;
        }
    }
    return index;
}

struct Compared {
    Shape shape;
    std::vector<std::uint8_t> values;
};

template <typename Predicate>
Compared broadcast_compare(const std::vector<float>& a, const Shape& a_shape,
                           const std::vector<float>& b, const Shape& b_shape,
                           Predicate predicate) {
    Compared result{broadcast({a_shape, b_shape}), {}};
    result.values.resize(count(result.shape));
    for (std::size_t flat = 0; flat < result.values.size(); ++flat) {
        result.values[flat] = predicate(a[input_index(flat, result.shape, a_shape)],
                                        b[input_index(flat, result.shape, b_shape)])
                                  ? std::uint8_t{1}
                                  : std::uint8_t{0};
    }
    return result;
}

struct Selected {
    Shape shape;
    std::vector<float> values;
};

inline Selected broadcast_where(const std::vector<std::uint8_t>& condition,
                                const Shape& condition_shape, const std::vector<float>& x,
                                const Shape& x_shape, const std::vector<float>& y,
                                const Shape& y_shape) {
    Selected result{broadcast({condition_shape, x_shape, y_shape}), {}};
    result.values.resize(count(result.shape));
    for (std::size_t flat = 0; flat < result.values.size(); ++flat) {
        const auto mask = condition[input_index(flat, result.shape, condition_shape)];
        const float* chosen = mask != 0 ? &x[input_index(flat, result.shape, x_shape)]
                                       : &y[input_index(flat, result.shape, y_shape)];
        std::memcpy(&result.values[flat], chosen, sizeof(float));
    }
    return result;
}

} // namespace selection
} // namespace reference
