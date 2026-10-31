// Obviously-correct single-threaded versions of the elementwise ops. Optimized kernels must match
// these bit for bit, since elementwise float arithmetic has no reordering to excuse differences.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

template <typename Op>
void binary(const float* a, const float* b, float* out, std::int64_t n, Op op) {
    for (std::int64_t i = 0; i < n; ++i) {
        out[i] = op(a[i], b[i]);
    }
}

inline void add(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x + y; });
}

inline void sub(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x - y; });
}

inline void mul(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x * y; });
}

inline void div(const float* a, const float* b, float* out, std::int64_t n) {
    binary(a, b, out, n, [](float x, float y) { return x / y; });
}

} // namespace reference

// Broadcasting binary op written from scratch, so tests do not rely on the library's own
// broadcast_shapes/expand. Shapes are NumPy-compatible; values are row-major.
namespace reference {

struct Broadcasted {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

template <typename Op>
Broadcasted broadcast_binary(const std::vector<float>& a, const std::vector<std::int64_t>& a_shape,
                             const std::vector<float>& b, const std::vector<std::int64_t>& b_shape,
                             Op op) {
    const std::size_t nd = std::max(a_shape.size(), b_shape.size());
    // Size of dimension d (counted from the left of the output) for one operand, 1 if missing.
    auto dim_of = [nd](const std::vector<std::int64_t>& s, std::size_t d) {
        const std::size_t lead = nd - s.size();
        return d < lead ? std::int64_t{1} : s[d - lead];
    };
    Broadcasted out;
    std::int64_t total = 1;
    for (std::size_t d = 0; d < nd; ++d) {
        // A size-1 dimension stretches to the other operand's size, which may be 0.
        const std::int64_t da = dim_of(a_shape, d), db = dim_of(b_shape, d);
        out.shape.push_back(da == 1 ? db : da);
        total *= out.shape.back();
    }
    for (std::int64_t flat = 0; flat < total; ++flat) {
        // Unravel the flat output index, then re-ravel it into each operand, pinning size-1 dims.
        std::int64_t rest = flat, ia = 0, ib = 0;
        std::int64_t stride_a = 1, stride_b = 1;
        for (std::size_t d = nd; d-- > 0;) {
            const std::int64_t index = rest % out.shape[d];
            rest /= out.shape[d];
            ia += (dim_of(a_shape, d) == 1 ? 0 : index) * stride_a;
            ib += (dim_of(b_shape, d) == 1 ? 0 : index) * stride_b;
            stride_a *= dim_of(a_shape, d);
            stride_b *= dim_of(b_shape, d);
        }
        out.values.push_back(op(a[static_cast<std::size_t>(ia)], b[static_cast<std::size_t>(ib)]));
    }
    return out;
}

} // namespace reference
