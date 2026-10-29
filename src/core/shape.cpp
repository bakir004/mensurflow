// Shape validation, stride computation, and broadcasting rules.
#include "mensurflow/shape.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mensurflow {

void validate_shape(const Shape& shape) {
    std::int64_t count = 1;
    for (std::int64_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("negative dimension in shape " + to_string(shape));
        }
        // Check before multiplying so the overflow itself never happens.
        if (dim != 0 && count > std::numeric_limits<std::int64_t>::max() / dim) {
            throw std::invalid_argument("element count overflows in shape " + to_string(shape));
        }
        count *= dim;
    }
}

std::int64_t numel(const Shape& shape) {
    std::int64_t count = 1;
    for (std::int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

Strides contiguous_strides(const Shape& shape) {
    Strides strides(shape.size());
    std::int64_t step = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = step;
        step *= shape[i];
    }
    return strides;
}

bool is_contiguous(const Shape& shape, const Strides& strides) {
    if (shape.size() != strides.size()) {
        return false;
    }
    if (numel(shape) == 0) {
        return true; // No elements, so no memory is touched.
    }
    std::int64_t expected = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        if (shape[i] != 1 && strides[i] != expected) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    const std::size_t ndim = std::max(a.size(), b.size());
    Shape out(ndim);
    // Walk from the last dimension; a missing leading dimension acts as size 1.
    for (std::size_t i = 0; i < ndim; ++i) {
        const std::int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
        const std::int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("cannot broadcast shapes " + to_string(a) + " and " +
                                        to_string(b));
        }
        out[ndim - 1 - i] = da == 1 ? db : da;
    }
    return out;
}

Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target) {
    if (shape.size() > target.size() || shape.size() != strides.size()) {
        throw std::invalid_argument("cannot broadcast " + to_string(shape) + " to " +
                                    to_string(target));
    }
    const std::size_t lead = target.size() - shape.size();
    Strides out(target.size(), 0); // New leading dimensions repeat the whole tensor.
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const std::int64_t dim = shape[i];
        const std::int64_t want = target[lead + i];
        if (dim == want) {
            out[lead + i] = strides[i];
        } else if (dim != 1) {
            throw std::invalid_argument("cannot broadcast " + to_string(shape) + " to " +
                                        to_string(target));
        }
        // dim == 1 and want != 1: stride stays 0, so every index reads the same element.
    }
    return out;
}

int normalize_dim(int dim, int ndim) {
    const int out = dim < 0 ? dim + ndim : dim;
    if (out < 0 || out >= ndim) {
        throw std::out_of_range("dim " + std::to_string(dim) + " out of range for " +
                                std::to_string(ndim) + "-d tensor");
    }
    return out;
}

std::string to_string(const Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

} // namespace mensurflow
