// Shape and stride math shared by Tensor and every op's validation.
// Strides are counted in elements (not bytes), like PyTorch.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mensurflow {

// Size of each dimension, outermost first. An empty Shape is a scalar (one element).
using Shape = std::vector<std::int64_t>;

// Element step to move one index along each dimension. Stride 0 repeats data (broadcast views).
using Strides = std::vector<std::int64_t>;

// Throws invalid_argument if any dimension is negative or the element count overflows.
void validate_shape(const Shape& shape);

// Number of elements; 1 for a scalar, 0 if any dimension is 0.
std::int64_t numel(const Shape& shape);

// Row-major strides: the last dimension has stride 1.
Strides contiguous_strides(const Shape& shape);

// True if the strides walk memory in row-major order with no gaps.
// Dimensions of size 1 may have any stride, since they are never stepped along.
bool is_contiguous(const Shape& shape, const Strides& strides);

// NumPy broadcasting: align from the right; sizes must match or one of them must be 1.
// Throws invalid_argument for incompatible shapes.
Shape broadcast_shapes(const Shape& a, const Shape& b);

// Strides that view a tensor of `shape` as the larger `target` shape without copying:
// broadcast dimensions get stride 0. Requires broadcast_shapes(shape, target) == target.
Strides broadcast_strides(const Shape& shape, const Strides& strides, const Shape& target);

// Map a possibly negative dim (-1 = last) into [0, ndim). Throws out_of_range.
int normalize_dim(int dim, int ndim);

// "[2, 3, 4]" for error messages.
std::string to_string(const Shape& shape);

} // namespace mensurflow
