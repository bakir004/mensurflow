// Data movement: joining, cutting and padding tensors. The last argument says where an op runs (see
// execution.hpp); the result is on the inputs' device.
#pragma once

#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <cstdint>
#include <vector>

namespace mensurflow {

// Joins tensors along `axis` (negative counts from the end) into a new contiguous tensor, like
// torch.cat. All inputs need the same rank, dtype and device, and the same size in every dimension
// except `axis`; inputs that are empty along `axis` are allowed. Views are accepted. Only float32,
// and only on the CPU so far. Not differentiable yet.
// Throws invalid_argument for an empty list, an undefined tensor, a 0-d tensor, an axis out of
// range, or mismatched ranks, sizes, dtypes or devices.
Tensor concat(const std::vector<Tensor>& tensors, int axis = 0,
              Execution execution = Execution::Auto);

// Cuts x along `axis` into pieces of `size` elements (the last piece may be smaller), like
// torch.split. The pieces are views that share x's memory: nothing is copied, so this costs the
// same at any size; call contiguous() on a piece that needs its own packed memory.
// Throws invalid_argument for a 0-d tensor, an axis out of range, or a size below 1.
std::vector<Tensor> split(const Tensor& x, std::int64_t size, int axis = 0);

// The same with one size per piece; the sizes must be non-negative and add up to x's size along
// `axis`. Throws invalid_argument otherwise.
std::vector<Tensor> split(const Tensor& x, const std::vector<std::int64_t>& sizes, int axis = 0);

// Constant padding, like torch.nn.functional.pad in "constant" mode: `padding` lists
// (before, after) pairs starting from the LAST dimension, so {1, 2} pads the last dimension with 1
// element before and 2 after, and {1, 2, 3, 4} also pads the dimension before it with 3 and 4. It
// may cover fewer dimensions than x has. The new elements are `value`; the result is a new
// contiguous tensor. Only float32, and only on the CPU so far. Not differentiable yet.
// Throws invalid_argument for an odd-length or too long padding list, a negative amount (cropping
// is not supported), or an undefined tensor.
Tensor pad(const Tensor& x, const std::vector<std::int64_t>& padding, float value = 0.0f,
           Execution execution = Execution::Auto);

// Select elements along `axis` using an I64 index tensor. `gather` requires indices to have the
// same rank as x; non-axis dimensions must not exceed x. Indices must be in [0, x.size(axis)).
Tensor gather(const Tensor& x, const Tensor& indices, int axis = 0,
              Execution execution = Execution::Auto);

// Select complete slices along `axis` using a one-dimensional I64 index tensor.
Tensor index_select(const Tensor& x, const Tensor& indices, int axis = 0,
                    Execution execution = Execution::Auto);

} // namespace mensurflow
