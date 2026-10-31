// Tensor: a typed, shaped view over shared device memory.
// Copying a Tensor is cheap and shares the same memory, like PyTorch; use an explicit copy
// op (added later) to duplicate data.
#pragma once

#include "mensurflow/device.hpp"
#include "mensurflow/dtype.hpp"
#include "mensurflow/shape.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace detail {
struct Storage;
struct TensorAccess;
} // namespace detail

class Tensor {
  public:
    // Undefined tensor with no storage; everything except defined() throws.
    Tensor() = default;

    // Uninitialized memory of the given shape on `device` (CPU unless told otherwise; use
    // default_device() to pick the best available). Throws runtime_error if the device is
    // unavailable.
    static Tensor empty(Shape shape, DType dtype = DType::F32, Device device = {});

    // Copies numel(shape) * dtype_size(dtype) bytes from contiguous host memory.
    static Tensor from_host(const void* data, Shape shape, DType dtype, Device device = {});

    template <typename T>
    static Tensor from_vector(const std::vector<T>& data, Shape shape, Device device = {}) {
        validate_shape(shape);
        if (static_cast<std::int64_t>(data.size()) != mensurflow::numel(shape)) {
            throw std::invalid_argument("from_vector: " + std::to_string(data.size()) +
                                        " values do not fill shape " + to_string(shape));
        }
        return from_host(data.data(), std::move(shape), dtype_of<T>, device);
    }

    // Views: new Tensors over the same storage with a different shape/strides/offset.
    // No data is copied, so they cost O(dim) regardless of tensor size.

    // Swap two dimensions (negative dims count from the end). Result is usually non-contiguous.
    Tensor transpose(int d0, int d1) const;

    // Elements [start, end) of one dimension, taking every step-th. Negative start/end count
    // from the end; out-of-range bounds are clamped, like Python slicing. step must be >= 1.
    Tensor slice(int d, std::int64_t start, std::int64_t end, std::int64_t step = 1) const;

    // Broadcast to a larger shape (NumPy rules) using stride 0; the data is repeated, not copied.
    Tensor expand(const Shape& shape) const;

    // Insert a size-1 dimension at position d, e.g. [3] -> unsqueeze(1) -> [3, 1].
    Tensor unsqueeze(int d) const;

    // Same elements in a new shape; one dimension may be -1 (inferred). Requires a contiguous
    // tensor, since otherwise the new shape cannot be expressed with strides alone.
    Tensor view(Shape shape) const;

    // True if both tensors use the same memory allocation (possibly different parts of it).
    bool shares_storage(const Tensor& other) const;

    // The same data on another device, copied through host memory into a contiguous tensor.
    // If the tensor is already on `device` it is returned as is (shared, not copied, and still
    // a strided view if it was one).
    Tensor to(Device device) const;

    // *this if already contiguous, otherwise a packed row-major copy on the same device.
    Tensor contiguous() const;

    // Writes numel() elements in row-major order to dst, following strides for views.
    // Blocks until the data is ready.
    void copy_to_host(void* dst) const;

    // The value of a one-element tensor, e.g. the result of dot(). Blocks like copy_to_host.
    template <typename T> T item() const {
        if (numel() != 1) {
            throw std::invalid_argument("item: tensor has " + std::to_string(numel()) +
                                        " elements, expected 1");
        }
        return to_vector<T>()[0];
    }

    template <typename T> std::vector<T> to_vector() const {
        if (dtype() != dtype_of<T>) {
            throw std::invalid_argument(std::string("to_vector: tensor is ") + dtype_name(dtype()) +
                                        ", requested " + dtype_name(dtype_of<T>));
        }
        std::vector<T> out(static_cast<std::size_t>(numel()));
        copy_to_host(out.data());
        return out;
    }

    bool defined() const { return storage_ != nullptr; }

    const Shape& shape() const;
    const Strides& strides() const;
    std::int64_t offset() const; // first element's position in storage, in elements
    DType dtype() const;
    Device device() const;

    int dim() const;                // number of dimensions; 0 for a scalar
    std::int64_t size(int d) const; // size of one dimension; negative d counts from the end
    std::int64_t numel() const;
    std::size_t itemsize() const; // bytes per element
    std::size_t nbytes() const;   // numel() * itemsize()
    bool is_contiguous() const;

  private:
    friend struct detail::TensorAccess;

    Tensor(std::shared_ptr<detail::Storage> storage, std::int64_t offset, Shape shape,
           Strides strides, DType dtype);

    void require_defined() const;
    Tensor make_view(std::int64_t offset, Shape shape, Strides strides) const;

    std::shared_ptr<detail::Storage> storage_;
    std::int64_t offset_ = 0;
    Shape shape_;
    Strides strides_;
    DType dtype_ = DType::F32;
};

} // namespace mensurflow
