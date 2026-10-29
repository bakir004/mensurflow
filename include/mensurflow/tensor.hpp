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
}

class Tensor {
  public:
    // Undefined tensor with no storage; everything except defined() throws.
    Tensor() = default;

    // Uninitialized memory of the given shape. Only the CPU device exists for now.
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

    // Writes numel() elements in row-major order to dst. Blocks until the data is ready.
    void copy_to_host(void* dst) const;

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
    Tensor(std::shared_ptr<detail::Storage> storage, std::int64_t offset, Shape shape,
           Strides strides, DType dtype);

    void require_defined() const;

    std::shared_ptr<detail::Storage> storage_;
    std::int64_t offset_ = 0;
    Shape shape_;
    Strides strides_;
    DType dtype_ = DType::F32;
};

} // namespace mensurflow
