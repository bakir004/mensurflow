// Tensor creation, metadata, and host transfers. Until the backend registry exists,
// only CPU memory is supported and transfers are plain memcpy.
#include "mensurflow/tensor.hpp"

#include "core/storage.hpp"

#include <cstring>

namespace mensurflow {

namespace {

// Temporary: replaced by a backend lookup once the registry exists.
void require_cpu(Device device) {
    if (device != Device{DeviceType::CPU, 0}) {
        throw std::runtime_error("device " + to_string(device) + " is not available (only cpu:0)");
    }
}

} // namespace

Tensor::Tensor(std::shared_ptr<detail::Storage> storage, std::int64_t offset, Shape shape,
               Strides strides, DType dtype)
    : storage_(std::move(storage)), offset_(offset), shape_(std::move(shape)),
      strides_(std::move(strides)), dtype_(dtype) {}

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
    validate_shape(shape);
    require_cpu(device);
    const auto bytes = static_cast<std::size_t>(mensurflow::numel(shape)) * dtype_size(dtype);
    auto storage = std::make_shared<detail::Storage>(device, bytes);
    Strides strides = contiguous_strides(shape);
    return Tensor(std::move(storage), 0, std::move(shape), std::move(strides), dtype);
}

Tensor Tensor::from_host(const void* data, Shape shape, DType dtype, Device device) {
    Tensor out = empty(std::move(shape), dtype, device);
    if (out.nbytes() > 0) {
        if (data == nullptr) {
            throw std::invalid_argument("from_host: data is null");
        }
        std::memcpy(out.storage_->data, data, out.nbytes());
    }
    return out;
}

void Tensor::copy_to_host(void* dst) const {
    require_defined();
    if (nbytes() == 0) {
        return;
    }
    if (dst == nullptr) {
        throw std::invalid_argument("copy_to_host: dst is null");
    }
    // Strided gather for views is added together with the view ops.
    if (!is_contiguous()) {
        throw std::runtime_error("copy_to_host: non-contiguous tensors are not supported yet");
    }
    const std::byte* src = storage_->data + static_cast<std::size_t>(offset_) * itemsize();
    std::memcpy(dst, src, nbytes());
}

void Tensor::require_defined() const {
    if (!defined()) {
        throw std::logic_error("operation on an undefined Tensor");
    }
}

const Shape& Tensor::shape() const {
    require_defined();
    return shape_;
}

const Strides& Tensor::strides() const {
    require_defined();
    return strides_;
}

std::int64_t Tensor::offset() const {
    require_defined();
    return offset_;
}

DType Tensor::dtype() const {
    require_defined();
    return dtype_;
}

Device Tensor::device() const {
    require_defined();
    return storage_->device;
}

int Tensor::dim() const { return static_cast<int>(shape().size()); }

std::int64_t Tensor::size(int d) const {
    return shape()[static_cast<std::size_t>(normalize_dim(d, dim()))];
}

std::int64_t Tensor::numel() const { return mensurflow::numel(shape()); }

std::size_t Tensor::itemsize() const { return dtype_size(dtype()); }

std::size_t Tensor::nbytes() const { return static_cast<std::size_t>(numel()) * itemsize(); }

bool Tensor::is_contiguous() const { return mensurflow::is_contiguous(shape(), strides()); }

} // namespace mensurflow
