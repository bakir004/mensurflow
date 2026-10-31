// Tensor creation, metadata, and host transfers. Memory comes from the device's backend through
// Storage; host-addressable memory (CPU, unified-memory GPUs) is read directly, anything else is
// staged through a host copy.
#include "mensurflow/tensor.hpp"

#include "core/storage.hpp"

#include <algorithm>
#include <cstring>

namespace mensurflow {

Tensor::Tensor(std::shared_ptr<detail::Storage> storage, std::int64_t offset, Shape shape,
               Strides strides, DType dtype)
    : storage_(std::move(storage)), offset_(offset), shape_(std::move(shape)),
      strides_(std::move(strides)), dtype_(dtype) {}

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
    validate_shape(shape);
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
        out.storage_->copy_from_host(0, data, out.nbytes());
    }
    return out;
}

Tensor Tensor::to(Device target) const {
    require_defined();
    if (target == device()) {
        return *this;
    }
    Tensor out = empty(shape_, dtype_, target);
    if (out.nbytes() > 0) {
        std::vector<std::byte> host(out.nbytes());
        copy_to_host(host.data());
        out.storage_->copy_from_host(0, host.data(), host.size());
    }
    return out;
}

Tensor Tensor::contiguous() const {
    require_defined();
    if (is_contiguous()) {
        return *this;
    }
    // Gather on the host, then upload. Correct on every device, but a GPU tensor makes a round
    // trip until a device-side strided copy kernel exists.
    Tensor out = empty(shape_, dtype_, device());
    std::vector<std::byte> host(out.nbytes());
    copy_to_host(host.data());
    out.storage_->copy_from_host(0, host.data(), host.size());
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
    const std::size_t item = itemsize();
    const std::size_t byte_offset = static_cast<std::size_t>(offset_) * item;
    auto* out = static_cast<std::byte*>(dst);
    if (is_contiguous()) {
        storage_->copy_to_host(out, byte_offset, nbytes());
        return;
    }

    // A strided view can touch any part of the allocation after its offset, so a device-only
    // buffer is first copied to the host from that offset onward.
    std::vector<std::byte> staged;
    const std::byte* base = nullptr;
    if (const std::byte* host = storage_->host_data()) {
        base = host + byte_offset;
    } else {
        staged.resize(storage_->nbytes - byte_offset);
        storage_->copy_to_host(staged.data(), byte_offset, staged.size());
        base = staged.data();
    }

    // Strided gather: visit every index in row-major order like an odometer, and copy each
    // element from offset + sum(index[i] * stride[i]). The last dimension is copied as one run
    // when its stride is 1, which is the common case for row slices.
    const int nd = dim();
    const bool rows_contiguous = strides_[nd - 1] == 1;
    const std::int64_t run = rows_contiguous ? shape_[nd - 1] : 1;
    const int outer = rows_contiguous ? nd - 1 : nd; // dimensions the odometer walks
    std::vector<std::int64_t> index(static_cast<std::size_t>(nd), 0);
    std::int64_t pos = 0; // element position relative to base for the current index
    for (std::int64_t done = 0; done < numel(); done += run) {
        std::memcpy(out, base + pos * static_cast<std::int64_t>(item),
                    static_cast<std::size_t>(run) * item);
        out += static_cast<std::size_t>(run) * item;
        // Advance the odometer: bump the last walked dimension, carrying into earlier ones.
        for (int d = outer - 1; d >= 0; --d) {
            if (++index[d] < shape_[d]) {
                pos += strides_[d];
                break;
            }
            pos -= (shape_[d] - 1) * strides_[d];
            index[d] = 0;
        }
    }
}

Tensor Tensor::make_view(std::int64_t offset, Shape shape, Strides strides) const {
    return Tensor(storage_, offset, std::move(shape), std::move(strides), dtype_);
}

Tensor Tensor::transpose(int d0, int d1) const {
    const int a = normalize_dim(d0, dim());
    const int b = normalize_dim(d1, dim());
    Shape shape = shape_;
    Strides strides = strides_;
    std::swap(shape[a], shape[b]);
    std::swap(strides[a], strides[b]);
    return make_view(offset_, std::move(shape), std::move(strides));
}

Tensor Tensor::slice(int d, std::int64_t start, std::int64_t end, std::int64_t step) const {
    const int axis = normalize_dim(d, dim());
    if (step < 1) {
        throw std::invalid_argument("slice: step must be >= 1");
    }
    const std::int64_t n = shape_[axis];
    // Python rules: negative bounds count from the end, then clamp into [0, n].
    auto clamp = [n](std::int64_t i) { return std::clamp(i < 0 ? i + n : i, std::int64_t{0}, n); };
    start = clamp(start);
    end = std::max(clamp(end), start);

    Shape shape = shape_;
    Strides strides = strides_;
    shape[axis] = (end - start + step - 1) / step; // ceil((end - start) / step)
    strides[axis] *= step;
    return make_view(offset_ + start * strides_[axis], std::move(shape), std::move(strides));
}

Tensor Tensor::expand(const Shape& shape) const {
    validate_shape(shape);
    Strides strides = broadcast_strides(this->shape(), strides_, shape);
    return make_view(offset_, shape, std::move(strides));
}

Tensor Tensor::unsqueeze(int d) const {
    // The new dimension may also go after the last one, so the valid range has dim() + 1 slots.
    const int axis = normalize_dim(d, dim() + 1);
    Shape shape = shape_;
    Strides strides = strides_;
    // A size-1 dimension is never stepped along; pick the stride a contiguous tensor would have.
    const std::int64_t stride = axis < dim() ? shape_[axis] * strides_[axis] : 1;
    shape.insert(shape.begin() + axis, 1);
    strides.insert(strides.begin() + axis, stride);
    return make_view(offset_, std::move(shape), std::move(strides));
}

Tensor Tensor::view(Shape shape) const {
    if (!is_contiguous()) {
        throw std::invalid_argument("view: tensor is not contiguous");
    }
    // Infer at most one -1 dimension from the remaining sizes.
    auto inferred = shape.end();
    std::int64_t known = 1;
    for (auto it = shape.begin(); it != shape.end(); ++it) {
        if (*it == -1) {
            if (inferred != shape.end()) {
                throw std::invalid_argument("view: only one dimension can be -1");
            }
            inferred = it;
        } else {
            known *= *it;
        }
    }
    if (inferred != shape.end()) {
        if (known == 0 || numel() % known != 0) {
            throw std::invalid_argument("view: cannot infer -1 in " + to_string(shape) + " for " +
                                        std::to_string(numel()) + " elements");
        }
        *inferred = numel() / known;
    }
    validate_shape(shape);
    if (mensurflow::numel(shape) != numel()) {
        throw std::invalid_argument("view: shape " + to_string(shape) + " does not match " +
                                    std::to_string(numel()) + " elements");
    }
    Strides strides = contiguous_strides(shape);
    return make_view(offset_, std::move(shape), std::move(strides));
}

bool Tensor::shares_storage(const Tensor& other) const {
    return defined() && storage_ == other.storage_;
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
