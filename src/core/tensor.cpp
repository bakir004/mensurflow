// Tensor creation, metadata, and host transfers. Memory comes from the device's backend through
// Storage; host-addressable memory (CPU, unified-memory GPUs) is read directly, anything else is
// staged through a host copy.
#include "mensurflow/tensor.hpp"

#include "core/storage.hpp"
#include "core/autograd.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

namespace mensurflow {

namespace {

void validate_fill_value(double value, DType dtype) {
    if (!std::isfinite(value)) {
        if (dtype != DType::F32) {
            throw std::invalid_argument("full: non-finite values require f32 dtype");
        }
        return;
    }
    switch (dtype) {
    case DType::F32:
        if (value < -std::numeric_limits<float>::max() ||
            value > std::numeric_limits<float>::max()) {
            throw std::invalid_argument("full: value does not fit f32");
        }
        return;
    case DType::I32:
        if (std::trunc(value) != value || value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("full: value does not fit i32");
        }
        return;
    case DType::I64:
        if (std::trunc(value) != value || value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
            value > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("full: value does not fit i64");
        }
        return;
    case DType::U8:
        if (std::trunc(value) != value || value < 0 || value > std::numeric_limits<std::uint8_t>::max()) {
            throw std::invalid_argument("full: value does not fit u8");
        }
        return;
    case DType::Bool:
        return;
    case DType::F16:
    case DType::BF16:
        throw std::invalid_argument("full: f16 and bf16 construction is not supported yet");
    }
    throw std::invalid_argument("full: invalid dtype");
}

void write_fill(std::byte* data, std::size_t count, DType dtype, double value) {
    switch (dtype) {
    case DType::F32: {
        const float converted = static_cast<float>(value);
        auto* out = reinterpret_cast<float*>(data);
        std::fill(out, out + count, converted);
        return;
    }
    case DType::I32: {
        const auto converted = static_cast<std::int32_t>(value);
        auto* out = reinterpret_cast<std::int32_t*>(data);
        std::fill(out, out + count, converted);
        return;
    }
    case DType::I64: {
        const auto converted = static_cast<std::int64_t>(value);
        auto* out = reinterpret_cast<std::int64_t*>(data);
        std::fill(out, out + count, converted);
        return;
    }
    case DType::U8: {
        const auto converted = static_cast<std::uint8_t>(value);
        std::fill(data, data + count, static_cast<std::byte>(converted));
        return;
    }
    case DType::Bool:
        std::fill(data, data + count, value == 0 ? std::byte{0} : std::byte{1});
        return;
    case DType::F16:
    case DType::BF16:
        throw std::invalid_argument("full: f16 and bf16 construction is not supported yet");
    }
    throw std::invalid_argument("full: invalid dtype");
}

} // namespace

Tensor::Tensor(std::shared_ptr<detail::Storage> storage, std::int64_t offset, Shape shape,
               Strides strides, DType dtype)
    : storage_(std::move(storage)), offset_(offset), shape_(std::move(shape)),
      strides_(std::move(strides)), dtype_(dtype),
      autograd_(std::make_shared<detail::AutogradState>()) {}

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
    validate_shape(shape);
    const auto bytes = static_cast<std::size_t>(mensurflow::numel(shape)) * dtype_size(dtype);
    auto storage = std::make_shared<detail::Storage>(device, bytes);
    Strides strides = contiguous_strides(shape);
    return Tensor(std::move(storage), 0, std::move(shape), std::move(strides), dtype);
}

Tensor Tensor::full(Shape shape, double value, DType dtype, Device device) {
    validate_shape(shape);
    validate_fill_value(value, dtype);
    const std::size_t count = static_cast<std::size_t>(mensurflow::numel(shape));
    std::vector<std::byte> host(count * dtype_size(dtype));
    write_fill(host.data(), count, dtype, value);
    return from_host(host.data(), std::move(shape), dtype, device);
}

Tensor Tensor::constant(Shape shape, double value, DType dtype, Device device) {
    return full(std::move(shape), value, dtype, device);
}

Tensor Tensor::zeros(Shape shape, DType dtype, Device device) {
    validate_shape(shape);
    std::vector<std::byte> host(
        static_cast<std::size_t>(mensurflow::numel(shape)) * dtype_size(dtype));
    return from_host(host.data(), std::move(shape), dtype, device);
}

Tensor Tensor::ones(Shape shape, DType dtype, Device device) {
    return full(std::move(shape), 1.0, dtype, device);
}

Tensor Tensor::eye(std::int64_t n, DType dtype, Device device) {
    if (n < 0) {
        throw std::invalid_argument("eye: dimension must be non-negative");
    }
    validate_fill_value(1.0, dtype);
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<std::byte> host(count * dtype_size(dtype));
    write_fill(host.data(), count, dtype, 0.0);
    for (std::int64_t i = 0; i < n; ++i) {
        write_fill(host.data() + static_cast<std::size_t>(i * n + i) * dtype_size(dtype), 1,
                   dtype, 1.0);
    }
    return from_host(host.data(), {n, n}, dtype, device);
}

Tensor Tensor::zeros_like(const Tensor& other) {
    other.require_defined();
    return zeros(other.shape_, other.dtype_, other.device());
}

Tensor Tensor::ones_like(const Tensor& other) {
    other.require_defined();
    return ones(other.shape_, other.dtype_, other.device());
}

Tensor Tensor::full_like(const Tensor& other, double value) {
    other.require_defined();
    return full(other.shape_, value, other.dtype_, other.device());
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

void Tensor::set_requires_grad(bool requires_grad) {
    require_defined();
    if (requires_grad && dtype_ != DType::F32) {
        throw std::invalid_argument("set_requires_grad: only f32 tensors support autograd");
    }
    autograd_->requires_grad = requires_grad;
    autograd_->node.reset();
    if (!requires_grad) {
        autograd_->grad.reset();
    }
}

bool Tensor::requires_grad() const {
    require_defined();
    return autograd_->requires_grad;
}

Tensor Tensor::grad() const {
    require_defined();
    return autograd_->grad == nullptr ? Tensor{} : *autograd_->grad;
}

bool Tensor::has_grad() const {
    require_defined();
    return autograd_->grad != nullptr;
}

void Tensor::zero_grad() {
    require_defined();
    if (autograd_->grad != nullptr) {
        autograd_->grad = std::make_shared<Tensor>(Tensor::zeros(shape_, DType::F32, device()));
    }
}

void Tensor::copy_from(const Tensor& source) {
    require_defined();
    source.require_defined();
    if (shape_ != source.shape_ || dtype_ != source.dtype_ || device() != source.device()) {
        throw std::invalid_argument("copy_from: tensors must have matching shape, dtype, and device");
    }
    const Tensor packed = source.contiguous();
    std::vector<std::byte> bytes(packed.nbytes());
    packed.copy_to_host(bytes.data());
    storage_->copy_from_host(static_cast<std::size_t>(offset_) * itemsize(), bytes.data(),
                             bytes.size());
}

void Tensor::backward() const {
    require_defined();
    detail::autograd::backward(*this);
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
    Tensor view(storage_, offset, std::move(shape), std::move(strides), dtype_);
    view.autograd_ = autograd_;
    return view;
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
