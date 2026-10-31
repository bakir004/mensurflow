// Lets kernels reach a Tensor's backend buffer without widening Tensor's public API.
#pragma once

#include "core/storage.hpp"
#include "mensurflow/tensor.hpp"

namespace mensurflow::detail {

struct TensorAccess {
    static Buffer& buffer(const Tensor& t) { return *t.storage_->buffer; }

    // Pointer to the tensor's first element for backends whose Buffer::native() is a plain
    // address (CPU, CUDA, ROCm). Not valid for Metal, which has to use native() as an MTLBuffer.
    template <typename T> static T* data(const Tensor& t) {
        auto* base = static_cast<std::byte*>(buffer(t).native());
        return reinterpret_cast<T*>(base + static_cast<std::size_t>(t.offset()) * sizeof(T));
    }
};

} // namespace mensurflow::detail
