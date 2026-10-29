// Aligned host allocation behind CPU tensors.
#include "core/storage.hpp"

#include <new>

namespace mensurflow::detail {

Storage::Storage(Device device, std::size_t nbytes) : device(device), nbytes(nbytes) {
    if (nbytes > 0) {
        data = static_cast<std::byte*>(::operator new(nbytes, std::align_val_t{alignment}));
    }
}

Storage::~Storage() {
    if (data != nullptr) {
        ::operator delete(data, std::align_val_t{alignment});
    }
}

} // namespace mensurflow::detail
