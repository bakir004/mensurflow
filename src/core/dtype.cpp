// Per-dtype metadata used by Tensor size math and error messages.
#include "mensurflow/dtype.hpp"

#include <stdexcept>

namespace mensurflow {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
    case DType::I32:
        return 4;
    case DType::F16:
    case DType::BF16:
        return 2;
    case DType::I64:
        return 8;
    case DType::U8:
    case DType::Bool:
        return 1;
    }
    throw std::invalid_argument("invalid DType");
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return "f32";
    case DType::F16:
        return "f16";
    case DType::BF16:
        return "bf16";
    case DType::I32:
        return "i32";
    case DType::I64:
        return "i64";
    case DType::U8:
        return "u8";
    case DType::Bool:
        return "bool";
    }
    throw std::invalid_argument("invalid DType");
}

bool is_floating_point(DType dtype) {
    return dtype == DType::F32 || dtype == DType::F16 || dtype == DType::BF16;
}

} // namespace mensurflow
