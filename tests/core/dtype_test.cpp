// Tests dtype sizes and names.
#include "mensurflow/dtype.hpp"
#include "test_util.hpp"

#include <stdexcept>
#include <string>

using namespace mensurflow;

int main() {
    return test::run("dtype_test", [] {
        test::require(dtype_size(DType::F32) == 4, "f32 size");
        test::require(dtype_size(DType::BF16) == 2, "bf16 size");
        test::require(dtype_size(DType::I64) == 8, "i64 size");
        test::require(dtype_size(DType::Bool) == 1, "bool size");
        test::require(std::string(dtype_name(DType::F16)) == "f16", "f16 name");
        test::require(is_floating_point(DType::BF16), "bf16 is float");
        test::require(!is_floating_point(DType::I32), "i32 is not float");
        test::require_throws<std::invalid_argument>([] { dtype_size(static_cast<DType>(99)); },
                                                    "invalid dtype");
    });
}
