// Tests Tensor creation, metadata, host round trips, and error handling on the CPU.
#include "mensurflow/tensor.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace mensurflow;

namespace {

void test_empty() {
    Tensor t = Tensor::empty({2, 3, 4});
    test::require(t.defined(), "defined");
    test::require(t.shape() == Shape{2, 3, 4}, "shape");
    test::require(t.strides() == Strides{12, 4, 1}, "contiguous strides");
    test::require(t.offset() == 0, "offset");
    test::require(t.dtype() == DType::F32, "default dtype");
    test::require(t.device() == Device{DeviceType::CPU, 0}, "default device");
    test::require(t.dim() == 3 && t.size(0) == 2 && t.size(-1) == 4, "dim and size");
    test::require(t.numel() == 24 && t.itemsize() == 4 && t.nbytes() == 96, "sizes");
    test::require(t.is_contiguous(), "is_contiguous");

    Tensor scalar = Tensor::empty({}, DType::I64);
    test::require(scalar.dim() == 0 && scalar.numel() == 1 && scalar.nbytes() == 8, "scalar");

    Tensor none = Tensor::empty({3, 0});
    test::require(none.numel() == 0 && none.nbytes() == 0, "zero-size tensor");
    test::require(none.to_vector<float>().empty(), "zero-size round trip");
}

void test_round_trip() {
    const std::vector<float> values{1.5f, -2.0f, 3.25f, 0.0f, 7.0f, -8.5f};
    Tensor t = Tensor::from_vector(values, {2, 3});
    test::require(t.to_vector<float>() == values, "f32 round trip");

    const std::vector<std::int32_t> ints{1, 2, 3, 4};
    Tensor ti = Tensor::from_vector(ints, {4});
    test::require(ti.dtype() == DType::I32, "dtype from element type");
    test::require(ti.to_vector<std::int32_t>() == ints, "i32 round trip");

    // from_vector copies, so changing the source afterwards does not affect the tensor.
    std::vector<float> source{1, 2};
    Tensor copied = Tensor::from_vector(source, {2});
    source[0] = 100;
    test::require(copied.to_vector<float>() == std::vector<float>{1, 2}, "from_vector copies");
}

void test_errors() {
    test::require_throws<std::invalid_argument>(
        [] { Tensor::from_vector(std::vector<float>{1, 2, 3}, {2, 2}); }, "size mismatch");
    test::require_throws<std::invalid_argument>([] { Tensor::empty({2, -1}); }, "negative dim");
    test::require_throws<std::runtime_error>(
        [] { Tensor::empty({2}, DType::F32, Device{DeviceType::Vulkan, 0}); }, "no vulkan yet");
    test::require_throws<std::runtime_error>(
        [] { Tensor::empty({2}, DType::F32, Device{DeviceType::CPU, 1}); }, "only cpu:0");
    test::require_throws<std::invalid_argument>(
        [] { Tensor::from_vector(std::vector<float>{1}, {1}).to_vector<std::int32_t>(); },
        "dtype mismatch");
    test::require_throws<std::invalid_argument>([] { Tensor::from_host(nullptr, {2}, DType::F32); },
                                                "null source");

    Tensor undefined;
    test::require(!undefined.defined(), "default is undefined");
    test::require_throws<std::logic_error>([&] { undefined.shape(); }, "undefined shape");
    test::require_throws<std::logic_error>([&] { undefined.to_vector<float>(); }, "undefined copy");
}

} // namespace

int main() {
    return test::run("tensor_test", [] {
        test_empty();
        test_round_trip();
        test_errors();
    });
}
