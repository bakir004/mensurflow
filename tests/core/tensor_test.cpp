// Tests Tensor creation, metadata, host round trips, and error handling on the CPU.
#include "mensurflow/mensurflow.hpp"
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

void test_constructors() {
    test::require(Tensor::zeros({2, 2}).to_vector<float>() ==
                      std::vector<float>{0, 0, 0, 0},
                  "zeros");
    test::require(Tensor::ones({2, 2}).to_vector<float>() ==
                      std::vector<float>{1, 1, 1, 1},
                  "ones");
    test::require(Tensor::constant({2, 2}, 2.5).to_vector<float>() ==
                      std::vector<float>{2.5f, 2.5f, 2.5f, 2.5f},
                  "constant");
    test::require(Tensor::full({3}, 7, DType::I32).to_vector<std::int32_t>() ==
                      std::vector<std::int32_t>{7, 7, 7},
                  "integer full");
    test::require(Tensor::eye(3).to_vector<float>() ==
                      std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1},
                  "eye");
    const Tensor source = Tensor::from_vector(std::vector<float>{4, 5}, {2});
    test::require(Tensor::zeros_like(source).to_vector<float>() ==
                      std::vector<float>{0, 0},
                  "zeros_like");
    test::require(Tensor::ones_like(source).to_vector<float>() ==
                      std::vector<float>{1, 1},
                  "ones_like");
    test::require(Tensor::full_like(source, 3).to_vector<float>() ==
                      std::vector<float>{3, 3},
                  "full_like");
    test::require_throws<std::invalid_argument>([] { Tensor::eye(-1); }, "negative eye");
    test::require_throws<std::invalid_argument>(
        [] { Tensor::full({2}, 1.5, DType::I32); }, "fractional integer full");
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

void test_autograd_contract() {
    Tensor x = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3});
    test::require(!x.requires_grad() && !x.has_grad(), "autograd disabled by default");

    x.set_requires_grad();
    test::require(x.requires_grad() && !x.has_grad(), "requires_grad marks a leaf");
    test::require(!x.grad().defined(), "leaf gradient starts undefined");

    test::require_throws<std::invalid_argument>(
        [&] { x.backward(); }, "backward requires a scalar output");

    Tensor scalar = Tensor::from_vector(std::vector<float>{2}, {});
    scalar.set_requires_grad();
    scalar.backward();
    test::require(scalar.grad().item<float>() == 1.0f, "scalar backward seed");

    x.zero_grad();
    test::require(!x.has_grad(), "zero_grad is safe without an accumulated gradient");
    x.set_requires_grad(false);
    test::require(!x.requires_grad(), "requires_grad can be disabled");

    Tensor ints = Tensor::from_vector(std::vector<std::int32_t>{1, 2}, {2});
    test::require_throws<std::invalid_argument>(
        [&] { ints.set_requires_grad(); }, "autograd only supports f32");
}

void test_autograd_operations() {
    Tensor x = Tensor::from_vector(std::vector<float>{2, -3}, {2});
    Tensor y = Tensor::from_vector(std::vector<float>{4, 5}, {2});
    x.set_requires_grad();
    y.set_requires_grad();

    Tensor loss = sum(relu(add(mul(x, y), x)));
    loss.backward();

    test::require(x.grad().to_vector<float>() == std::vector<float>{5, 0},
                  "composed gradient for x");
    test::require(y.grad().to_vector<float>() == std::vector<float>{2, 0},
                  "composed gradient for y");

    x.zero_grad();
    y.zero_grad();
    test::require(!x.grad().to_vector<float>().empty(), "zero_grad keeps gradient storage");
    test::require(x.grad().to_vector<float>() == std::vector<float>{0, 0},
                  "zero_grad clears gradient");
}

} // namespace

int main() {
    return test::run("tensor_test", [] {
        test_empty();
        test_round_trip();
        test_constructors();
        test_errors();
        test_autograd_contract();
        test_autograd_operations();
    });
}
