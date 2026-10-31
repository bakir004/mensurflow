// Tests moving data between devices and packing views: Tensor::to, contiguous, and item. Runs on
// every available device, so on a GPU build it covers real host/device copies.
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace mensurflow;

namespace {

std::vector<float> iota(int n) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), 0.0f);
    return v;
}

void test_round_trip(Device device) {
    const auto values = iota(24);
    const Tensor t = Tensor::from_vector(values, {2, 3, 4}, device);
    test::require(t.device() == device, "created on the requested device");
    test::require(t.to_vector<float>() == values, "host round trip");
}

void test_to(Device device) {
    const auto values = iota(12);
    const Tensor cpu = Tensor::from_vector(values, {3, 4});
    const Tensor moved = cpu.to(device);
    test::require(moved.device() == device && moved.shape() == Shape{3, 4}, "moved");
    test::require(moved.to_vector<float>() == values, "values survive the move");
    test::require(moved.to({DeviceType::CPU, 0}).to_vector<float>() == values, "and back");
    test::require(moved.to(device).shares_storage(moved), "same device is not a copy");

    // A view that crosses devices is packed on the way, since the copy has no use for the
    // source's strides. On the same device it is returned unchanged, so only values are checked.
    const Tensor column = cpu.transpose(0, 1).to(device);
    test::require(column.shape() == Shape{4, 3}, "transposed shape");
    test::require(column.to_vector<float>() ==
                      (std::vector<float>{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}),
                  "transposed values");
    if (device != Device{DeviceType::CPU, 0}) {
        test::require(column.is_contiguous(), "cross-device copy is packed");
    }
}

void test_contiguous(Device device) {
    const Tensor t = Tensor::from_vector(iota(12), {3, 4}, device);
    test::require(t.contiguous().shares_storage(t), "contiguous input is returned as is");

    const Tensor view = t.transpose(0, 1);
    test::require(!view.is_contiguous(), "transpose is a strided view");
    const Tensor packed = view.contiguous();
    test::require(packed.is_contiguous() && !packed.shares_storage(t), "packed copy");
    test::require(packed.device() == device, "stays on the device");
    test::require(packed.to_vector<float>() == view.to_vector<float>(), "same logical values");

    const Tensor every_other = t.slice(1, 0, 4, 2).contiguous();
    test::require(every_other.to_vector<float>() == (std::vector<float>{0, 2, 4, 6, 8, 10}),
                  "strided slice packed");
}

void test_item() {
    test::require(Tensor::from_vector(std::vector<float>{2.5f}, {}).item<float>() == 2.5f,
                  "scalar");
    test::require(Tensor::from_vector(std::vector<int>{7}, {1, 1}).item<int>() == 7, "1x1");
    test::require_throws<std::invalid_argument>(
        [] { Tensor::from_vector(iota(2), {2}).item<float>(); }, "two elements");
    test::require_throws<std::invalid_argument>(
        [] { Tensor::from_vector(std::vector<float>{1}, {}).item<int>(); }, "wrong type");
}

} // namespace

int main() {
    return test::run("transfer_test", [] {
        for (Device device : list_devices()) {
            std::cout << "  transfers on " << to_string(device) << '\n';
            test_round_trip(device);
            test_to(device);
            test_contiguous(device);
        }
        test_item();
    });
}
