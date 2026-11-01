#include "mensurflow/mensurflow.hpp"
#include "backends/cpu/reduction.hpp"
#include "backends/cpu/sequential_reduction.hpp"
#include "reference/reduction.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;
namespace sequential = mensurflow::cpu::sequential;

namespace {

std::vector<float> make_input(std::int64_t n) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] =
            static_cast<float>((static_cast<int>(i * 37 % 2001) - 1000)) / 250.0f;
    }
    return values;
}

void check_reduction(Device device, const std::vector<float>& values) {
    const Tensor input = Tensor::from_vector(values, {static_cast<std::int64_t>(values.size())},
                                             device);
    const auto want_sum = reference::sum(values.data(), static_cast<std::int64_t>(values.size()));
    test::require(test::close_f32(sum(input).item<float>(), static_cast<float>(want_sum.value),
                                  2e-4f),
                  "sum reference");
    if (!values.empty()) {
        const auto want_mean =
            reference::mean(values.data(), static_cast<std::int64_t>(values.size()));
        test::require(test::close_f32(mean(input).item<float>(), static_cast<float>(want_mean.value),
                                      2e-4f),
                      "mean reference");
        test::require(test::close_f32(max(input).item<float>(),
                                      reference::max(values.data(),
                                                     static_cast<std::int64_t>(values.size()))),
                      "max reference");
    }
}

void test_public_api(Device device) {
    for (std::int64_t n : {1, 2, 15, 16, 17, 32768, 32769, 100003}) {
        check_reduction(device, make_input(n));
    }
    const Tensor empty = Tensor::empty({0}, DType::F32, device);
    test::require(sum(empty).item<float>() == 0.0f, "empty sum is zero");
    test::require_throws<std::invalid_argument>([&] { mean(empty); }, "empty mean");
    test::require_throws<std::invalid_argument>([&] { max(empty); }, "empty max");
    test::require_throws<std::invalid_argument>([&] { sum(Tensor()); }, "undefined sum");
    test::require_throws<std::runtime_error>(
        [&] {
            const Tensor ints =
                Tensor::from_vector(std::vector<std::int32_t>{1, 2, 3}, {3}, device);
            sum(ints);
        },
        "unsupported dtype");
}

void test_views(Device device) {
    const auto values = make_input(1001);
    const Tensor input = Tensor::from_vector(values, {std::int64_t{1001}}, device);
    const Tensor sliced = input.slice(0, 0, 1001, 2);
    std::vector<float> expected;
    for (std::size_t i = 0; i < values.size(); i += 2) {
        expected.push_back(values[i]);
    }
    check_reduction(device, expected);
    test::require(test::close_f32(sum(sliced).item<float>(),
                                  sum(Tensor::from_vector(
                                          expected,
                                          {static_cast<std::int64_t>(expected.size())}, device))
                                      .item<float>(),
                                  2e-4f),
                  "strided slice sum");
}

void test_special_values(Device device) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor input =
        Tensor::from_vector(std::vector<float>{-2.0f, nan, 4.0f}, {3}, device);
    test::require(std::isnan(sum(input).item<float>()), "sum propagates NaN");
    test::require(std::isnan(mean(input).item<float>()), "mean propagates NaN");
    test::require(std::isnan(max(input).item<float>()), "max propagates NaN");
    const Tensor zeros = Tensor::from_vector(std::vector<float>{-0.0f, 0.0f}, {2}, device);
    test::require(!std::signbit(max(zeros).item<float>()), "max follows fmax zero semantics");
}

void test_kernels() {
    const auto values = make_input(100003);
    const auto n = static_cast<std::int64_t>(values.size());
    test::require(test::close_f32(cpu::sum_f32(values.data(), n),
                                  sequential::sum_f32(values.data(), n), 2e-4f),
                  "parallel and sequential sum");
    test::require(test::close_f32(cpu::mean_f32(values.data(), n),
                                  sequential::mean_f32(values.data(), n), 2e-4f),
                  "parallel and sequential mean");
    test::require(test::close_f32(cpu::max_f32(values.data(), n),
                                  sequential::max_f32(values.data(), n)),
                  "parallel and sequential max");
}

} // namespace

int main() {
    return test::run("reduction_test", [] {
        for (Device device : list_devices()) {
            test_public_api(device);
            test_views(device);
            test_special_values(device);
        }
        test_kernels();
    });
}
