#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential_softmax.hpp"
#include "backends/cpu/softmax.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

void check_close(const std::vector<float>& got, const std::vector<float>& want,
                 const std::string& label) {
    test::require(got.size() == want.size(), label + ": size");
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(std::abs(got[i] - want[i]) <= 2e-6f,
                      label + ": entry " + std::to_string(i));
    }
}

void test_kernel() {
    const std::int64_t rows = 3, width = 5;
    const std::vector<float> input = {-1000, -999, -998, -997, -996,
                                      1,     2,    3,    4,    5,
                                      -2,    0,    2,    4,    6};
    std::vector<float> expected(input.size());
    for (std::int64_t row = 0; row < rows; ++row) {
        float maximum = input[static_cast<std::size_t>(row * width)];
        for (std::int64_t i = 1; i < width; ++i) {
            maximum = std::max(maximum, input[static_cast<std::size_t>(row * width + i)]);
        }
        double total = 0;
        for (std::int64_t i = 0; i < width; ++i) {
            total += std::exp(static_cast<double>(input[static_cast<std::size_t>(row * width + i)]) -
                              maximum);
        }
        for (std::int64_t i = 0; i < width; ++i) {
            expected[static_cast<std::size_t>(row * width + i)] = static_cast<float>(
                std::exp(static_cast<double>(input[static_cast<std::size_t>(row * width + i)]) -
                         maximum) /
                total);
        }
    }
    std::vector<float> sequential(input.size());
    cpu::sequential::softmax_f32(input.data(), sequential.data(), rows, width);
    check_close(sequential, expected, "sequential softmax");
    const int original = cpu::max_threads();
    for (int threads : {1, original}) {
        cpu::set_max_threads(threads);
        std::vector<float> parallel(input.size());
        cpu::softmax_f32(input.data(), parallel.data(), rows, width);
        check_close(parallel, expected, "parallel softmax");
    }
    cpu::set_max_threads(original);
}

void test_special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    float output[3] = {};
    const float with_infinities[3] = {inf, 1.0f, inf};
    cpu::softmax_f32(with_infinities, output, 1, 3);
    test::require(output[0] == 0.5f && output[1] == 0.0f && output[2] == 0.5f,
                  "positive infinities share probability");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float with_nan[3] = {1.0f, nan, 2.0f};
    cpu::softmax_f32(with_nan, output, 1, 3);
    test::require(std::isnan(output[0]) && std::isnan(output[1]) && std::isnan(output[2]),
                  "NaN propagates through row");
}

bool supported(Device device) {
    return detail::Registry::instance().dispatcher().find(
               {detail::OpId::Softmax, device.type, DType::F32}) != nullptr;
}

void test_public(Device device) {
    const std::vector<float> values = {1, 2, 3, 4, 5, 6};
    const Tensor input = Tensor::from_vector(values, {2, 3}, device);
    const Tensor result = softmax(input);
    test::require(result.shape() == (Shape{2, 3}) && result.is_contiguous(),
                  "softmax preserves shape");
    for (std::int64_t row = 0; row < 2; ++row) {
        float total = 0;
        for (std::int64_t i = 0; i < 3; ++i) {
            total += result.to_vector<float>()[static_cast<std::size_t>(row * 3 + i)];
        }
        test::require(test::close_f32(total, 1.0f, 2e-6f), "softmax row sums to one");
    }
    const Tensor stored = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, {2, 4},
                                              device);
    const Tensor view = stored.slice(0, 0, 2, 1).slice(1, 0, 4, 2);
    test::require(!view.is_contiguous(), "softmax input view");
    const Tensor view_result = softmax(view, 1);
    test::require(view_result.shape() == (Shape{2, 2}), "softmax view shape");
    test::require_throws<std::invalid_argument>([&] { softmax(input, 0); }, "non-last axis");
    test::require_throws<std::invalid_argument>([&] { softmax(Tensor(), -1); }, "undefined input");
    test::require_throws<std::invalid_argument>(
        [&] { softmax(Tensor::empty({}, DType::F32, device)); }, "scalar input");
}

} // namespace

int main() {
    return test::run("softmax_test", [] {
        test_kernel();
        test_special_values();
        for (Device device : list_devices()) {
            if (supported(device)) {
                test_public(device);
            } else {
                test::require_throws<detail::NotImplemented>(
                    [&] { softmax(Tensor::empty({2, 2}, DType::F32, device)); },
                    "unsupported softmax");
            }
        }
    });
}
