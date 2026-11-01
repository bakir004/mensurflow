#include "backends/cpu/argmax.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential_argmax.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

void require_indices(const std::vector<std::int64_t>& got,
                      const std::vector<std::int64_t>& want, const std::string& label) {
    if (got != want) {
        std::string values;
        for (std::int64_t value : got) {
            values += std::to_string(value) + " ";
        }
        test::require(false, label + ": got " + values);
    }
}

void test_kernels() {
    const std::vector<float> input = {1, 5, 5, 2, -1, 4, 3, 4, 0, 0, 0, -2};
    std::vector<std::int64_t> expected = {1, 2, 1, 0};
    std::vector<std::int64_t> sequential(expected.size());
    cpu::sequential::argmax_f32(input.data(), sequential.data(), 4, 3);
    require_indices(sequential, expected, "sequential argmax");
    const int original = cpu::max_threads();
    for (int threads : {1, original}) {
        cpu::set_max_threads(threads);
        std::vector<std::int64_t> parallel(expected.size());
        cpu::argmax_f32(input.data(), parallel.data(), 4, 3);
        require_indices(parallel, expected, "parallel argmax");
    }
    cpu::set_max_threads(original);
}

void test_edges() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float values[] = {1, nan, 3, nan, 2, 1};
    std::int64_t output[2] = {};
    cpu::argmax_f32(values, output, 2, 3);
    test::require(output[0] == 1 && output[1] == 0, "first NaN wins within each row");
    test::require_throws<std::invalid_argument>(
        [&] { cpu::argmax_f32(values, output, 2, 0); }, "empty axis");
    test::require_throws<std::invalid_argument>(
        [&] { cpu::argmax_f32(nullptr, output, 1, 3); }, "null input");
}

bool supported(Device device) {
    return detail::Registry::instance().dispatcher().find(
               {detail::OpId::Argmax, device.type, DType::F32}) != nullptr;
}

void test_public(Device device) {
    const Tensor input =
        Tensor::from_vector(std::vector<float>{1, 9, 3, 7, 7, 2}, {2, 3}, device);
    const Tensor result = argmax(input);
    test::require(result.dtype() == DType::I64 && result.shape() == (Shape{2}),
                  "argmax output dtype and shape");
    require_indices(result.to_vector<std::int64_t>(), {1, 0}, "public argmax");

    const Tensor stored =
        Tensor::from_vector(std::vector<float>{1, 9, 3, 7, 7, 2, 8, 4}, {2, 4}, device);
    const Tensor view = stored.slice(1, 0, 4, 2);
    require_indices(argmax(view, 1).to_vector<std::int64_t>(), {1, 1}, "strided argmax");

    test::require_throws<std::invalid_argument>([&] { argmax(input, 0); }, "non-last axis");
    test::require_throws<std::invalid_argument>(
        [&] { argmax(Tensor::empty({2, 0}, DType::F32, device)); }, "empty axis");
    test::require_throws<std::invalid_argument>([&] { argmax(Tensor()); }, "undefined input");
}

} // namespace

int main() {
    return test::run("argmax_test", [] {
        test_kernels();
        test_edges();
        for (Device device : list_devices()) {
            if (supported(device)) {
                test_public(device);
            } else {
                test::require_throws<detail::NotImplemented>(
                    [&] { argmax(Tensor::empty({2, 2}, DType::F32, device)); },
                    "unsupported argmax");
            }
        }
    });
}
