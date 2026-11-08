// CPU activation forward contracts: independent float64 values, API views/errors,
// thread invariance, parameter forwarding and errors for pending backward rules.
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"
#include "backends/cpu/threads.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/elementwise.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

struct Activation {
    const char* name;
    detail::OpId id;
    cpu::ElementwiseOp parallel_op;
    Tensor (*apply)(const Tensor&, Execution);
    void (*kernel)(const float*, float*, std::int64_t);
    double (*reference)(double);
};

// Non-default parameters exercise the full API -> OpArgs -> adapter -> kernel path.
const Activation activations[] = {
    {"leaky_relu", detail::OpId::LeakyRelu, cpu::ElementwiseOp::leaky_relu,
     [](const Tensor& x, Execution e) { return leaky_relu(x, 0.2f, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::leaky_relu_f32(x, y, n, 0.2f); },
     [](double x) { return reference::leaky_relu_value(x, 0.2f); }},
    {"gelu", detail::OpId::Gelu, cpu::ElementwiseOp::gelu,
     [](const Tensor& x, Execution e) { return gelu(x, false, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::gelu_f32(x, y, n); },
     [](double x) { return reference::gelu_value(x); }},
    {"gelu_tanh", detail::OpId::Gelu, cpu::ElementwiseOp::gelu_tanh,
     [](const Tensor& x, Execution e) { return gelu(x, true, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::gelu_f32(x, y, n, true); },
     [](double x) { return reference::gelu_value(x, true); }},
    {"silu", detail::OpId::Silu, cpu::ElementwiseOp::silu, silu, cpu::silu_f32, reference::silu_value},
    {"elu", detail::OpId::Elu, cpu::ElementwiseOp::elu,
     [](const Tensor& x, Execution e) { return elu(x, 1.5f, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::elu_f32(x, y, n, 1.5f); },
     [](double x) { return reference::elu_value(x, 1.5); }},
    {"softplus", detail::OpId::Softplus, cpu::ElementwiseOp::softplus,
     [](const Tensor& x, Execution e) { return softplus(x, 2.0f, 15.0f, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::softplus_f32(x, y, n, 2.0f, 15.0f); },
     [](double x) { return reference::softplus_value(x, 2.0, 15.0); }}};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void check_values(const Activation& op, const std::vector<float>& input,
                  const std::vector<float>& actual) {
    test::require(input.size() == actual.size(), std::string(op.name) + " output size");
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float expected = static_cast<float>(op.reference(input[i]));
        test::require(test::close_f32(actual[i], expected, 3e-6f),
                      std::string(op.name) + " vs reference at " + std::to_string(i) +
                          ", input=" + std::to_string(input[i]));
    }
}

void check_api(const Activation& op, const Tensor& input) {
    const auto before = input.to_vector<float>();
    const Tensor out = op.apply(input, Execution::CPU);
    test::require(out.shape() == input.shape() && out.dtype() == DType::F32 &&
                      out.device() == input.device(),
                  std::string(op.name) + " metadata");
    test::require(out.is_contiguous() && !out.shares_storage(input),
                  std::string(op.name) + " new contiguous output");
    check_values(op, before, out.to_vector<float>());
    test::require(same_bits(before, input.to_vector<float>()),
                  std::string(op.name) + " preserves input");
}

void test_api_and_errors() {
    const Tensor matrix = Tensor::from_vector<float>({-3, -2, -1, -0.0f, 0, 1, 2, 3}, {2, 4});
    for (const Activation& op : activations) {
        test::require(detail::Registry::instance().dispatcher().find(
                          {op.id, DeviceType::CPU, DType::F32}) != nullptr,
                      std::string(op.name) + " registered");
        test::require(std::string(detail::op_name(op.id)) != "", "operation name");
        check_api(op, matrix);
        check_api(op, matrix.transpose(0, 1));
        check_api(op, matrix.slice(1, 1, 4, 2));
        check_api(op, matrix.slice(0, 1, 2));
        check_api(op, Tensor::from_vector<float>({-2, 0, 3}, {1, 3}).expand({4, 3}));
        check_api(op, Tensor::full({}, -0.75));
        check_api(op, Tensor::empty({2, 0}, DType::F32));
        test::require_throws<std::invalid_argument>(
            [&] { op.apply(Tensor{}, Execution::CPU); }, std::string(op.name) + " undefined");
        for (const Shape& shape : {Shape{2}, Shape{0}}) {
            test::require_throws<detail::NotImplemented>(
                [&] { op.apply(Tensor::empty(shape, DType::I32), Execution::CPU); },
                std::string(op.name) + " unsupported dtype, including empty");
            const Tensor cpu_input = Tensor::zeros(shape);
            const auto automatic = op.apply(cpu_input, Execution::Auto);
            test::require(automatic.device() == cpu_input.device(), "Auto output device");
            for (Device d : list_devices()) {
                if (d.type != DeviceType::CPU) {
                    test::require_throws<detail::NotImplemented>(
                        [&] { op.apply(cpu_input.to(d), Execution::GPU); },
                        std::string(op.name) + " unimplemented GPU kernel");
                }
            }
        }
    }
    const Tensor x = Tensor::full({1}, 1.0);
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float beta : {0.0f, -1.0f, inf, nan}) {
        test::require_throws<std::invalid_argument>([&] { softplus(x, beta); }, "softplus beta");
    }
    for (float threshold : {inf, -inf, nan}) {
        test::require_throws<std::invalid_argument>([&] { softplus(x, 1.0f, threshold); },
                                                    "softplus threshold");
    }
}

std::vector<std::int64_t> thread_sizes(cpu::ElementwiseOp op) {
    std::vector<std::int64_t> sizes{0, 1, 7, 8, 9, 17, 100003};
    const auto cutoff = cpu::elementwise_parallel_cutoff(op);
    // INT64_MAX means always serial; practical sizes exercise that policy without a huge allocation.
    if (cutoff != std::numeric_limits<std::int64_t>::max()) {
        if (cutoff > 0) {
            sizes.push_back(cutoff - 1);
        }
        sizes.push_back(cutoff);
        sizes.push_back(cutoff + 1);
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

void test_threads_and_in_place() {
    const int original = cpu::max_threads();
    for (const Activation& op : activations) {
        for (std::int64_t n : thread_sizes(op.parallel_op)) {
            std::vector<float> input(static_cast<std::size_t>(n));
            for (std::size_t i = 0; i < input.size(); ++i) {
                input[i] = (static_cast<int>(i % 161) - 80) / 8.0f;
            }
            cpu::set_max_threads(1);
            std::vector<float> baseline(input.size());
            op.kernel(input.data(), baseline.data(), n);
            check_values(op, input, baseline);
            op.kernel(nullptr, nullptr, 0);
            for (int threads : {1, 2, 3, original}) {
                cpu::set_max_threads(threads);
                std::vector<float> actual(input.size());
                op.kernel(input.data(), actual.data(), n);
                test::require(same_bits(baseline, actual), std::string(op.name) + " threads");
                actual = input;
                op.kernel(actual.data(), actual.data(), n);
                test::require(same_bits(baseline, actual), std::string(op.name) + " in-place");
            }
        }
    }
    cpu::set_max_threads(original);
}

void test_parameters_and_special_values() {
    const Tensor x = Tensor::from_vector<float>({-4, -1, -0.0f, 0, 1, 4}, {6});
    for (float slope : {0.0f, 0.01f, 0.2f, 1.0f, -0.5f}) {
        const auto got = leaky_relu(x, slope).to_vector<float>();
        const auto input = x.to_vector<float>();
        for (std::size_t i = 0; i < got.size(); ++i) {
            const float want = static_cast<float>(reference::leaky_relu_value(input[i], slope));
            test::require(test::close_f32(got[i], want), "leaky_relu custom slope");
        }
    }
    test::require(same_bits(leaky_relu(x).to_vector<float>(),
                            leaky_relu(x, 0.01f).to_vector<float>()),
                  "leaky_relu default slope");
    test::require(same_bits(gelu(x).to_vector<float>(), gelu(x, false).to_vector<float>()),
                  "GELU default exact");
    test::require(same_bits(elu(x).to_vector<float>(), elu(x, 1.0f).to_vector<float>()),
                  "ELU default alpha");
    test::require(same_bits(softplus(x).to_vector<float>(),
                            softplus(x, 1.0f, 20.0f).to_vector<float>()), "softplus defaults");
    test::require(std::abs(gelu(Tensor::full({}, 1.0)).item<float>() - 0.841344746f) < 1e-7f,
                  "known Gaussian CDF value");
    test::require(std::abs(silu(Tensor::full({}, 1.0)).item<float>() - 0.731058579f) < 1e-7f,
                  "known SiLU value");
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float large = std::numeric_limits<float>::max();
    const std::vector<float> values{0.0f, -0.0f, tiny, -tiny, -104, -100, -20, -14, -6,
                                     6, 20, 100, 1000, large, -large, inf, -inf, nan};
    const Tensor special = Tensor::from_vector(values, {static_cast<std::int64_t>(values.size())});
    for (const Activation& op : activations) {
        check_api(op, special);
    }
    test::require(softplus(Tensor::full({}, 1000.0)).item<float>() == 1000.0f,
                  "softplus avoids exponential overflow");
    for (float beta : {0.1f, 0.5f, 2.0f}) {
        for (float threshold : {-2.0f, 0.0f, 5.0f, 20.0f}) {
            const auto actual = softplus(special, beta, threshold).to_vector<float>();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const float want = static_cast<float>(reference::softplus_value(values[i], beta,
                                                                                threshold));
                test::require(test::close_f32(actual[i], want, 3e-6f), "softplus parameters");
            }
        }
    }
}

void test_backward_pending() {
    for (const Activation& op : activations) {
        Tensor x = Tensor::full({}, -0.75);
        x.set_requires_grad();
        const Tensor output = op.apply(x, Execution::CPU);
        check_values(op, {-0.75f}, output.to_vector<float>());
        test::require(output.requires_grad(), std::string(op.name) + " records the pending operation");
        test::require_throws<std::logic_error>([&] { output.backward(); },
                                              std::string(op.name) + " backward is not implemented");
    }
}

} // namespace

int main() {
    return test::run("activation_test", [] {
        test_api_and_errors();
        test_threads_and_in_place();
        test_parameters_and_special_values();
        test_backward_pending();
    });
}
