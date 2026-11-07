// CPU activation contracts: independent float64 values, API views/errors, thread invariance,
// parameter forwarding and backward derivatives checked by numerical differentiation.
#include "backends/cpu/ops/elementwise/elementwise.hpp"
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
    Tensor (*apply)(const Tensor&, Execution);
    void (*kernel)(const float*, float*, std::int64_t);
    double (*reference)(double);
};

// Non-default parameters exercise the full API -> OpArgs -> adapter -> kernel path.
const Activation activations[] = {
    {"leaky_relu", detail::OpId::LeakyRelu,
     [](const Tensor& x, Execution e) { return leaky_relu(x, 0.2f, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::leaky_relu_f32(x, y, n, 0.2f); },
     [](double x) { return reference::leaky_relu_value(x, 0.2f); }},
    {"gelu", detail::OpId::Gelu,
     [](const Tensor& x, Execution e) { return gelu(x, false, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::gelu_f32(x, y, n); },
     [](double x) { return reference::gelu_value(x); }},
    {"gelu_tanh", detail::OpId::Gelu,
     [](const Tensor& x, Execution e) { return gelu(x, true, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::gelu_f32(x, y, n, true); },
     [](double x) { return reference::gelu_value(x, true); }},
    {"silu", detail::OpId::Silu, silu, cpu::silu_f32, reference::silu_value},
    {"elu", detail::OpId::Elu,
     [](const Tensor& x, Execution e) { return elu(x, 1.5f, e); },
     [](const float* x, float* y, std::int64_t n) { cpu::elu_f32(x, y, n, 1.5f); },
     [](double x) { return reference::elu_value(x, 1.5); }},
    {"softplus", detail::OpId::Softplus,
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

void test_threads_and_in_place() {
    const int original = cpu::max_threads();
    for (std::int64_t n : {std::int64_t{0}, std::int64_t{1}, std::int64_t{7}, std::int64_t{8},
                           std::int64_t{9}, std::int64_t{17}, cpu::parallel_threshold - 1,
                           cpu::parallel_threshold, cpu::parallel_threshold + 1,
                           std::int64_t{100003}}) {
        std::vector<float> input(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = (static_cast<int>(i % 161) - 80) / 8.0f;
        }
        for (const Activation& op : activations) {
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

void test_backward() {
    for (const Activation& op : activations) {
        // Values avoid the nonsmooth origin and the softplus linear cutoff.
        const std::vector<float> points{-3.0f, -0.75f, 0.25f, 2.0f, 8.0f};
        Tensor x = Tensor::from_vector(points, {static_cast<std::int64_t>(points.size())});
        x.set_requires_grad();
        const Tensor weights = Tensor::from_vector<float>({0.5f, -2.0f, 3.0f, 0.25f, 1.5f}, {5});
        sum(mul(op.apply(x, Execution::CPU), weights)).backward();
        const auto gradient = x.grad().to_vector<float>();
        const auto upstream = weights.to_vector<float>();
        for (std::size_t i = 0; i < points.size(); ++i) {
            const double h = 1e-4;
            const double numerical =
                (op.reference(points[i] + h) - op.reference(points[i] - h)) / (2.0 * h);
            const double expected = numerical * upstream[i];
            test::require(std::fabs(gradient[i] - expected) <= 2e-5 * (1.0 + std::fabs(expected)),
                          std::string(op.name) + " weighted backward vs finite differences");
        }
    }
    for (bool negative_zero : {false, true}) {
        Tensor x = Tensor::full({}, negative_zero ? -0.0 : 0.0);
        x.set_requires_grad();
        leaky_relu(x, 0.2f).backward();
        test::require(x.grad().item<float>() == 0.2f, "leaky_relu zero derivative convention");
        x.zero_grad();
        elu(x, 1.5f).backward();
        test::require(x.grad().item<float>() == 1.5f, "ELU zero derivative convention");
    }
    // Composition exercises a non-leaf activation input, not only independent leaf callbacks.
    Tensor x = Tensor::full({}, -2.0);
    x.set_requires_grad();
    leaky_relu(square(x), 0.2f).backward();
    test::require(x.grad().item<float>() == -4.0f, "activation chain rule");
}

void test_backward_limits() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const Activation& op : activations) {
        Tensor x = Tensor::from_vector<float>({-inf, -0.0f, 0.0f, inf, nan}, {5});
        x.set_requires_grad();
        sum(op.apply(x, Execution::CPU)).backward();
        const auto gradient = x.grad().to_vector<float>();
        const float at_zero = op.id == detail::OpId::LeakyRelu ? 0.2f
                              : op.id == detail::OpId::Elu    ? 1.5f
                                                             : 0.5f;
        const float at_negative_inf = op.id == detail::OpId::LeakyRelu ? 0.2f : 0.0f;
        test::require(gradient[0] == at_negative_inf && gradient[1] == at_zero &&
                          gradient[2] == at_zero && gradient[3] == 1.0f,
                      std::string(op.name) + " backward at infinity and signed zero");
        // Leaky ReLU follows its branch convention for NaN, like the existing ReLU rule.
        if (op.id != detail::OpId::LeakyRelu) {
            test::require(std::isnan(gradient[4]), std::string(op.name) + " backward NaN");
        }
    }
}

} // namespace

int main() {
    return test::run("activation_test", [] {
        test_api_and_errors();
        test_threads_and_in_place();
        test_parameters_and_special_values();
        test_backward();
        test_backward_limits();
    });
}
