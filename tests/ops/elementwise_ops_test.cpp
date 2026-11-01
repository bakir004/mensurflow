// Tests binary and unary elementwise ops on every device this build can use: values against an
// independent broadcasting reference, views, empty shapes, and error handling. Complements
// elementwise_test.cpp, which tests the CPU kernels directly.
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/elementwise.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;

namespace {

using Shape64 = std::vector<std::int64_t>;
using TensorOp = Tensor (*)(const Tensor&, const Tensor&);
using FloatOp = float (*)(float, float);

struct Op {
    detail::OpId id;
    const char* name;
    TensorOp apply;
    FloatOp expected;
};

const Op ops[] = {{detail::OpId::Add, "add", add, [](float x, float y) { return x + y; }},
                  {detail::OpId::Sub, "sub", sub, [](float x, float y) { return x - y; }},
                  {detail::OpId::Mul, "mul", mul, [](float x, float y) { return x * y; }},
                  {detail::OpId::Div, "div", div, [](float x, float y) { return x / y; }}};

struct UnaryOp {
    detail::OpId id;
    const char* name;
    Tensor (*apply)(const Tensor&);
    void (*reference)(const float*, float*, std::int64_t);
    float reference_tolerance = 0.0f;
};

const UnaryOp unary_ops[] = {{detail::OpId::Neg, "neg", neg, reference::neg},
                             {detail::OpId::Abs, "abs", mensurflow::abs, reference::abs},
                             {detail::OpId::Square, "square", square, reference::square},
                             {detail::OpId::Relu, "relu", relu, reference::relu},
                             {detail::OpId::Sqrt, "sqrt", mensurflow::sqrt, reference::sqrt, 4e-7f},
                             {detail::OpId::Exp, "exp", mensurflow::exp, reference::exp, 4e-7f},
                             {detail::OpId::Sigmoid, "sigmoid", sigmoid, reference::sigmoid,
                              4e-7f}};

// Some vendor backends cannot express every op (cuBLAS and rocBLAS have no elementwise divide).
// Those ops must fail with NotImplemented rather than crash or return garbage; they are not
// checked for values.
template <typename Operation> bool supported(const Operation& op, Device device) {
    return detail::Registry::instance().dispatcher().find({op.id, device.type, DType::F32}) !=
           nullptr;
}

void check_unsupported(const Op& op, Device device) {
    const Tensor x = Tensor::from_vector(std::vector<float>{1, 2}, {2}, device);
    test::require_throws<detail::NotImplemented>([&] { op.apply(x, x); },
                                                 std::string(op.name) + " without a kernel");
}

void check_unsupported(const UnaryOp& op, Device device) {
    for (const Shape& shape : {Shape{2}, Shape{0}}) {
        const Tensor x = Tensor::empty(shape, DType::F32, device);
        test::require_throws<detail::NotImplemented>([&] { op.apply(x); },
                                                     std::string(op.name) + " without a kernel");
    }
}

// Multiples of 0.25 in [-4, 4], including zeros so division produces inf and NaN.
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

std::int64_t count(const Shape64& shape) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        n *= d;
    }
    return n;
}

// CPU results must match the reference exactly. Vendor libraries on a GPU may round division
// differently in the last bit, so there a small relative error is allowed; non-finite results
// must still agree on which kind they are.
bool close_enough(float got, float want, bool exact) {
    if (std::isnan(want) || std::isnan(got)) {
        return std::isnan(want) && std::isnan(got);
    }
    if (std::isinf(want) || std::isinf(got)) {
        return got == want;
    }
    if (exact) {
        return std::memcmp(&got, &want, sizeof(float)) == 0 || got == want; // also +0 == -0
    }
    return std::abs(got - want) <= 4e-7f * std::max({std::abs(want), std::abs(got), 1e-30f});
}

// Runs `op` on tensors built from (a, a_shape) and (b, b_shape) and checks every element.
void check_case(const Op& op, Device device, const std::vector<float>& a, const Shape64& a_shape,
                const std::vector<float>& b, const Shape64& b_shape, const std::string& what) {
    const std::string where = std::string(op.name) + " " + what + " on " + to_string(device);
    const Tensor ta = Tensor::from_vector(a, a_shape, device);
    const Tensor tb = Tensor::from_vector(b, b_shape, device);
    const Tensor out = op.apply(ta, tb);
    const auto want = reference::broadcast_binary(a, a_shape, b, b_shape, op.expected);

    test::require(out.shape() == want.shape, where + ": shape " + to_string(out.shape()));
    test::require(out.dtype() == DType::F32 && out.device() == device, where + ": dtype/device");
    test::require(out.is_contiguous(), where + ": result is packed");
    test::require(!out.shares_storage(ta) && !out.shares_storage(tb), where + ": new storage");
    const auto got = out.to_vector<float>();
    const bool exact = device == Device{DeviceType::CPU, 0};
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(close_enough(got[i], want.values[i], exact),
                      where + ": element " + std::to_string(i) + " got " + std::to_string(got[i]) +
                          ", want " + std::to_string(want.values[i]));
    }
    // Inputs must be untouched.
    test::require(ta.to_vector<float>() == a && tb.to_vector<float>() == b, where + ": inputs");
}

void test_shapes(Device device) {
    struct Case {
        Shape64 a, b;
        const char* what;
    };

    const Case cases[] = {{{6}, {6}, "vectors"},
                          {{2, 3}, {2, 3}, "same shape"},
                          {{}, {2, 3}, "scalar and matrix"},
                          {{2, 3}, {}, "matrix and scalar"},
                          {{2, 3}, {3}, "row broadcast"},
                          {{2, 3}, {2, 1}, "column broadcast"},
                          {{4, 1}, {1, 5}, "outer"},
                          {{2, 1, 3}, {4, 1}, "3-D"},
                          {{257, 129}, {129}, "past the parallel cutoff"},
                          {{512, 512}, {512, 512}, "262k elements"},
                          {{0}, {0}, "empty"},
                          {{2, 0}, {1}, "empty matrix"},
                          {{0}, {1}, "empty and size 1"}};
    for (const Case& c : cases) {
        const auto a = make_input(count(c.a), 1);
        const auto b = make_input(count(c.b), 2);
        for (const Op& op : ops) {
            if (supported(op, device)) {
                check_case(op, device, a, c.a, b, c.b, c.what);
            }
        }
    }
}

void test_views(Device device) {
    // transpose gives a non-contiguous input; slices give a step and a non-zero offset.
    const auto a = make_input(24, 3);
    const auto b = make_input(24, 4);
    const Tensor ta = Tensor::from_vector(a, {4, 6}, device);
    const Tensor tb = Tensor::from_vector(b, {6, 4}, device);
    for (const Op& op : ops) {
        if (!supported(op, device)) {
            continue;
        }
        const Tensor out = op.apply(ta, tb.transpose(0, 1)); // {4,6} with a {4,6} view
        const auto got = out.to_vector<float>();
        for (std::int64_t i = 0; i < 4; ++i) {
            for (std::int64_t j = 0; j < 6; ++j) {
                const float want = op.expected(a[static_cast<std::size_t>(i * 6 + j)],
                                               b[static_cast<std::size_t>(j * 4 + i)]);
                test::require(close_enough(got[static_cast<std::size_t>(i * 6 + j)], want,
                                           device == Device{DeviceType::CPU, 0}),
                              std::string(op.name) + " transposed view on " + to_string(device));
            }
        }
        const Tensor every_other = op.apply(ta.slice(1, 0, 6, 2), ta.slice(1, 1, 6, 2));
        test::require(every_other.shape() == (Shape{4, 3}), "strided slice shape");
        const Tensor offset = op.apply(ta.slice(0, 1, 3), tb.transpose(0, 1).slice(0, 1, 3));
        test::require(offset.shape() == (Shape{2, 6}), "offset slice shape");
    }
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void check_unary_case(const UnaryOp& op, const Tensor& input, const std::string& what) {
    const std::string where = std::string(op.name) + " " + what;
    const auto before = input.to_vector<float>();
    std::vector<float> want(before.size());
    op.reference(before.data(), want.data(), input.numel());
    const Tensor out = op.apply(input);
    test::require(out.shape() == input.shape(), where + ": shape");
    test::require(out.dtype() == input.dtype() && out.device() == input.device(),
                  where + ": dtype/device");
    test::require(out.is_contiguous() && !out.shares_storage(input), where + ": new packed output");
    test::require(test::matches_f32(out.to_vector<float>(), want, op.reference_tolerance),
                  where + ": values");
    test::require(same_bits(input.to_vector<float>(), before), where + ": input unchanged");
}

void test_unary_shapes_and_views(Device device) {
    for (const UnaryOp& op : unary_ops) {
        if (!supported(op, device)) {
            continue;
        }
        for (const Shape& shape : {Shape{}, Shape{6}, Shape{2, 3}, Shape{100003}, Shape{0},
                                  Shape{2, 0}}) {
            check_unary_case(op, Tensor::from_vector(make_input(count(shape), 9), shape, device),
                             "shape " + to_string(shape));
        }
        const Tensor matrix = Tensor::from_vector(make_input(12, 10), {3, 4}, device);
        check_unary_case(op, matrix.transpose(0, 1), "transpose");
        check_unary_case(op, matrix.slice(0, 1, 3), "contiguous offset view");
        check_unary_case(op, matrix.slice(1, 1, 4, 2), "strided offset view");
        const Tensor row = Tensor::from_vector(std::vector<float>{1, -2, 3}, {1, 3}, device);
        check_unary_case(op, row.expand({4, 3}), "broadcast view");
    }
}

void test_unary_special_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float large = std::numeric_limits<float>::max();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values{0.0f, -0.0f, inf, -inf, tiny, -tiny, large, -large,
                                    1.1f, -1.1f, nan, -nan};
    const Tensor x = Tensor::from_vector(values, {static_cast<std::int64_t>(values.size())});
    for (const UnaryOp& op : unary_ops) {
        check_unary_case(op, x, "special values");
    }
    test::require(same_bits(neg(neg(x)).to_vector<float>(), values),
                  "double negation preserves every input bit");
}

void test_unary_known_values() {
    const Tensor x = Tensor::from_vector(std::vector<float>{-2.0f, -0.0f, 0.0f, 3.0f}, {4});
    test::require(same_bits(mensurflow::abs(x).to_vector<float>(), {2.0f, 0.0f, 0.0f, 3.0f}),
                  "known abs values and positive zeros");
    test::require(same_bits(square(x).to_vector<float>(), {4.0f, 0.0f, 0.0f, 9.0f}),
                  "known square values and positive zeros");
    test::require(same_bits(relu(x).to_vector<float>(), {0.0f, -0.0f, 0.0f, 3.0f}),
                  "known relu values and preserved negative zero");
}

void test_transcendental_known_values() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float tiny = std::numeric_limits<float>::denorm_min();
    const auto roots = mensurflow::sqrt(Tensor::from_vector(
        std::vector<float>{-inf, -4, -0.0f, 0.0f, 4, inf, nan}, {7})).to_vector<float>();
    test::require(std::isnan(roots[0]) && std::isnan(roots[1]) && std::isnan(roots[6]),
                  "sqrt domain errors and NaN propagation");
    test::require(roots[2] == 0 && std::signbit(roots[2]) && roots[3] == 0 &&
                      !std::signbit(roots[3]) && roots[4] == 2 && roots[5] == inf,
                  "sqrt preserves signed zeros and handles perfect squares and infinity");

    const auto powers = mensurflow::exp(Tensor::from_vector(
        std::vector<float>{-inf, -1000, -100, -0.0f, 0.0f, 88, 89, 1000, inf, nan},
        {10})).to_vector<float>();
    test::require(powers[0] == 0 && !std::signbit(powers[0]) && powers[1] == 0 && powers[2] > 0,
                  "exp underflow and representable subnormal results");
    test::require(powers[3] == 1 && powers[4] == 1 && std::isfinite(powers[5]) &&
                      powers[6] == inf && powers[7] == inf && powers[8] == inf &&
                      std::isnan(powers[9]),
                  "exp zero, float32 overflow, infinities, and NaN");

    const auto probabilities = sigmoid(Tensor::from_vector(
        std::vector<float>{-inf, -1000, -100, -2, -0.0f, 0.0f, 2, 1000, inf, nan},
        {10})).to_vector<float>();
    const std::vector<float> expected{0, 0, 27 * tiny, 0.119202922f, 0.5f,
                                      0.5f, 0.880797078f, 1, 1, nan};
    test::require(test::matches_f32(probabilities, expected, 4e-7f),
                  "known stable sigmoid values, including the negative tail");
    test::require(probabilities[2] > 0, "sigmoid(-100) must retain a representable subnormal");
    for (std::size_t i = 0; i + 1 < probabilities.size(); ++i) {
        test::require(std::isfinite(probabilities[i]) && probabilities[i] >= 0 &&
                          probabilities[i] <= 1,
                      "sigmoid finite outputs stay in [0, 1], including extreme inputs");
    }
}

void check_scaled_output(const Tensor& x, const Tensor& y, const Tensor& out,
                         const Shape& shape, const std::vector<float>& want) {
    test::require(out.shape() == shape && out.dtype() == x.dtype() && out.device() == x.device(),
                  "scaled/bias result shape, dtype, and device");
    test::require(out.is_contiguous() && !out.shares_storage(x) && !out.shares_storage(y),
                  "scaled/bias result has fresh contiguous storage");
    test::require(test::matches_f32(out.to_vector<float>(), want),
                  "scaled/bias result matches independent reference");
}

void check_axpy_case(float alpha, const Tensor& x, const Tensor& y) {
    const auto before_x = x.to_vector<float>(), before_y = y.to_vector<float>();
    const auto want = reference::broadcast_binary(before_x, x.shape(), before_y, y.shape(),
        [alpha](float a, float b) { return reference::axpy_value(alpha, a, b); });
    check_scaled_output(x, y, axpy(alpha, x, y), want.shape, want.values);
    test::require(same_bits(before_x, x.to_vector<float>()) &&
                      same_bits(before_y, y.to_vector<float>()), "axpy preserves both inputs");
}

void check_bias_case(const Tensor& x, const Tensor& bias) {
    const auto before_x = x.to_vector<float>(), before_bias = bias.to_vector<float>();
    std::vector<float> want(before_x.size());
    reference::bias_add(before_x.data(), before_bias.data(), want.data(), x.numel(), x.size(1));
    check_scaled_output(x, bias, bias_add(x, bias), x.shape(), want);
    test::require(same_bits(before_x, x.to_vector<float>()) &&
                      same_bits(before_bias, bias.to_vector<float>()), "bias_add preserves inputs");
}

void test_axpy_and_bias(Device device) {
    const Tensor matrix = Tensor::from_vector(make_input(12, 20), {3, 4}, device);
    const Tensor vector = Tensor::from_vector(make_input(8, 21), {8}, device);
    const Tensor bias = vector.slice(0, 0, 8, 2); // noncontiguous [4] bias
    const bool has_axpy = detail::Registry::instance().dispatcher().find(
        {detail::OpId::Axpy, device.type, DType::F32}) != nullptr;
    const bool has_bias = detail::Registry::instance().dispatcher().find(
        {detail::OpId::BiasAdd, device.type, DType::F32}) != nullptr;
    if (has_axpy) {
        for (float alpha : {0.0f, -2.0f, 0.25f, 1.25f}) {
            check_axpy_case(alpha, matrix, bias);
        }
        check_axpy_case(-2, Tensor::from_vector(std::vector<float>{3}, {}, device), matrix);
        check_axpy_case(0.5f, matrix.transpose(0, 1),
                        Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device));
        check_axpy_case(2, matrix.slice(1, 1, 4, 2), vector.slice(0, 1, 5, 2));
        check_axpy_case(2, bias.unsqueeze(0).expand({3, 4}), matrix);
        check_axpy_case(2, Tensor::empty({0, 4}, DType::F32, device), bias);
    } else {
        for (const Shape& shape : {Shape{3, 4}, Shape{0, 4}}) {
            const Tensor x = Tensor::empty(shape, DType::F32, device);
            test::require_throws<detail::NotImplemented>([&] { axpy(2, x, bias); },
                                                         "axpy without a kernel, including empty");
        }
    }
    if (has_bias) {
        check_bias_case(matrix, bias);
        check_bias_case(matrix.slice(0, 1, 3), bias);
        check_bias_case(matrix.slice(1, 1, 4, 2), vector.slice(0, 1, 5, 2));
        check_bias_case(matrix.transpose(0, 1),
                         Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device));
        check_bias_case(bias.unsqueeze(0).expand({3, 4}), bias);
        check_bias_case(matrix,
                        Tensor::from_vector(std::vector<float>{2}, {1}, device).expand({4}));
        for (const Shape& shape : {Shape{0, 4}, Shape{3, 0}, Shape{0, 0}}) {
            check_bias_case(Tensor::empty(shape, DType::F32, device),
                            Tensor::from_vector(
                                std::vector<float>(static_cast<std::size_t>(shape[1]), 0.0f),
                                {shape[1]}, device));
        }
    } else {
        for (const Shape& shape : {Shape{3, 4}, Shape{0, 4}, Shape{3, 0}}) {
            const Tensor x = Tensor::empty(shape, DType::F32, device);
            const Tensor b = Tensor::empty({shape[1]}, DType::F32, device);
            test::require_throws<detail::NotImplemented>([&] { bias_add(x, b); },
                                                         "bias_add missing kernel, even if empty");
        }
    }
    const Tensor ints = Tensor::empty({3, 4}, DType::I32, device);
    const Tensor int_bias = Tensor::empty({4}, DType::I32, device);
    test::require_throws<std::invalid_argument>([&] { axpy(2, Tensor{}, bias); },
                                                "axpy undefined x");
    test::require_throws<std::invalid_argument>([&] { axpy(2, matrix, Tensor{}); },
                                                "axpy undefined y");
    test::require_throws<std::invalid_argument>([&] { axpy(2, matrix, vector); },
                                                "axpy bad broadcast");
    test::require_throws<std::invalid_argument>([&] { axpy(2, matrix, int_bias); },
                                                "axpy dtype mismatch");
    test::require_throws<std::invalid_argument>([&] { bias_add(Tensor{}, bias); },
                                                "bias_add undefined x");
    test::require_throws<std::invalid_argument>([&] { bias_add(matrix, Tensor{}); },
                                                "bias_add undefined bias");
    test::require_throws<std::invalid_argument>([&] { bias_add(vector, bias); }, "bias_add x rank");
    test::require_throws<std::invalid_argument>([&] { bias_add(matrix, bias.unsqueeze(0)); },
                                                "bias_add bias must be a vector");
    test::require_throws<std::invalid_argument>([&] { bias_add(matrix, vector); },
                                                "bias_add wrong features");
    test::require_throws<std::invalid_argument>([&] { bias_add(matrix, int_bias); },
                                                "bias_add dtype mismatch");
    test::require_throws<detail::NotImplemented>([&] { axpy(2, ints, int_bias); },
                                                "axpy unsupported dtype");
    test::require_throws<detail::NotImplemented>([&] { bias_add(ints, int_bias); },
                                                "bias_add unsupported dtype");
    const Tensor empty = Tensor::empty({0, 4}, DType::I32, device);
    test::require_throws<detail::NotImplemented>([&] { axpy(2, empty, int_bias); },
                                                "empty axpy unsupported dtype");
    test::require_throws<detail::NotImplemented>([&] { bias_add(empty, int_bias); },
                                                "empty bias_add unsupported dtype");
}

void test_axpy_fused_values() {
    const Tensor x = Tensor::from_vector(std::vector<float>{0x1.fffffcp-1f}, {1});
    const Tensor y = Tensor::from_vector(std::vector<float>{-1}, {1});
    test::require(axpy(0x1.000002p0f, x, y).item<float>() == -0x1p-46f,
                  "public axpy retains cancellation bits with fused rounding");
    const float large = std::numeric_limits<float>::max();
    const Tensor big = Tensor::from_vector(std::vector<float>{large}, {});
    const Tensor negative = Tensor::from_vector(std::vector<float>{-large}, {});
    test::require(axpy(2, big, negative).item<float>() == large,
                  "public axpy does not overflow its intermediate product");
    const Tensor infinite = Tensor::from_vector(
        std::vector<float>{std::numeric_limits<float>::infinity()}, {});
    test::require(std::isnan(axpy(0, infinite, big).item<float>()),
                  "zero alpha still evaluates IEEE 754 zero times infinity");
}

void test_ieee_division() {
    // x / 0 is inf or NaN, as in IEEE 754. Checked on the CPU, where rounding is exact.
    const Device cpu{DeviceType::CPU, 0};
    const Tensor x = Tensor::from_vector(std::vector<float>{1, -1, 0}, {3}, cpu);
    const Tensor zero = Tensor::from_vector(std::vector<float>{0, 0, 0}, {3}, cpu);
    const auto q = div(x, zero).to_vector<float>();
    test::require(std::isinf(q[0]) && q[0] > 0 && std::isinf(q[1]) && q[1] < 0 && std::isnan(q[2]),
                  "1/0 = inf, -1/0 = -inf, 0/0 = nan");
}

void test_errors(Device device) {
    const Tensor m = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
    const Tensor v4 = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {4}, device);
    const Tensor ints = Tensor::from_vector(std::vector<std::int32_t>{1, 2, 3}, {3}, device);
    for (const Op& op : ops) {
        const std::string name = op.name;
        test::require_throws<std::invalid_argument>([&] { op.apply(m, v4); },
                                                    name + " shapes do not broadcast");
        test::require_throws<std::invalid_argument>([&] { op.apply(Tensor(), m); },
                                                    name + " undefined input");
        test::require_throws<std::invalid_argument>([&] { op.apply(ints, m); },
                                                    name + " dtype mismatch");
        // No int32 kernel exists on any backend yet; the failure must be a clear error.
        test::require_throws<std::runtime_error>([&] { op.apply(ints, ints); },
                                                 name + " unsupported dtype");
    }
    for (const UnaryOp& op : unary_ops) {
        test::require_throws<std::invalid_argument>([&] { op.apply(Tensor{}); },
                                                    std::string(op.name) + " undefined input");
        test::require_throws<detail::NotImplemented>([&] { op.apply(ints); },
                                                     std::string(op.name) + " unsupported dtype");
        const Tensor empty_ints = Tensor::empty({0}, DType::I32, device);
        test::require_throws<detail::NotImplemented>(
            [&] { op.apply(empty_ints); }, std::string(op.name) + " empty unsupported dtype");
    }
}

void test_mixed_devices() {
    const auto devices = list_devices();
    if (devices.size() < 2) {
        return; // Needs a GPU build.
    }
    const Tensor on_cpu = Tensor::from_vector(std::vector<float>{1, 2}, {2}, devices[0]);
    const Tensor elsewhere = on_cpu.to(devices[1]);
    test::require_throws<std::invalid_argument>([&] { add(on_cpu, elsewhere); }, "mixed devices");
    test::require_throws<std::invalid_argument>([&] { axpy(2, on_cpu, elsewhere); },
                                                "axpy mixed devices");
    const Tensor column = on_cpu.view({2, 1});
    const Tensor bias = Tensor::from_vector(std::vector<float>{1}, {1}, devices[1]);
    test::require_throws<std::invalid_argument>([&] { bias_add(column, bias); },
                                                "bias_add mixed devices");
    test::require(add(elsewhere, elsewhere).to_vector<float>() == (std::vector<float>{2, 4}),
                  "moved tensor works");
}

void test_added_cpu_ops() {
    const Tensor x = Tensor::from_vector(std::vector<float>{-4.0f, -1.0f, 0.0f, 1.0f, 4.0f}, {5});
    const Tensor y = Tensor::from_vector(std::vector<float>{-2.0f, -2.0f, 2.0f, 2.0f, 2.0f}, {5});
    const auto values = x.to_vector<float>();
    const auto logs = log(Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 4.0f}, {3}))
                          .to_vector<float>();
    test::require(std::abs(logs[1] - std::log(2.0f)) < 1e-6f, "log");
    test::require(std::abs(rsqrt(Tensor::from_vector(std::vector<float>{1.0f, 4.0f}, {2}))
                               .to_vector<float>()[1] -
                           0.5f) < 1e-6f,
                  "rsqrt");
    test::require(std::abs(tanh(x).to_vector<float>()[3] - std::tanh(1.0f)) < 1e-6f, "tanh");
    test::require(minimum(x, y).to_vector<float>() ==
                      std::vector<float>{-4.0f, -2.0f, 0.0f, 1.0f, 2.0f},
                  "minimum");
    test::require(maximum(x, y).to_vector<float>() ==
                      std::vector<float>{-2.0f, -1.0f, 2.0f, 2.0f, 4.0f},
                  "maximum");
    test::require(clamp(x, -1.0f, 1.0f).to_vector<float>() ==
                      std::vector<float>{-1.0f, -1.0f, 0.0f, 1.0f, 1.0f},
                  "clamp");
    test::require(values[0] == -4.0f, "input unchanged");
    test::require_throws<std::invalid_argument>([&] { clamp(x, 2.0f, 1.0f); },
                                                "invalid clamp bounds");
}

} // namespace

int main() {
    return test::run("elementwise_ops_test", [] {
        for (Device device : list_devices()) {
            std::cout << "  elementwise ops on " << to_string(device) << '\n';
            for (const Op& op : ops) {
                if (!supported(op, device)) {
                    std::cout << "    no " << op.name << " kernel on " << to_string(device) << '\n';
                    check_unsupported(op, device);
                }
            }
            for (const UnaryOp& op : unary_ops) {
                if (!supported(op, device)) {
                    std::cout << "    no " << op.name << " kernel on " << to_string(device) << '\n';
                    check_unsupported(op, device);
                }
            }
            test_shapes(device);
            test_views(device);
            test_unary_shapes_and_views(device);
            test_axpy_and_bias(device);
            test_errors(device);
        }
        test_ieee_division();
        test_axpy_fused_values();
        test_unary_special_values();
        test_unary_known_values();
        test_transcendental_known_values();
        test_added_cpu_ops();
        test_mixed_devices();
    });
}
