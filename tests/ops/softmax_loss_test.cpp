// softmax, log_softmax, the losses and attention: CPU kernels against double-precision references at
// several thread counts and sizes around the parallel cutoff, numerical edge cases, the public API
// (axes, views, reductions, empty inputs, errors) and autograd.
#include "backends/cpu/ops/loss/loss.hpp"
#include "backends/cpu/ops/softmax/softmax.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/softmax_loss.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

constexpr float inf = std::numeric_limits<float>::infinity();

std::vector<float> values(std::int64_t n, int seed = 0, float spread = 3.0f) {
    std::vector<float> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const int k = static_cast<int>((i * 37 + seed * 101) % 97) - 48;
        out[static_cast<std::size_t>(i)] = spread * static_cast<float>(k) / 48.0f;
    }
    return out;
}

std::vector<float> probabilities(std::int64_t n) {
    std::vector<float> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<float>((i * 7) % 11) / 10.0f; // 0 .. 1
    }
    return out;
}

std::vector<std::int64_t> labels(std::int64_t rows, std::int64_t classes) {
    std::vector<std::int64_t> out(static_cast<std::size_t>(rows));
    for (std::int64_t r = 0; r < rows; ++r) {
        out[static_cast<std::size_t>(r)] = (r * 13 + 5) % classes;
    }
    return out;
}

// |actual - expected| <= rel * |expected| + abs, with NaN matching NaN and infinities matching.
bool near(double actual, double expected, double rel = 2e-5, double abs = 1e-6) {
    if (std::isnan(expected) || std::isnan(actual)) {
        return std::isnan(expected) && std::isnan(actual);
    }
    if (std::isinf(expected) || std::isinf(actual)) {
        return actual == expected;
    }
    return std::abs(actual - expected) <= rel * std::abs(expected) + abs;
}

void require_near(const std::vector<float>& actual, const std::vector<float>& expected,
                  const std::string& what, double rel = 2e-5, double abs = 1e-6) {
    test::require(actual.size() == expected.size(), what + ": size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!near(actual[i], expected[i], rel, abs)) {
            throw std::runtime_error(what + ": element " + std::to_string(i) + " is " +
                                     std::to_string(actual[i]) + ", expected " +
                                     std::to_string(expected[i]));
        }
    }
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> to_float(const std::vector<double>& v) {
    return std::vector<float>(v.begin(), v.end());
}

const int all_threads = cpu::max_threads();
const std::vector<int> thread_counts = {1, 2, 3, all_threads};

// ---- kernels ---------------------------------------------------------------------------------

void test_softmax_kernels() {
    // {outer, length, inner}: tiny, around the 32768-element cutoff, long lines, strided lines.
    const std::int64_t shapes[][3] = {{1, 1, 1},     {3, 7, 1},     {1000, 32, 1}, {1025, 32, 1},
                                      {64, 1024, 1}, {2, 20000, 1}, {4, 5, 6},     {3, 3000, 7},
                                      {1, 0, 1},     {0, 4, 1}};
    for (const auto& s : shapes) {
        const std::int64_t outer = s[0], length = s[1], inner = s[2];
        const std::int64_t n = outer * length * inner;
        const auto x = values(n, 1);
        const auto g = values(n, 2, 1.0f);
        const std::string where =
            std::to_string(outer) + "x" + std::to_string(length) + "x" + std::to_string(inner);
        for (const bool log : {false, true}) {
            const auto expected = reference::softmax(x, outer, length, inner, log);
            std::vector<float> first;
            std::vector<float> first_backward;
            for (const int threads : thread_counts) {
                cpu::set_max_threads(threads);
                std::vector<float> out(x.size(), 123.0f);
                (log ? cpu::log_softmax_f32 : cpu::softmax_f32)(x.data(), out.data(), outer, length,
                                                                inner);
                require_near(out, expected, (log ? "log_softmax " : "softmax ") + where, 2e-5,
                             1e-6);
                std::vector<float> backward(x.size(), 123.0f);
                (log ? cpu::log_softmax_backward_f32 : cpu::softmax_backward_f32)(
                    out.data(), g.data(), backward.data(), outer, length, inner);
                require_near(
                    backward, reference::softmax_backward(out, g, outer, length, inner, log),
                    (log ? "log_softmax_backward " : "softmax_backward ") + where, 1e-4, 1e-5);
                // One line is always one thread's work: the bits do not depend on the thread count.
                if (first.empty()) {
                    first = out;
                    first_backward = backward;
                } else {
                    test::require(same_bits(out, first) && same_bits(backward, first_backward),
                                  where + ": same bits at every thread count");
                }
            }
        }
    }
    cpu::set_max_threads(all_threads);

    // Stability and special values, one row each.
    const std::vector<float> rows = {1000.0f,  1001.0f,  1002.0f,                    // huge logits
                                     -1000.0f, -1001.0f, -1002.0f, -inf, -inf, -inf, // fully masked
                                     -inf,     0.0f,     -inf,                       // one visible
                                     1.0f,     NAN,      2.0f, // NaN poisons the row
                                     -inf,     NAN,      -inf};
    std::vector<float> out(rows.size()), log_out(rows.size());
    cpu::softmax_f32(rows.data(), out.data(), 6, 3, 1);
    cpu::log_softmax_f32(rows.data(), log_out.data(), 6, 3, 1);
    require_near(std::vector<float>(out.begin(), out.begin() + 6),
                 {0.09003057f, 0.24472847f, 0.66524096f, 0.66524096f, 0.24472847f, 0.09003057f},
                 "softmax shift invariance");
    require_near(std::vector<float>(log_out.begin(), log_out.begin() + 3),
                 {-2.40760596f, -1.40760596f, -0.40760596f}, "log_softmax at 1000");
    test::require(out[6] == 0.0f && out[7] == 0.0f && out[8] == 0.0f, "masked row: zeros");
    test::require(std::isinf(log_out[6]) && log_out[6] < 0, "masked row: log -inf");
    test::require(out[9] == 0.0f && out[10] == 1.0f && out[11] == 0.0f, "one visible key");
    test::require(log_out[10] == 0.0f && std::isinf(log_out[9]), "one visible key, log");
    for (int i = 12; i < 18; ++i) {
        test::require(std::isnan(out[i]) && std::isnan(log_out[i]), "NaN poisons its row");
    }
}

void test_loss_kernels() {
    // Cross-entropy: {rows, classes}, including rows wider than one reduction chunk.
    const std::int64_t shapes[][2] = {{1, 1},     {5, 3},     {2000, 17}, {2048, 16},
                                      {3, 20000}, {40000, 2}, {0, 4}};
    for (const auto& s : shapes) {
        const std::int64_t rows = s[0], classes = s[1];
        const auto x = values(rows * classes, 3, 6.0f);
        const auto y = labels(rows, classes);
        const auto per_row = reference::cross_entropy(x, y, classes);
        const std::string where = std::to_string(rows) + "x" + std::to_string(classes);
        for (int reduction = 0; reduction <= 2; ++reduction) {
            std::vector<float> first;
            for (const int threads : thread_counts) {
                cpu::set_max_threads(threads);
                std::vector<float> out(reduction == 0 ? static_cast<std::size_t>(rows) : 1, 123.0f);
                cpu::cross_entropy_f32(x.data(), y.data(), out.data(), rows, classes, reduction);
                if (reduction == 0) {
                    require_near(out, to_float(per_row), "cross_entropy none " + where);
                } else {
                    const double total = reference::total(per_row);
                    const double expected =
                        reduction == 2 ? total / static_cast<double>(rows) : total;
                    test::require(rows == 0 && reduction == 2 ? std::isnan(out[0])
                                                              : near(out[0], expected, 1e-5),
                                  "cross_entropy reduced " + where);
                }
                if (first.empty()) {
                    first = out;
                } else {
                    test::require(same_bits(out, first),
                                  "cross_entropy same bits at every thread count " + where);
                }

                // Backward with the upstream gradient each reduction passes.
                const auto upstream = values(reduction == 0 ? rows : 1, 4, 2.0f);
                std::vector<double> per_row_upstream(static_cast<std::size_t>(rows));
                for (std::int64_t r = 0; r < rows; ++r) {
                    per_row_upstream[static_cast<std::size_t>(r)] =
                        reduction == 0   ? upstream[static_cast<std::size_t>(r)]
                        : reduction == 1 ? upstream[0]
                                         : upstream[0] / static_cast<double>(rows);
                }
                std::vector<float> dx(x.size(), 123.0f);
                cpu::cross_entropy_backward_f32(x.data(), y.data(), upstream.data(), dx.data(),
                                                rows, classes, reduction);
                require_near(dx, reference::cross_entropy_backward(x, y, classes, per_row_upstream),
                             "cross_entropy_backward " + where, 1e-4, 1e-6);
            }
        }
    }

    // MSE and BCE: sizes around the cutoff and the chunk.
    for (const std::int64_t n : {0, 1, 1000, 16384, 32769, 100003}) {
        const auto x = values(n, 5, 8.0f);
        const auto t = probabilities(n);
        const auto mse = reference::mse(x, t);
        const auto bce = reference::bce_with_logits(x, t);
        const std::string where = std::to_string(n);
        for (int reduction = 0; reduction <= 2; ++reduction) {
            std::vector<float> first_mse, first_bce;
            for (const int threads : thread_counts) {
                cpu::set_max_threads(threads);
                const std::size_t size = reduction == 0 ? static_cast<std::size_t>(n) : 1;
                std::vector<float> out_mse(size, 123.0f), out_bce(size, 123.0f);
                cpu::mse_loss_f32(x.data(), t.data(), out_mse.data(), n, reduction);
                cpu::bce_with_logits_f32(x.data(), t.data(), out_bce.data(), n, reduction);
                if (reduction == 0) {
                    require_near(out_mse, to_float(mse), "mse none " + where);
                    require_near(out_bce, to_float(bce), "bce none " + where, 2e-5, 2e-6);
                } else {
                    const double div = reduction == 2 ? static_cast<double>(n) : 1.0;
                    if (n == 0 && reduction == 2) {
                        test::require(std::isnan(out_mse[0]) && std::isnan(out_bce[0]),
                                      "mean of nothing is NaN");
                    } else {
                        test::require(near(out_mse[0], reference::total(mse) / div, 1e-5),
                                      "mse reduced " + where);
                        test::require(near(out_bce[0], reference::total(bce) / div, 1e-5),
                                      "bce reduced " + where);
                    }
                }
                if (first_mse.empty()) {
                    first_mse = out_mse;
                    first_bce = out_bce;
                } else {
                    test::require(same_bits(out_mse, first_mse) && same_bits(out_bce, first_bce),
                                  "mse/bce same bits at every thread count " + where);
                }

                const auto upstream = values(reduction == 0 ? n : 1, 6, 2.0f);
                std::vector<float> d_mse(x.size()), d_bce(x.size());
                std::vector<float> e_mse(x.size()), e_bce(x.size());
                for (std::size_t i = 0; i < x.size(); ++i) {
                    const double u = reduction == 0   ? upstream[i]
                                     : reduction == 1 ? upstream[0]
                                                      : upstream[0] / static_cast<double>(n);
                    e_mse[i] = static_cast<float>(2.0 * (static_cast<double>(x[i]) - t[i]) * u);
                    e_bce[i] = static_cast<float>((reference::sigmoid(x[i]) - t[i]) * u);
                }
                cpu::mse_loss_backward_f32(x.data(), t.data(), upstream.data(), d_mse.data(), n,
                                           reduction);
                cpu::bce_with_logits_backward_f32(x.data(), t.data(), upstream.data(), d_bce.data(),
                                                  n, reduction);
                require_near(d_mse, e_mse, "mse backward " + where, 1e-5, 1e-7);
                require_near(d_bce, e_bce, "bce backward " + where, 1e-5, 1e-7);
            }
        }
    }
    cpu::set_max_threads(all_threads);

    // BCE far out on the logit axis: exact losses, no overflow, no rounding to zero.
    const std::vector<float> x = {100.0f, -100.0f, 100.0f, -100.0f, 30.0f, 0.0f};
    const std::vector<float> t = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.5f};
    std::vector<float> loss(x.size()), grad(x.size());
    const std::vector<float> ones(x.size(), 1.0f);
    cpu::bce_with_logits_f32(x.data(), t.data(), loss.data(), 6, 0);
    cpu::bce_with_logits_backward_f32(x.data(), t.data(), ones.data(), grad.data(), 6, 0);
    require_near(loss, {100.0f, 100.0f, 0.0f, 0.0f, 9.357623e-14f, 0.6931472f}, "bce extremes",
                 1e-5, 1e-30);
    // At x = 30, sigmoid(x) - 1 = -9.36e-14: computing sigmoid first would round it to exactly 0
    // (as PyTorch does); the kernel avoids the cancellation and keeps it.
    require_near(grad, {1.0f, -1.0f, 0.0f, 0.0f, -9.357623e-14f, 0.0f}, "bce extreme gradients",
                 1e-5, 1e-30);
    // Cross-entropy with logits of 1e4: still finite and exact.
    const std::vector<float> big = {10000.0f, 0.0f, -10000.0f};
    const std::vector<std::int64_t> label = {1};
    float ce = 0.0f;
    cpu::cross_entropy_f32(big.data(), label.data(), &ce, 1, 3, 0);
    test::require(near(ce, 10000.0, 1e-6), "cross_entropy with huge logits");
}

// ---- public API --------------------------------------------------------------------------------

void test_softmax_api() {
    const auto raw = values(24, 7);
    const Tensor x = Tensor::from_vector(raw, {2, 3, 4});
    // Last axis, a middle axis and the first axis, as [outer, length, inner].
    require_near(softmax(x).to_vector<float>(), reference::softmax(raw, 6, 4, 1, false), "axis -1");
    require_near(softmax(x, 1).to_vector<float>(), reference::softmax(raw, 2, 3, 4, false),
                 "axis 1");
    require_near(log_softmax(x, 0).to_vector<float>(), reference::softmax(raw, 1, 2, 12, true),
                 "log axis 0");
    test::require(softmax(x, -3).shape() == x.shape(), "negative axis");
    // A transposed view: the result equals softmax of the packed copy.
    const Tensor t = x.transpose(1, 2);
    require_near(softmax(t).to_vector<float>(), softmax(t.contiguous()).to_vector<float>(),
                 "transposed input");
    // 0-d and empty.
    const Tensor scalar = Tensor::from_vector(std::vector<float>{5.0f}, {});
    test::require(softmax(scalar).item<float>() == 1.0f, "0-d softmax is 1");
    test::require(log_softmax(scalar, 0).item<float>() == 0.0f, "0-d log_softmax is 0");
    test::require(softmax(Tensor::empty({3, 0})).shape() == Shape{3, 0}, "empty");
    // Backward ops through the API.
    const Tensor y = softmax(x);
    const Tensor g = Tensor::from_vector(values(24, 8), {2, 3, 4});
    require_near(softmax_backward(y, g).to_vector<float>(),
                 reference::softmax_backward(y.to_vector<float>(), values(24, 8), 6, 4, 1, false),
                 "softmax_backward API");
    const Tensor ly = log_softmax(x, 1);
    require_near(log_softmax_backward(ly, g, 1).to_vector<float>(),
                 reference::softmax_backward(ly.to_vector<float>(), values(24, 8), 2, 3, 4, true),
                 "log_softmax_backward API", 1e-4, 1e-5);

    test::require_throws<std::invalid_argument>([&] { softmax(x, 3); }, "axis out of range");
    test::require_throws<std::invalid_argument>([&] { softmax(x, -4); },
                                                "negative axis out of range");
    test::require_throws<std::invalid_argument>([&] { softmax(Tensor::zeros({3}, DType::I32)); },
                                                "non-F32 input");
    test::require_throws<std::invalid_argument>([&] { softmax(Tensor()); }, "undefined input");
    test::require_throws<std::invalid_argument>([&] { softmax_backward(y, Tensor::zeros({2, 3})); },
                                                "backward shape mismatch");
}

void test_loss_api() {
    const auto raw = values(12, 9, 4.0f);
    const Tensor logits = Tensor::from_vector(raw, {4, 3});
    const std::vector<std::int64_t> raw_labels = {2, 0, 1, 2};
    const Tensor y = Tensor::from_vector(raw_labels, {4});
    const auto per_row = reference::cross_entropy(raw, raw_labels, 3);
    test::require(near(cross_entropy(logits, y).item<float>(), reference::total(per_row) / 4.0),
                  "cross_entropy mean");
    test::require(cross_entropy(logits, y).shape() == Shape{}, "mean gives a scalar");
    test::require(
        near(cross_entropy(logits, y, Reduction::Sum).item<float>(), reference::total(per_row)),
        "cross_entropy sum");
    require_near(cross_entropy(logits, y, Reduction::None).to_vector<float>(), to_float(per_row),
                 "cross_entropy none");
    // Views: transposed logits are packed.
    const Tensor t = Tensor::from_vector(raw, {3, 4}).transpose(0, 1);
    test::require(near(cross_entropy(t, y).item<float>(),
                       reference::total(reference::cross_entropy(t.contiguous().to_vector<float>(),
                                                                 raw_labels, 3)) /
                           4.0),
                  "cross_entropy transposed logits");
    test::require(
        std::isnan(
            cross_entropy(Tensor::empty({0, 3}), Tensor::empty({0}, DType::I64)).item<float>()),
        "cross_entropy mean of no rows is NaN");

    const Tensor a = Tensor::from_vector(values(6, 1), {2, 3});
    const Tensor b = Tensor::from_vector(probabilities(6), {2, 3});
    test::require(near(mse_loss(a, b).item<float>(),
                       reference::total(reference::mse(values(6, 1), probabilities(6))) / 6.0),
                  "mse mean");
    test::require(mse_loss(a, b, Reduction::None).shape() == Shape{2, 3}, "mse none shape");
    test::require(
        near(binary_cross_entropy_with_logits(a, b, Reduction::Sum).item<float>(),
             reference::total(reference::bce_with_logits(values(6, 1), probabilities(6)))),
        "bce sum");
    test::require(mse_loss(Tensor::empty({0}), Tensor::empty({0}), Reduction::Sum).item<float>() ==
                      0.0f,
                  "sum of nothing is 0");

    // Backward ops through the API and their gradient shapes.
    const Tensor scalar_one = Tensor::from_vector(std::vector<float>{1.0f}, {});
    test::require(cross_entropy_backward(logits, y, scalar_one).shape() == Shape{4, 3},
                  "cross_entropy_backward shape");
    test::require(mse_loss_backward(a, b, Tensor::ones({2, 3}), Reduction::None).shape() ==
                      Shape{2, 3},
                  "mse_loss_backward none");

    using IA = std::invalid_argument;
    test::require_throws<IA>(
        [&] {
            cross_entropy(logits, Tensor::from_vector(std::vector<std::int64_t>{0, 1, 3, 0}, {4}));
        },
        "label out of range");
    test::require_throws<IA>(
        [&] {
            cross_entropy(logits, Tensor::from_vector(std::vector<std::int64_t>{0, -1, 1, 0}, {4}));
        },
        "negative label");
    test::require_throws<IA>([&] { cross_entropy(logits, Tensor::zeros({4}, DType::F32)); },
                             "float labels");
    test::require_throws<IA>([&] { cross_entropy(logits, Tensor::zeros({3}, DType::I64)); },
                             "label count");
    test::require_throws<IA>([&] { cross_entropy(Tensor::zeros({12}), y); }, "1-d logits");
    test::require_throws<IA>([&] { cross_entropy(Tensor::empty({4, 0}), y); }, "no classes");
    test::require_throws<IA>([&] { mse_loss(a, Tensor::zeros({3, 2})); }, "shape mismatch");
    test::require_throws<IA>(
        [&] { binary_cross_entropy_with_logits(a, Tensor::zeros({2, 3}, DType::I32)); },
        "integer target");
    test::require_throws<IA>([&] { cross_entropy_backward(logits, y, Tensor::ones({4})); },
                             "mean backward wants one value");
    test::require_throws<IA>([&] { mse_loss_backward(a, b, Tensor::ones({2}), Reduction::None); },
                             "none backward wants the loss shape");
}

// ---- autograd ---------------------------------------------------------------------------------

void test_autograd() {
    // softmax: d/dx sum(softmax(x) * w) = softmax_backward(y, w).
    const auto raw = values(15, 11);
    const auto w = values(15, 12, 1.0f);
    Tensor x = Tensor::from_vector(raw, {3, 5});
    x.set_requires_grad();
    const Tensor weights = Tensor::from_vector(w, {3, 5});
    sum(mul(softmax(x), weights)).backward();
    require_near(
        x.grad().to_vector<float>(),
        reference::softmax_backward(reference::softmax(raw, 3, 5, 1, false), w, 3, 5, 1, false),
        "softmax autograd", 1e-4, 1e-6);

    Tensor lx = Tensor::from_vector(raw, {3, 5});
    lx.set_requires_grad();
    sum(mul(log_softmax(lx, 0), weights)).backward();
    require_near(
        lx.grad().to_vector<float>(),
        reference::softmax_backward(reference::softmax(raw, 1, 3, 5, true), w, 1, 3, 5, true),
        "log_softmax autograd over axis 0", 1e-4, 1e-6);

    // cross_entropy (mean) against finite differences in double.
    const std::vector<std::int64_t> raw_labels = {4, 0, 2};
    Tensor logits = Tensor::from_vector(raw, {3, 5});
    logits.set_requires_grad();
    Tensor labels_tensor = Tensor::from_vector(raw_labels, {3});
    const Tensor loss = cross_entropy(logits, labels_tensor);
    // Changing the caller's labels after forward does not change the gradient.
    labels_tensor.copy_from(Tensor::from_vector(std::vector<std::int64_t>{0, 0, 0}, {3}));
    loss.backward();
    const auto grad = logits.grad().to_vector<float>();
    for (std::size_t i = 0; i < raw.size(); ++i) {
        auto plus = raw, minus = raw;
        plus[i] += 1e-3f;
        minus[i] -= 1e-3f;
        const double numeric = (reference::total(reference::cross_entropy(plus, raw_labels, 5)) -
                                reference::total(reference::cross_entropy(minus, raw_labels, 5))) /
                               (3.0 * (static_cast<double>(plus[i]) - minus[i]));
        test::require(near(grad[i], numeric, 1e-2, 1e-4),
                      "cross_entropy autograd vs finite difference at " + std::to_string(i));
    }

    // mse: gradients for both sides; target's is the negation.
    Tensor p = Tensor::from_vector(values(6, 13), {6});
    Tensor q = Tensor::from_vector(probabilities(6), {6});
    p.set_requires_grad();
    q.set_requires_grad();
    mse_loss(p, q, Reduction::Sum).backward();
    const auto pv = p.to_vector<float>(), qv = q.to_vector<float>();
    std::vector<float> expected(6), negated(6);
    for (std::size_t i = 0; i < 6; ++i) {
        expected[i] = 2.0f * (pv[i] - qv[i]);
        negated[i] = -expected[i];
    }
    require_near(p.grad().to_vector<float>(), expected, "mse autograd input", 1e-6, 1e-7);
    require_near(q.grad().to_vector<float>(), negated, "mse autograd target", 1e-6, 1e-7);

    // bce (mean): input (sigmoid - t) / n, target -x / n.
    Tensor bx = Tensor::from_vector(values(4, 14, 5.0f), {2, 2});
    Tensor bt = Tensor::from_vector(probabilities(4), {2, 2});
    bx.set_requires_grad();
    bt.set_requires_grad();
    binary_cross_entropy_with_logits(bx, bt).backward();
    const auto xv = bx.to_vector<float>(), tv = bt.to_vector<float>();
    std::vector<float> ex(4), et(4);
    for (std::size_t i = 0; i < 4; ++i) {
        ex[i] = static_cast<float>((reference::sigmoid(xv[i]) - tv[i]) / 4.0);
        et[i] = -xv[i] / 4.0f;
    }
    require_near(bx.grad().to_vector<float>(), ex, "bce autograd input", 1e-5, 1e-7);
    require_near(bt.grad().to_vector<float>(), et, "bce autograd target", 1e-6, 1e-7);

    // Without requires_grad nothing is recorded.
    test::require(!softmax(Tensor::from_vector(raw, {3, 5})).requires_grad(), "no grad recorded");
}

// ---- attention ---------------------------------------------------------------------------------

void test_attention() {
    const std::int64_t batch = 2, heads = 3, lq = 5, lk = 7, d = 8, dv = 4;
    const auto qv = values(batch * heads * lq * d, 21, 1.0f);
    const auto kv = values(batch * heads * lk * d, 22, 1.0f);
    const auto vv = values(batch * heads * lk * dv, 23, 1.0f);
    const Tensor q = Tensor::from_vector(qv, {batch, heads, lq, d});
    const Tensor k = Tensor::from_vector(kv, {batch, heads, lk, d});
    const Tensor v = Tensor::from_vector(vv, {batch, heads, lk, dv});

    // Bool mask [lq, lk], broadcast over batch and heads: key j hidden from query i when (i + j) % 3 == 0,
    // and query 2 sees nothing at all.
    std::vector<std::uint8_t> keep_bytes(static_cast<std::size_t>(lq * lk));
    std::vector<bool> keep(keep_bytes.size());
    for (std::int64_t i = 0; i < lq; ++i) {
        for (std::int64_t j = 0; j < lk; ++j) {
            const bool on = i != 2 && (i + j) % 3 != 0;
            keep_bytes[static_cast<std::size_t>(i * lk + j)] = on;
            keep[static_cast<std::size_t>(i * lk + j)] = on;
        }
    }
    const Tensor mask = Tensor::from_host(keep_bytes.data(), {lq, lk}, DType::Bool);
    std::vector<bool> causal(static_cast<std::size_t>(lq * lk));
    for (std::int64_t i = 0; i < lq; ++i) {
        for (std::int64_t j = 0; j < lk; ++j) {
            causal[static_cast<std::size_t>(i * lk + j)] = j <= i;
        }
    }

    struct Case {
        const Tensor* mask;
        bool causal;
        float scale;
        const char* name;
    };

    for (const Case& c : {Case{nullptr, false, 0.0f, "plain"}, Case{&mask, false, 0.0f, "mask"},
                          Case{nullptr, true, 0.0f, "causal"}, Case{&mask, true, 0.5f, "both"}}) {
        const auto out = attention(q, k, v, c.mask, c.causal, c.scale).to_vector<float>();
        test::require(out.size() == static_cast<std::size_t>(batch * heads * lq * dv),
                      std::string("attention shape ") + c.name);
        std::vector<bool> visible;
        if (c.mask != nullptr || c.causal) {
            visible.assign(keep.size(), true);
            for (std::size_t i = 0; i < keep.size(); ++i) {
                visible[i] = (c.mask == nullptr || keep[i]) && (!c.causal || causal[i]);
            }
        }
        const double scale = c.scale == 0.0f ? 1.0 / std::sqrt(static_cast<double>(d)) : c.scale;
        for (std::int64_t b = 0; b < batch * heads; ++b) {
            const auto slice = [](const std::vector<float>& all, std::int64_t at, std::int64_t n) {
                return std::vector<float>(all.begin() + at * n, all.begin() + (at + 1) * n);
            };
            const auto expected =
                reference::attention(slice(qv, b, lq * d), slice(kv, b, lk * d),
                                     slice(vv, b, lk * dv), lq, lk, d, dv, scale, visible);
            require_near(slice(out, b, lq * dv), expected, std::string("attention ") + c.name, 1e-4,
                         1e-6);
        }
        if (c.mask != nullptr) {
            for (std::int64_t b = 0; b < batch * heads; ++b) {
                for (std::int64_t col = 0; col < dv; ++col) {
                    test::require(out[static_cast<std::size_t>((b * lq + 2) * dv + col)] == 0.0f,
                                  "a query that sees no key gets zeros");
                }
            }
        }
    }

    // k and v shared by every head ([batch, 1, lk, d]) broadcast like gemm_strided_batched.
    const Tensor k1 = Tensor::from_vector(values(batch * lk * d, 24, 1.0f), {batch, 1, lk, d});
    const Tensor v1 = Tensor::from_vector(values(batch * lk * dv, 25, 1.0f), {batch, 1, lk, dv});
    test::require(attention(q, k1, v1).shape() == Shape{batch, heads, lq, dv}, "broadcast k and v");
    test::require(attention(q, k, v, nullptr, false, 1.0f, Execution::CPU).shape() ==
                      Shape{batch, heads, lq, dv},
                  "explicit scale and execution");

    using IA = std::invalid_argument;
    test::require_throws<IA>([&] { attention(q, Tensor::zeros({batch, heads, lk, d + 1}), v); },
                             "depth mismatch");
    test::require_throws<IA>([&] { attention(q, k, Tensor::zeros({batch, heads, lk + 1, dv})); },
                             "key count mismatch");
    test::require_throws<IA>([&] { attention(Tensor::zeros({d}), k, v); }, "1-d q");
    const Tensor float_mask = Tensor::ones({lq, lk});
    test::require_throws<IA>([&] { attention(q, k, v, &float_mask); }, "non-Bool mask");
    const Tensor wide_mask = Tensor::from_host(std::vector<std::uint8_t>(4 * lq * lk, 1).data(),
                                               {4, 1, lq, lk}, DType::Bool);
    test::require_throws<IA>([&] { attention(q, k, v, &wide_mask); },
                             "mask that does not broadcast to the scores");
}

} // namespace

int main() {
    return test::run("softmax_loss_test", [] {
        test_softmax_kernels();
        test_loss_kernels();
        test_softmax_api();
        test_loss_api();
        test_autograd();
        test_attention();
    });
}
