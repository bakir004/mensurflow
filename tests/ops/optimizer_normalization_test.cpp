// sgd, adam, adamw, layer_norm, rms_norm and batch_norm: known answers from PyTorch, double-precision
// references at several thread counts and sizes around the parallel cutoff, finite-difference checks
// of the reference gradients, bit-for-bit equality across thread counts, numerical edge cases, the
// public API (views, empty inputs, errors that leave the state untouched) and autograd.
#include "backends/cpu/threads.hpp"
#include "core/autograd.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/optimizer_normalization.hpp"
#include "reference/optimizer_normalization_torch.hpp"
#include "test_util.hpp"

#include <algorithm>
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
namespace ref = reference;

namespace {

constexpr float inf = std::numeric_limits<float>::infinity();

std::vector<float> values(std::int64_t n, int seed = 0, float spread = 2.0f) {
    std::vector<float> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        const int k = static_cast<int>((i * 37 + seed * 101) % 97) - 48;
        out[static_cast<std::size_t>(i)] = spread * static_cast<float>(k) / 48.0f;
    }
    return out;
}

// Strictly positive values (variance-like).
std::vector<float> positive(std::int64_t n, int seed = 0) {
    std::vector<float> out = values(n, seed, 1.0f);
    for (float& v : out) {
        v = 0.5f + std::abs(v);
    }
    return out;
}

Tensor make(const std::vector<float>& v, Shape shape) {
    return Tensor::from_vector(v, std::move(shape));
}

// Largest absolute value, for tolerances that scale with the data.
double scale_of(const ref::Vec& v) {
    double m = 0.0;
    for (double x : v) {
        m = std::max(m, std::abs(x));
    }
    return m;
}

// |actual - expected| <= rel * max|expected| + abs, with NaN matching NaN.
void require_close(const std::vector<float>& actual, const ref::Vec& expected,
                   const std::string& what, double rel = 2e-5, double abs = 1e-6) {
    test::require(actual.size() == expected.size(), what + ": size");
    const double tolerance = rel * scale_of(expected) + abs;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expected[i]) || std::isnan(actual[i])) {
            test::require(std::isnan(expected[i]) && std::isnan(actual[i]),
                          what + ": NaN mismatch at " + std::to_string(i));
        } else if (!(std::abs(actual[i] - expected[i]) <= tolerance)) {
            throw std::runtime_error(what + ": element " + std::to_string(i) + " is " +
                                     std::to_string(actual[i]) + ", expected " +
                                     std::to_string(expected[i]));
        }
    }
}

// Known answers are float results of PyTorch's own float kernels, so allow float rounding.
template <std::size_t N>
void require_torch(const std::vector<float>& actual, const float (&expected)[N],
                   const std::string& what, double rel = 2e-5, double abs = 2e-6) {
    require_close(actual, ref::Vec(expected, expected + N), what, rel, abs);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

const int all_threads = cpu::max_threads();
const std::vector<int> thread_counts = {1, 2, 3, all_threads};

// Restores the thread count when a test ends, normally or not.
struct ThreadGuard {
    ~ThreadGuard() { cpu::set_max_threads(all_threads); }
};

// ---- the reference formulas themselves -------------------------------------------------------

// The references are the oracle, so check their gradients against central differences of their
// own forward functions (in double, where the difference is accurate to about 1e-8).
void test_reference_gradients() {
    const double eps = 1e-5;
    const double h = 1e-6;
    auto loss = [](const ref::Vec& y, const ref::Vec& coef) {
        double total = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            total += y[i] * coef[i];
        }
        return total;
    };
    auto check = [&](const ref::Vec& analytic, const ref::Vec& at, auto&& forward,
                     const std::string& what) {
        for (std::size_t k = 0; k < at.size(); ++k) {
            ref::Vec up = at, down = at;
            up[k] += h;
            down[k] -= h;
            const double numeric = (forward(up) - forward(down)) / (2 * h);
            test::require(std::abs(numeric - analytic[k]) <= 1e-6 * (1.0 + std::abs(numeric)),
                          "reference gradient " + what + " element " + std::to_string(k));
        }
    };

    {
        const std::int64_t rows = 3, width = 5;
        const ref::Vec x = ref::to_double(values(rows * width, 1)),
                       coef = ref::to_double(values(rows * width, 2));
        const ref::Vec w = ref::to_double(positive(width, 3)), b = ref::to_double(values(width, 4));
        const auto g = ref::layer_norm_backward(x, coef, w, rows, width, eps);
        check(
            g.dx, x,
            [&](const ref::Vec& v) {
                return loss(ref::layer_norm(v, w, b, rows, width, eps), coef);
            },
            "layer dx");
        check(
            g.dw, w,
            [&](const ref::Vec& v) {
                return loss(ref::layer_norm(x, v, b, rows, width, eps), coef);
            },
            "layer dw");
        check(
            g.db, b,
            [&](const ref::Vec& v) {
                return loss(ref::layer_norm(x, w, v, rows, width, eps), coef);
            },
            "layer db");
        const auto r = ref::rms_norm_backward(x, coef, w, rows, width, eps);
        check(
            r.dx, x,
            [&](const ref::Vec& v) { return loss(ref::rms_norm(v, w, rows, width, eps), coef); },
            "rms dx");
        check(
            r.dw, w,
            [&](const ref::Vec& v) { return loss(ref::rms_norm(x, v, rows, width, eps), coef); },
            "rms dw");
    }
    for (const bool training : {true, false}) {
        const std::int64_t n = 3, c = 2, s = 2;
        const ref::Vec x = ref::to_double(values(n * c * s, 5)),
                       coef = ref::to_double(values(n * c * s, 6));
        const ref::Vec w = ref::to_double(positive(c, 7)), b = ref::to_double(values(c, 8));
        const ref::Vec rm = {0.3, -0.2}, rv = {1.2, 0.8};
        const auto g = ref::batch_norm_backward(x, coef, w, rm, rv, n, c, s, training, eps);
        const std::string tag = training ? "train " : "eval ";
        auto fwd = [&](const ref::Vec& xx, const ref::Vec& ww, const ref::Vec& bb) {
            return loss(ref::batch_norm(xx, ww, bb, rm, rv, n, c, s, training, 0.1, eps).y, coef);
        };
        check(g.dx, x, [&](const ref::Vec& v) { return fwd(v, w, b); }, "batch " + tag + "dx");
        check(g.dw, w, [&](const ref::Vec& v) { return fwd(x, v, b); }, "batch " + tag + "dw");
        check(g.db, b, [&](const ref::Vec& v) { return fwd(x, w, v); }, "batch " + tag + "db");
    }
}

// ---- optimizers ------------------------------------------------------------------------------

std::vector<float> torch_p0() {
    std::vector<float> p(6);
    for (int i = 0; i < 6; ++i)
        p[static_cast<std::size_t>(i)] = static_cast<float>(i - 2) / 4.0f;
    return p;
}

std::vector<float> torch_grad(int t) {
    std::vector<float> g(6);
    for (int i = 0; i < 6; ++i)
        g[static_cast<std::size_t>(i)] = static_cast<float>((i * 7 + t * 3) % 11 - 5) / 8.0f;
    return g;
}

void test_optimizers_against_torch() {
    {
        Tensor p = make(torch_p0(), {6}), velocity = Tensor::zeros({6});
        for (int t = 1; t <= 3; ++t)
            sgd(p, make(torch_grad(t), {6}), 0.1f, 0.9f, 0.01f, &velocity);
        require_torch(p.to_vector<float>(), torch_reference::opt_sgd_momentum,
                      "sgd momentum vs torch");
    }
    {
        Tensor p = make(torch_p0(), {6});
        for (int t = 1; t <= 3; ++t)
            sgd(p, make(torch_grad(t), {6}), 0.1f);
        require_torch(p.to_vector<float>(), torch_reference::opt_sgd_plain, "sgd vs torch");
    }
    {
        Tensor p = make(torch_p0(), {6}), m = Tensor::zeros({6}), v = Tensor::zeros({6});
        for (int t = 1; t <= 3; ++t)
            adam(p, make(torch_grad(t), {6}), m, v, 0.01f, t, 0.9f, 0.999f, 1e-8f, 0.1f);
        require_torch(p.to_vector<float>(), torch_reference::opt_adam, "adam vs torch");
        require_torch(m.to_vector<float>(), torch_reference::opt_adam_m,
                      "adam first moment vs torch");
        require_torch(v.to_vector<float>(), torch_reference::opt_adam_v,
                      "adam second moment vs torch", 2e-5, 1e-9);
    }
    {
        Tensor p = make(torch_p0(), {6}), m = Tensor::zeros({6}), v = Tensor::zeros({6});
        for (int t = 1; t <= 3; ++t)
            adamw(p, make(torch_grad(t), {6}), m, v, 0.01f, t, 0.9f, 0.999f, 1e-8f, 0.1f);
        require_torch(p.to_vector<float>(), torch_reference::opt_adamw, "adamw vs torch");
    }
}

struct OptimizerCase {
    const char* name;
    float lr, momentum, beta1, beta2, eps, decay;
};

void test_optimizer_kernels() {
    ThreadGuard guard;
    // Sizes around the 32768-element cutoff; five steps so the state matters.
    const std::int64_t sizes[] = {1, 7, 1000, 32767, 32768, 32769, 100003};
    const OptimizerCase cases[] = {{"plain", 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f, 0.0f},
                                   {"decay", 0.05f, 0.0f, 0.8f, 0.99f, 1e-6f, 0.1f},
                                   {"momentum", 0.02f, 0.9f, 0.9f, 0.999f, 1e-8f, 0.01f}};
    for (const std::int64_t n : sizes) {
        for (const auto& c : cases) {
            const std::string where = std::string(c.name) + " n=" + std::to_string(n);
            // Reference trajectories in double.
            ref::Vec p_sgd = ref::to_double(values(n, 1)), p_adam = p_sgd, p_adamw = p_sgd;
            ref::Vec vel(static_cast<std::size_t>(n), 0.0);
            ref::Vec m1(vel), v1(vel), m2(vel), v2(vel);
            std::vector<std::vector<float>> final_bits;
            for (const int threads : thread_counts) {
                cpu::set_max_threads(threads);
                Tensor ps = make(values(n, 1), {n}), pa = make(values(n, 1), {n}),
                       pw = make(values(n, 1), {n});
                Tensor velocity = Tensor::zeros({n});
                Tensor ma = Tensor::zeros({n}), va = Tensor::zeros({n}), mw = Tensor::zeros({n}),
                       vw = Tensor::zeros({n});
                ref::Vec rs = ref::to_double(values(n, 1)), ra = rs, rw = rs;
                ref::Vec rvel(static_cast<std::size_t>(n), 0.0), rm1(rvel), rv1(rvel), rm2(rvel),
                    rv2(rvel);
                for (int step = 1; step <= 5; ++step) {
                    const auto gv = values(n, 10 + step, 1.0f);
                    const Tensor g = make(gv, {n});
                    const ref::Vec gd = ref::to_double(gv);
                    sgd(ps, g, c.lr, c.momentum, c.decay, c.momentum > 0 ? &velocity : nullptr);
                    rs = ref::sgd_step(rs, gd, rvel, c.lr, c.momentum, c.decay);
                    adam(pa, g, ma, va, c.lr, step, c.beta1, c.beta2, c.eps, c.decay);
                    ra = ref::adam_step(ra, gd, rm1, rv1, c.lr, step, c.beta1, c.beta2, c.eps,
                                        c.decay, false);
                    adamw(pw, g, mw, vw, c.lr, step, c.beta1, c.beta2, c.eps, c.decay);
                    rw = ref::adam_step(rw, gd, rm2, rv2, c.lr, step, c.beta1, c.beta2, c.eps,
                                        c.decay, true);
                }
                require_close(ps.to_vector<float>(), rs, "sgd " + where, 1e-5, 1e-6);
                require_close(pa.to_vector<float>(), ra, "adam " + where, 1e-4, 1e-6);
                require_close(pw.to_vector<float>(), rw, "adamw " + where, 1e-4, 1e-6);
                if (c.momentum > 0)
                    require_close(velocity.to_vector<float>(), rvel, "sgd velocity " + where, 1e-5,
                                  1e-6);
                require_close(ma.to_vector<float>(), rm1, "adam m " + where, 1e-5, 1e-7);
                require_close(va.to_vector<float>(), rv1, "adam v " + where, 1e-5, 1e-9);
                // Elements are independent, so every thread count gives identical bits.
                std::vector<float> bits = ps.to_vector<float>();
                for (const Tensor* t : {&pa, &pw, &ma, &va, &mw, &vw}) {
                    const auto more = t->to_vector<float>();
                    bits.insert(bits.end(), more.begin(), more.end());
                }
                if (final_bits.empty()) {
                    final_bits.push_back(bits);
                } else {
                    test::require(same_bits(bits, final_bits[0]),
                                  "optimizers: same bits at every thread count, " + where);
                }
            }
        }
    }
}

void test_optimizer_api() {
    using IA = std::invalid_argument;
    const Tensor g = Tensor::ones({4});

    // Views: a strided gradient is packed; a view of a larger tensor is a valid parameter.
    {
        Tensor p = Tensor::zeros({2, 3});
        const Tensor big = make(values(6, 1), {3, 2});
        sgd(p, big.transpose(0, 1), 1.0f); // gradient is a transposed view
        const auto expected = big.transpose(0, 1).contiguous().to_vector<float>();
        for (std::size_t i = 0; i < 6; ++i)
            test::require(p.to_vector<float>()[i] == -expected[i], "strided gradient");
        Tensor table = Tensor::zeros({4, 3});
        Tensor row = table.slice(0, 1, 3); // rows 1..2: contiguous view with an offset
        sgd(row, Tensor::ones({2, 3}), 0.5f);
        const auto after = table.to_vector<float>();
        test::require(after[0] == 0.0f && after[3] == -0.5f && after[8] == -0.5f &&
                          after[9] == 0.0f,
                      "an offset view is updated in place and its neighbours are not");
    }

    // The update is visible through every handle to the same storage.
    {
        Tensor p = Tensor::ones({3});
        Tensor alias = p;
        sgd(p, Tensor::ones({3}), 0.25f);
        test::require(alias.to_vector<float>()[0] == 0.75f, "shared storage sees the update");
    }

    // Non-finite gradients propagate; zero learning rate and zero gradient leave the parameter.
    {
        Tensor p = Tensor::ones({2});
        sgd(p, make({NAN, 1.0f}, {2}), 0.1f);
        test::require(std::isnan(p.to_vector<float>()[0]) && p.to_vector<float>()[1] == 0.9f,
                      "NaN gradient");
        Tensor q = Tensor::ones({2}), m = Tensor::zeros({2}), v = Tensor::zeros({2});
        adam(q, Tensor::zeros({2}), m, v, 0.1f, 1);
        test::require(q.to_vector<float>()[0] == 1.0f, "zero gradient leaves Adam's parameter");
        adam(q, g.slice(0, 0, 2), m, v, 0.0f, 2);
        test::require(q.to_vector<float>()[0] == 1.0f, "zero learning rate");
    }

    // Large steps: bias correction tends to 1 and stays finite.
    {
        Tensor p = Tensor::ones({2}), m = Tensor::zeros({2}), v = Tensor::zeros({2});
        adam(p, Tensor::ones({2}), m, v, 0.1f, 1000000000LL);
        test::require(std::isfinite(p.to_vector<float>()[0]), "huge step count");
    }

    // Empty tensors are valid.
    {
        Tensor p = Tensor::zeros({0}), m = Tensor::zeros({0}), v = Tensor::zeros({0});
        sgd(p, Tensor::zeros({0}), 0.1f);
        adam(p, Tensor::zeros({0}), m, v, 0.1f, 1);
        adamw(p, Tensor::zeros({0}), m, v, 0.1f, 1);
    }

    // Errors throw before changing anything.
    {
        Tensor p = Tensor::ones({4}), m = Tensor::zeros({4}), v = Tensor::zeros({4});
        auto unchanged = [&](const std::string& what) {
            const auto values = p.to_vector<float>();
            test::require(values[0] == 1.0f && m.to_vector<float>()[0] == 0.0f &&
                              v.to_vector<float>()[0] == 0.0f,
                          what + ": state untouched after the error");
        };
        test::require_throws<IA>([&] { sgd(p, Tensor::ones({3}), 0.1f); }, "gradient shape");
        unchanged("gradient shape");
        test::require_throws<IA>([&] { sgd(p, Tensor::ones({4}, DType::I32), 0.1f); },
                                 "gradient dtype");
        test::require_throws<IA>([&] { sgd(p, g, -0.1f); }, "negative learning rate");
        test::require_throws<IA>([&] { sgd(p, g, NAN); }, "NaN learning rate");
        test::require_throws<IA>([&] { sgd(p, g, 0.1f, 1.0f, 0.0f, &m); }, "momentum of 1");
        test::require_throws<IA>([&] { sgd(p, g, 0.1f, -0.1f, 0.0f, &m); }, "negative momentum");
        test::require_throws<IA>([&] { sgd(p, g, 0.1f, 0.9f); }, "momentum without a buffer");
        test::require_throws<IA>([&] { sgd(p, g, 0.1f, 0.0f, -1.0f); }, "negative weight decay");
        Tensor wrong = Tensor::zeros({3});
        test::require_throws<IA>([&] { sgd(p, g, 0.1f, 0.9f, 0.0f, &wrong); }, "buffer shape");
        test::require_throws<IA>([&] { sgd(p, g, 0.1f, 0.9f, 0.0f, &p); },
                                 "buffer is the parameter");
        test::require_throws<IA>([&] { adam(p, g, m, v, 0.1f, 0); }, "step 0");
        test::require_throws<IA>([&] { adam(p, g, m, v, 0.1f, -3); }, "negative step");
        test::require_throws<IA>([&] { adam(p, g, m, v, 0.1f, 1, 1.0f); }, "beta1 of 1");
        test::require_throws<IA>([&] { adam(p, g, m, v, 0.1f, 1, 0.9f, -0.1f); }, "negative beta2");
        test::require_throws<IA>([&] { adam(p, g, m, v, 0.1f, 1, 0.9f, 0.999f, 0.0f); },
                                 "epsilon 0");
        test::require_throws<IA>([&] { adam(p, g, m, m, 0.1f, 1); }, "moments share storage");
        test::require_throws<IA>([&] { adam(p, g, p, v, 0.1f, 1); }, "moment is the parameter");
        Tensor wrong_v = Tensor::zeros({5});
        test::require_throws<IA>([&] { adamw(p, g, m, wrong_v, 0.1f, 1); }, "moment shape");
        Tensor transposed = Tensor::zeros({2, 4}).transpose(0, 1);
        test::require_throws<IA>([&] { sgd(transposed, Tensor::zeros({4, 2}), 0.1f); },
                                 "non-contiguous parameter");
        Tensor undefined;
        test::require_throws<IA>([&] { sgd(undefined, g, 0.1f); }, "undefined parameter");
        unchanged("all errors");
    }
}

// ---- layer and RMS norm ----------------------------------------------------------------------

void test_norm_against_torch() {
    const std::int64_t rows = 3, width = 5;
    std::vector<float> xv(15), dyv(15), wv(5), bv(5);
    for (int i = 0; i < 15; ++i) {
        xv[static_cast<std::size_t>(i)] = static_cast<float>((i * 37) % 97 - 48) / 16.0f;
        dyv[static_cast<std::size_t>(i)] = static_cast<float>(((i * 13) % 29) - 14) / 16.0f;
    }
    for (int j = 0; j < 5; ++j) {
        wv[static_cast<std::size_t>(j)] = static_cast<float>(1 + j % 5) / 4.0f;
        bv[static_cast<std::size_t>(j)] = static_cast<float>(j % 3 - 1) / 8.0f;
    }
    const Tensor x = make(xv, {rows, width}), dy = make(dyv, {rows, width}), w = make(wv, {width}),
                 b = make(bv, {width});
    require_torch(layer_norm(x, {width}, &w, &b).to_vector<float>(), torch_reference::layer_y,
                  "layer_norm vs torch");
    const auto lg = layer_norm_backward(x, dy, {width}, &w);
    require_torch(lg.input.to_vector<float>(), torch_reference::layer_dx, "layer_norm dx vs torch");
    require_torch(lg.weight.to_vector<float>(), torch_reference::layer_dw,
                  "layer_norm dw vs torch");
    require_torch(lg.bias.to_vector<float>(), torch_reference::layer_db, "layer_norm db vs torch");
    require_torch(rms_norm(x, {width}, &w).to_vector<float>(), torch_reference::rms_y,
                  "rms_norm vs torch");
    const auto rg = rms_norm_backward(x, dy, {width}, &w);
    require_torch(rg.input.to_vector<float>(), torch_reference::rms_dx, "rms_norm dx vs torch");
    require_torch(rg.weight.to_vector<float>(), torch_reference::rms_dw, "rms_norm dw vs torch");
    test::require(!rg.bias.defined(), "rms_norm has no bias gradient");
}

void test_norm_kernels() {
    ThreadGuard guard;
    // {rows, width}: width 1 (everything is the bias), tiny, around the cutoff (32768 elements),
    // fewer rows than blocks, more rows than blocks, and one long line.
    const std::int64_t shapes[][2] = {{1, 1},    {5, 1},     {3, 5},     {7, 33},    {64, 512},
                                      {65, 505}, {1024, 32}, {1025, 32}, {200, 300}, {3, 4000}};
    for (const auto& s : shapes) {
        const std::int64_t rows = s[0], width = s[1];
        const std::int64_t n = rows * width;
        const std::string where = std::to_string(rows) + "x" + std::to_string(width);
        const std::vector<float> xv = values(n, 1, 3.0f), gv = values(n, 2, 1.0f),
                                 wv = positive(width, 3), bv = values(width, 4, 1.0f);
        const Tensor x = make(xv, {rows, width}), g = make(gv, {rows, width}),
                     w = make(wv, {width}), b = make(bv, {width});
        const ref::Vec xd = ref::to_double(xv), gd = ref::to_double(gv), wd = ref::to_double(wv),
                       bd = ref::to_double(bv);
        const float eps = 1e-5f;
        const ref::Vec y_layer = ref::layer_norm(xd, wd, bd, rows, width, eps);
        const ref::Vec y_rms = ref::rms_norm(xd, wd, rows, width, eps);
        const auto g_layer = ref::layer_norm_backward(xd, gd, wd, rows, width, eps);
        const auto g_rms = ref::rms_norm_backward(xd, gd, wd, rows, width, eps);
        std::vector<std::vector<float>> first;
        for (const int threads : thread_counts) {
            cpu::set_max_threads(threads);
            std::vector<std::vector<float>> results;
            results.push_back(layer_norm(x, {width}, &w, &b, eps).to_vector<float>());
            const auto lb = layer_norm_backward(x, g, {width}, &w, eps);
            results.push_back(lb.input.to_vector<float>());
            results.push_back(lb.weight.to_vector<float>());
            results.push_back(lb.bias.to_vector<float>());
            results.push_back(rms_norm(x, {width}, &w, eps).to_vector<float>());
            const auto rb = rms_norm_backward(x, g, {width}, &w, eps);
            results.push_back(rb.input.to_vector<float>());
            results.push_back(rb.weight.to_vector<float>());
            require_close(results[0], y_layer, "layer_norm " + where);
            require_close(results[1], g_layer.dx, "layer_norm dx " + where, 5e-5, 1e-5);
            require_close(results[2], g_layer.dw, "layer_norm dw " + where, 5e-5, 1e-5);
            require_close(results[3], g_layer.db, "layer_norm db " + where, 5e-5, 1e-5);
            require_close(results[4], y_rms, "rms_norm " + where);
            require_close(results[5], g_rms.dx, "rms_norm dx " + where, 5e-5, 1e-5);
            require_close(results[6], g_rms.dw, "rms_norm dw " + where, 5e-5, 1e-5);
            if (first.empty()) {
                first = results;
            } else {
                for (std::size_t i = 0; i < results.size(); ++i) {
                    test::require(same_bits(results[i], first[i]),
                                  "norm same bits at every thread count " + where + " result " +
                                      std::to_string(i));
                }
            }
        }
    }
}

void test_norm_api_and_numerics() {
    using IA = std::invalid_argument;
    ThreadGuard guard;

    // Default weight and bias: scale 1, bias 0.
    {
        const Tensor x = make(values(12, 1), {3, 4});
        const Tensor ones = Tensor::ones({4}), zeros = Tensor::zeros({4});
        test::require(same_bits(layer_norm(x, {4}).to_vector<float>(),
                                layer_norm(x, {4}, &ones, &zeros).to_vector<float>()),
                      "layer_norm defaults");
        test::require(same_bits(rms_norm(x, {4}).to_vector<float>(),
                                rms_norm(x, {4}, &ones).to_vector<float>()),
                      "rms_norm defaults");
    }

    // Normalizing over more than one trailing dimension: [4, 2, 3] over {2, 3} is 4 lines of 6.
    {
        const auto xv = values(24, 5, 2.0f);
        const Tensor x = make(xv, {4, 2, 3}), flat = make(xv, {4, 6});
        test::require(same_bits(layer_norm(x, {2, 3}).to_vector<float>(),
                                layer_norm(flat, {6}).to_vector<float>()),
                      "layer_norm over two trailing dimensions");
        const Tensor w = make(positive(6, 6), {2, 3});
        const auto yd = ref::layer_norm(ref::to_double(xv), ref::to_double(positive(6, 6)),
                                        ref::Vec(6, 0.0), 4, 6, 1e-5);
        require_close(layer_norm(x, {2, 3}, &w).to_vector<float>(), yd,
                      "layer_norm with a 2-d weight");
        const auto lb = layer_norm_backward(x, make(values(24, 7, 1.0f), {4, 2, 3}), {2, 3}, &w);
        test::require(lb.weight.shape() == Shape{2, 3} && lb.input.shape() == x.shape(),
                      "backward shapes, 2-d");
    }

    // Strided and transposed inputs are packed first and give the same result as packed ones.
    {
        const Tensor base = make(values(60, 8), {5, 12});
        const Tensor view = base.transpose(0, 1); // [12, 5], strided
        const Tensor packed = view.contiguous();
        test::require(same_bits(layer_norm(view, {5}).to_vector<float>(),
                                layer_norm(packed, {5}).to_vector<float>()),
                      "layer_norm on a transposed view");
        test::require(same_bits(rms_norm(view, {5}).to_vector<float>(),
                                rms_norm(packed, {5}).to_vector<float>()),
                      "rms_norm on a transposed view");
        const Tensor slice = base.slice(1, 2, 12, 2); // every other column, [5, 5]
        test::require(same_bits(layer_norm(slice, {5}).to_vector<float>(),
                                layer_norm(slice.contiguous(), {5}).to_vector<float>()),
                      "layer_norm on a stepped slice");
    }

    // A large common offset: the two-pass variance keeps full accuracy where E[x^2] - E[x]^2 loses it.
    {
        const std::int64_t rows = 4, width = 1000;
        std::vector<float> xv = values(rows * width, 3, 1.0f);
        for (float& v : xv)
            v += 10000.0f;
        const auto yd = ref::layer_norm(ref::to_double(xv), ref::Vec(width, 1.0),
                                        ref::Vec(width, 0.0), rows, width, 1e-5);
        require_close(layer_norm(make(xv, {rows, width}), {width}).to_vector<float>(), yd,
                      "layer_norm with offset 1e4", 2e-3, 1e-3);
    }

    // Constant lines: zero variance, so the output is the bias and epsilon keeps it finite.
    {
        const Tensor x = Tensor::full({2, 8}, 3.5);
        const Tensor b = make(values(8, 2, 1.0f), {8});
        const auto y = layer_norm(x, {8}, nullptr, &b).to_vector<float>();
        const auto bv = b.to_vector<float>();
        for (std::size_t i = 0; i < 16; ++i)
            test::require(y[i] == bv[i % 8], "constant line gives the bias");
        const auto r = rms_norm(Tensor::zeros({2, 8}), {8}).to_vector<float>();
        for (float v : r)
            test::require(v == 0.0f, "rms_norm of zeros is zeros");
        const auto grad = layer_norm_backward(x, Tensor::ones({2, 8}), {8});
        for (float v : grad.input.to_vector<float>())
            test::require(std::isfinite(v), "finite gradient for a constant line");
    }

    // A NaN or infinity poisons its own line and nothing else.
    {
        std::vector<float> xv = values(15, 4, 1.0f);
        xv[7] = NAN;  // line 1
        xv[10] = inf; // line 2
        const auto y = layer_norm(make(xv, {3, 5}), {5}).to_vector<float>();
        for (int j = 0; j < 5; ++j) {
            test::require(std::isfinite(y[static_cast<std::size_t>(j)]), "line 0 is clean");
            test::require(std::isnan(y[static_cast<std::size_t>(5 + j)]), "NaN line");
            test::require(std::isnan(y[static_cast<std::size_t>(10 + j)]), "infinite line");
        }
    }

    // Empty inputs: no rows is fine and gives zero parameter gradients.
    {
        const Tensor x = Tensor::zeros({0, 4});
        test::require(layer_norm(x, {4}).numel() == 0 && rms_norm(x, {4}).numel() == 0,
                      "empty forward");
        const auto lb = layer_norm_backward(x, Tensor::zeros({0, 4}), {4});
        test::require(lb.input.numel() == 0 &&
                          lb.weight.to_vector<float>() == std::vector<float>(4, 0.0f),
                      "empty backward");
    }

    // Errors.
    {
        const Tensor x = make(values(12, 1), {3, 4});
        test::require_throws<IA>([&] { layer_norm(x, {5}); }, "normalized shape mismatch");
        test::require_throws<IA>([&] { layer_norm(x, {3}); },
                                 "normalizing over a non-trailing size");
        test::require_throws<IA>([&] { layer_norm(x, {}); }, "empty normalized shape");
        test::require_throws<IA>([&] { layer_norm(x, {2, 3, 4}); },
                                 "too many normalized dimensions");
        test::require_throws<IA>([&] { layer_norm(Tensor::zeros({3, 0}), {0}); },
                                 "empty normalized dimension");
        test::require_throws<IA>([&] { layer_norm(Tensor::ones({3, 4}, DType::I32), {4}); },
                                 "integer input");
        const Tensor bad = Tensor::ones({3});
        test::require_throws<IA>([&] { layer_norm(x, {4}, &bad); }, "weight shape");
        test::require_throws<IA>([&] { layer_norm(x, {4}, nullptr, &bad); }, "bias shape");
        test::require_throws<IA>([&] { rms_norm(x, {4}, &bad); }, "rms weight shape");
        test::require_throws<IA>([&] { layer_norm(x, {4}, nullptr, nullptr, 0.0f); }, "epsilon 0");
        test::require_throws<IA>([&] { rms_norm(x, {4}, nullptr, NAN); }, "epsilon NaN");
        test::require_throws<IA>([&] { layer_norm_backward(x, Tensor::ones({3, 5}), {4}); },
                                 "gradient shape");
        test::require_throws<IA>(
            [&] { rms_norm_backward(x, Tensor::ones({3, 4}, DType::I32), {4}); }, "gradient dtype");
        const Tensor undefined;
        test::require_throws<IA>([&] { layer_norm(undefined, {4}); }, "undefined input");
    }
}

// ---- batch norm ------------------------------------------------------------------------------

void test_batch_norm_against_torch() {
    auto xs = [](std::int64_t n) {
        std::vector<float> v(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] = static_cast<float>((i * 37) % 97 - 48) / 16.0f;
        return v;
    };
    auto dys = [](std::int64_t n) {
        std::vector<float> v(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] = static_cast<float>(((i * 13) % 29) - 14) / 16.0f;
        return v;
    };
    const Tensor w = make({0.25f, 0.5f, 0.75f}, {3}), b = make({-0.125f, 0.0f, 0.125f}, {3});
    const std::vector<float> rm0 = {0.25f, -0.5f, 1.0f}, rv0 = {1.5f, 0.75f, 2.0f};
    {
        const Tensor x = make(xs(24), {4, 3, 2}), dy = make(dys(24), {4, 3, 2});
        Tensor rm = make(rm0, {3}), rv = make(rv0, {3});
        const Tensor y = batch_norm(x, rm, rv, &w, &b, true, 0.1f, 1e-5f);
        require_torch(y.to_vector<float>(), torch_reference::bn_train_y,
                      "batch_norm train vs torch");
        require_torch(rm.to_vector<float>(), torch_reference::bn_train_running_mean,
                      "running mean vs torch");
        require_torch(rv.to_vector<float>(), torch_reference::bn_train_running_var,
                      "running variance vs torch");
        const auto g = batch_norm_backward(x, dy, rm, rv, &w, true, 1e-5f);
        require_torch(g.input.to_vector<float>(), torch_reference::bn_train_dx,
                      "batch_norm train dx vs torch", 2e-5, 5e-6);
        require_torch(g.weight.to_vector<float>(), torch_reference::bn_train_dw,
                      "batch_norm train dw vs torch");
        require_torch(g.bias.to_vector<float>(), torch_reference::bn_train_db,
                      "batch_norm train db vs torch");

        Tensor erm = make(rm0, {3}), erv = make(rv0, {3});
        const Tensor ey = batch_norm(x, erm, erv, &w, &b, false, 0.1f, 1e-5f);
        require_torch(ey.to_vector<float>(), torch_reference::bn_eval_y,
                      "batch_norm eval vs torch");
        test::require(erm.to_vector<float>() == rm0 && erv.to_vector<float>() == rv0,
                      "eval leaves the running statistics");
        const auto eg = batch_norm_backward(x, dy, erm, erv, &w, false, 1e-5f);
        require_torch(eg.input.to_vector<float>(), torch_reference::bn_eval_dx,
                      "batch_norm eval dx vs torch");
        require_torch(eg.weight.to_vector<float>(), torch_reference::bn_eval_dw,
                      "batch_norm eval dw vs torch");
        require_torch(eg.bias.to_vector<float>(), torch_reference::bn_eval_db,
                      "batch_norm eval db vs torch");
    }
    {
        const Tensor x = make(xs(15), {5, 3}), dy = make(dys(15), {5, 3});
        Tensor rm = make(rm0, {3}), rv = make(rv0, {3});
        const Tensor y = batch_norm(x, rm, rv, &w, &b, true, 0.2f, 1e-5f);
        require_torch(y.to_vector<float>(), torch_reference::bn2_y, "batch_norm [N,C] vs torch");
        require_torch(rm.to_vector<float>(), torch_reference::bn2_running_mean,
                      "[N,C] running mean vs torch");
        require_torch(rv.to_vector<float>(), torch_reference::bn2_running_var,
                      "[N,C] running variance vs torch");
        const auto g = batch_norm_backward(x, dy, rm, rv, &w, true, 1e-5f);
        require_torch(g.input.to_vector<float>(), torch_reference::bn2_dx, "[N,C] dx vs torch",
                      2e-5, 5e-6);
    }
}

void test_batch_norm_kernels() {
    ThreadGuard guard;
    // {n, c, s}: [N, C] layouts (s == 1), images, fewer samples than blocks, more than blocks, a
    // few channels with a lot of data, and sizes around the cutoff.
    const std::int64_t shapes[][3] = {{2, 3, 1},    {33, 7, 1},  {200, 16, 1}, {5, 4, 6},
                                      {8, 4, 16},   {1, 3, 50},  {2, 1, 300},  {70, 3, 25},
                                      {16, 5, 410}, {64, 8, 64}, {3000, 11, 1}};
    for (const auto& s : shapes) {
        const std::int64_t n = s[0], c = s[1], sp = s[2];
        const std::int64_t total = n * c * sp;
        const std::string where =
            std::to_string(n) + "x" + std::to_string(c) + "x" + std::to_string(sp);
        if (n * sp < 2)
            continue;
        const auto xv = values(total, 1, 3.0f), gv = values(total, 2, 1.0f);
        const auto wv = positive(c, 3), bv = values(c, 4, 1.0f), rmv = values(c, 5, 0.5f),
                   rvv = positive(c, 6);
        const Tensor x = make(xv, {n, c, sp}), g = make(gv, {n, c, sp}), w = make(wv, {c}),
                     b = make(bv, {c});
        // Shapes with s == 1 are given as [n, c] so the kernel's row-block path runs.
        const Shape x_shape = sp == 1 ? Shape{n, c} : Shape{n, c, sp};
        const Tensor xs = make(xv, x_shape), gs = make(gv, x_shape);
        const float eps = 1e-5f, momentum = 0.1f;
        for (const bool training : {true, false}) {
            const auto fwd = ref::batch_norm(
                ref::to_double(xv), ref::to_double(wv), ref::to_double(bv), ref::to_double(rmv),
                ref::to_double(rvv), n, c, sp, training, momentum, eps);
            const auto bwd = ref::batch_norm_backward(ref::to_double(xv), ref::to_double(gv),
                                                      ref::to_double(wv), ref::to_double(rmv),
                                                      ref::to_double(rvv), n, c, sp, training, eps);
            std::vector<std::vector<float>> first;
            for (const int threads : thread_counts) {
                cpu::set_max_threads(threads);
                Tensor rm = make(rmv, {c}), rv = make(rvv, {c});
                const Tensor y = batch_norm(xs, rm, rv, &w, &b, training, momentum, eps);
                const auto bg = batch_norm_backward(xs, gs, rm, rv, &w, training, eps);
                // Backward in training does not use the (already updated) running statistics.
                std::vector<std::vector<float>> results = {
                    y.to_vector<float>(),         rm.to_vector<float>(),
                    rv.to_vector<float>(),        bg.input.to_vector<float>(),
                    bg.weight.to_vector<float>(), bg.bias.to_vector<float>()};
                const std::string tag = std::string(training ? "train " : "eval ") + where;
                require_close(results[0], fwd.y, "batch_norm " + tag);
                require_close(results[1], fwd.running_mean, "running mean " + tag);
                require_close(results[2], fwd.running_var, "running variance " + tag);
                if (!training) {
                    // evaluation backward reads the (unchanged) running statistics
                    require_close(results[3], bwd.dx, "batch_norm dx " + tag, 5e-5, 1e-5);
                } else {
                    // recompute the reference with the batch statistics only
                    require_close(results[3], bwd.dx, "batch_norm dx " + tag, 5e-5, 1e-5);
                }
                require_close(results[4], bwd.dw, "batch_norm dw " + tag, 5e-5, 1e-5);
                require_close(results[5], bwd.db, "batch_norm db " + tag, 5e-5, 1e-5);
                if (first.empty()) {
                    first = results;
                } else {
                    for (std::size_t i = 0; i < results.size(); ++i) {
                        test::require(same_bits(results[i], first[i]),
                                      "batch_norm same bits at every thread count " + tag +
                                          " result " + std::to_string(i));
                    }
                }
            }
        }
    }
}

void test_batch_norm_api() {
    using IA = std::invalid_argument;
    const Tensor x = make(values(24, 1), {4, 3, 2});

    // Running statistics: momentum 0 keeps them, momentum 1 replaces them with the batch's.
    {
        Tensor rm = Tensor::zeros({3}), rv = Tensor::ones({3});
        batch_norm(x, rm, rv, nullptr, nullptr, true, 0.0f);
        test::require(rm.to_vector<float>() == std::vector<float>(3, 0.0f) &&
                          rv.to_vector<float>() == std::vector<float>(3, 1.0f),
                      "momentum 0 keeps the running statistics");
        batch_norm(x, rm, rv, nullptr, nullptr, true, 1.0f);
        const auto ref_stats = ref::batch_norm(ref::to_double(x.to_vector<float>()),
                                               ref::Vec(3, 1.0), ref::Vec(3, 0.0), ref::Vec(3, 0.0),
                                               ref::Vec(3, 1.0), 4, 3, 2, true, 1.0, 1e-5);
        require_close(rm.to_vector<float>(), ref_stats.running_mean, "momentum 1 mean");
        require_close(rv.to_vector<float>(), ref_stats.running_var, "momentum 1 unbiased variance");
    }

    // Repeated training steps converge the running statistics to the data's (momentum 0.1 -> 1 - 0.9^k).
    {
        Tensor rm = Tensor::zeros({3}), rv = Tensor::ones({3});
        for (int i = 0; i < 200; ++i)
            batch_norm(x, rm, rv, nullptr, nullptr, true, 0.1f);
        const auto expected = ref::batch_norm(ref::to_double(x.to_vector<float>()),
                                              ref::Vec(3, 1.0), ref::Vec(3, 0.0), ref::Vec(3, 0.0),
                                              ref::Vec(3, 1.0), 4, 3, 2, true, 1.0, 1e-5);
        require_close(rm.to_vector<float>(), expected.running_mean, "running mean converges", 1e-4,
                      1e-5);
        require_close(rv.to_vector<float>(), expected.running_var, "running variance converges",
                      1e-4, 1e-5);
    }

    // Eval mode accepts a single sample; a strided input is packed; defaults are scale 1, bias 0.
    {
        Tensor rm = Tensor::zeros({3}), rv = Tensor::ones({3});
        const Tensor one = make(values(3, 2), {1, 3});
        const auto y = batch_norm(one, rm, rv, nullptr, nullptr, false).to_vector<float>();
        const auto in = one.to_vector<float>();
        for (int i = 0; i < 3; ++i)
            test::require(std::abs(y[static_cast<std::size_t>(i)] -
                                   in[static_cast<std::size_t>(i)] / std::sqrt(1.0f + 1e-5f)) <
                              1e-6f,
                          "identity statistics");
        const Tensor strided = make(values(24, 3), {4, 3, 2}).transpose(0, 2); // [2, 3, 4] strided
        Tensor rm1 = Tensor::zeros({3}), rv1 = Tensor::ones({3}), rm2 = Tensor::zeros({3}),
               rv2 = Tensor::ones({3});
        test::require(same_bits(batch_norm(strided, rm1, rv1).to_vector<float>(),
                                batch_norm(strided.contiguous(), rm2, rv2).to_vector<float>()),
                      "batch_norm on a strided input");
        test::require(same_bits(rm1.to_vector<float>(), rm2.to_vector<float>()) &&
                          same_bits(rv1.to_vector<float>(), rv2.to_vector<float>()),
                      "same running statistics from a strided input");
    }

    // Empty batch: fine in evaluation, an error in training.
    {
        Tensor rm = Tensor::zeros({3}), rv = Tensor::ones({3});
        test::require(batch_norm(Tensor::zeros({0, 3}), rm, rv, nullptr, nullptr, false).numel() ==
                          0,
                      "empty eval");
        test::require_throws<IA>([&] { batch_norm(Tensor::zeros({0, 3}), rm, rv); },
                                 "empty training batch");
    }

    // Errors leave the running statistics untouched.
    {
        Tensor rm = Tensor::zeros({3}), rv = Tensor::ones({3});
        auto untouched = [&](const std::string& what) {
            test::require(rm.to_vector<float>() == std::vector<float>(3, 0.0f) &&
                              rv.to_vector<float>() == std::vector<float>(3, 1.0f),
                          what + ": running statistics untouched");
        };
        test::require_throws<IA>([&] { batch_norm(Tensor::zeros({4}), rm, rv); }, "1-d input");
        test::require_throws<IA>([&] { batch_norm(make(values(8, 1), {4, 2}), rm, rv); },
                                 "channel count mismatch");
        test::require_throws<IA>([&] { batch_norm(make(values(3, 1), {1, 3}), rm, rv); },
                                 "training with one value per channel");
        test::require_throws<IA>([&] { batch_norm(Tensor::ones({4, 3}, DType::I32), rm, rv); },
                                 "integer input");
        Tensor wrong = Tensor::zeros({4}), wrong_v = Tensor::ones({4});
        test::require_throws<IA>([&] { batch_norm(x, wrong, rv); }, "running mean shape");
        test::require_throws<IA>([&] { batch_norm(x, rm, wrong_v); }, "running variance shape");
        const Tensor bad = Tensor::ones({4});
        test::require_throws<IA>([&] { batch_norm(x, rm, rv, &bad); }, "weight shape");
        test::require_throws<IA>([&] { batch_norm(x, rm, rv, nullptr, &bad); }, "bias shape");
        test::require_throws<IA>([&] { batch_norm(x, rm, rv, nullptr, nullptr, true, -0.1f); },
                                 "negative momentum");
        test::require_throws<IA>([&] { batch_norm(x, rm, rv, nullptr, nullptr, true, 1.5f); },
                                 "momentum above 1");
        test::require_throws<IA>([&] { batch_norm(x, rm, rv, nullptr, nullptr, true, 0.1f, 0.0f); },
                                 "epsilon 0");
        Tensor strided_mean = Tensor::zeros({6}).slice(0, 0, 6, 2);
        test::require_throws<IA>([&] { batch_norm(x, strided_mean, rv); },
                                 "non-contiguous running mean");
        test::require_throws<IA>([&] { batch_norm_backward(x, Tensor::ones({4, 3, 3}), rm, rv); },
                                 "backward gradient shape");
        test::require_throws<IA>(
            [&] { batch_norm_backward(make(values(3, 1), {1, 3}), Tensor::ones({1, 3}), rm, rv); },
            "training backward with one value per channel");
        untouched("all errors");
    }
}

// ---- autograd --------------------------------------------------------------------------------

// loss = sum(y * coef); y's gradient is then exactly coef.
Tensor weighted_sum(const Tensor& y, const Tensor& coef) { return sum(mul(y, coef)); }

void test_autograd() {
    const std::int64_t rows = 4, width = 6;
    const auto xv = values(rows * width, 1, 2.0f), cv = values(rows * width, 2, 1.0f);
    const auto wv = positive(width, 3), bv = values(width, 4, 1.0f);
    const Tensor coef = make(cv, {rows, width});
    const ref::Vec xd = ref::to_double(xv), cd = ref::to_double(cv), wd = ref::to_double(wv);

    {
        Tensor x = make(xv, {rows, width}), w = make(wv, {width}), b = make(bv, {width});
        x.set_requires_grad();
        w.set_requires_grad();
        b.set_requires_grad();
        weighted_sum(layer_norm(x, {width}, &w, &b), coef).backward();
        const auto expected = ref::layer_norm_backward(xd, cd, wd, rows, width, 1e-5);
        require_close(x.grad().to_vector<float>(), expected.dx, "layer_norm autograd dx", 5e-5,
                      1e-5);
        require_close(w.grad().to_vector<float>(), expected.dw, "layer_norm autograd dw", 5e-5,
                      1e-5);
        require_close(b.grad().to_vector<float>(), expected.db, "layer_norm autograd db", 5e-5,
                      1e-5);
    }
    {
        // Only the weight needs a gradient: x gets none.
        Tensor x = make(xv, {rows, width}), w = make(wv, {width});
        w.set_requires_grad();
        weighted_sum(layer_norm(x, {width}, &w), coef).backward();
        test::require(w.has_grad() && !x.has_grad(), "layer_norm: only the weight gets a gradient");
        // No tensor requires a gradient: nothing is recorded.
        const Tensor plain = layer_norm(make(xv, {rows, width}), {width});
        test::require(!plain.requires_grad(), "no graph without requires_grad");
        // NoGrad: nothing is recorded even for a tracked input.
        Tensor tracked = make(xv, {rows, width});
        tracked.set_requires_grad();
        {
            detail::autograd::NoGradGuard guard;
            test::require(!layer_norm(tracked, {width}).requires_grad(),
                          "no graph inside NoGradGuard");
        }
    }
    {
        Tensor x = make(xv, {rows, width}), w = make(wv, {width});
        x.set_requires_grad();
        w.set_requires_grad();
        weighted_sum(rms_norm(x, {width}, &w), coef).backward();
        const auto expected = ref::rms_norm_backward(xd, cd, wd, rows, width, 1e-5);
        require_close(x.grad().to_vector<float>(), expected.dx, "rms_norm autograd dx", 5e-5, 1e-5);
        require_close(w.grad().to_vector<float>(), expected.dw, "rms_norm autograd dw", 5e-5, 1e-5);
    }
    // A norm inside a small network: gradients flow through to an earlier parameter.
    {
        Tensor x = make(xv, {rows, width});
        x.set_requires_grad();
        const Tensor y = layer_norm(mul(x, make(wv, {width})), {width});
        weighted_sum(y, coef).backward();
        test::require(x.has_grad() && x.grad().shape() == x.shape(),
                      "gradient through layer_norm and mul");
    }

    // batch_norm, training and evaluation, on [N, C, S].
    for (const bool training : {true, false}) {
        const std::int64_t n = 5, c = 3, s = 4;
        const auto bx = values(n * c * s, 5, 2.0f), bc = values(n * c * s, 6, 1.0f);
        const auto bw = positive(c, 7), bb = values(c, 8, 1.0f), rmv = values(c, 9, 0.5f),
                   rvv = positive(c, 10);
        Tensor x = make(bx, {n, c, s}), w = make(bw, {c}), b = make(bb, {c});
        x.set_requires_grad();
        w.set_requires_grad();
        b.set_requires_grad();
        Tensor rm = make(rmv, {c}), rv = make(rvv, {c});
        const Tensor y = batch_norm(x, rm, rv, &w, &b, training, 0.1f);
        // The caller may keep updating the statistics before backward; the graph must not see it.
        rm.copy_from(Tensor::full({c}, 100.0));
        rv.copy_from(Tensor::full({c}, 100.0));
        weighted_sum(y, make(bc, {n, c, s})).backward();
        const auto expected = ref::batch_norm_backward(
            ref::to_double(bx), ref::to_double(bc), ref::to_double(bw), ref::to_double(rmv),
            ref::to_double(rvv), n, c, s, training, 1e-5);
        const std::string tag = training ? "train" : "eval";
        require_close(x.grad().to_vector<float>(), expected.dx, "batch_norm autograd dx " + tag,
                      5e-5, 1e-5);
        require_close(w.grad().to_vector<float>(), expected.dw, "batch_norm autograd dw " + tag,
                      5e-5, 1e-5);
        require_close(b.grad().to_vector<float>(), expected.db, "batch_norm autograd db " + tag,
                      5e-5, 1e-5);
    }

    // A finite-difference check of the public op, through float arithmetic (looser).
    {
        const std::int64_t n = 4, w_ = 5;
        const auto fx = values(n * w_, 11, 2.0f), fc = values(n * w_, 12, 1.0f);
        Tensor x = make(fx, {n, w_});
        x.set_requires_grad();
        weighted_sum(layer_norm(x, {w_}), make(fc, {n, w_})).backward();
        const auto analytic = x.grad().to_vector<float>();
        for (std::size_t k = 0; k < fx.size(); ++k) {
            auto up = fx, down = fx;
            const float h = 1e-2f;
            up[k] += h;
            down[k] -= h;
            const float lu =
                weighted_sum(layer_norm(make(up, {n, w_}), {w_}), make(fc, {n, w_})).item<float>();
            const float ld = weighted_sum(layer_norm(make(down, {n, w_}), {w_}), make(fc, {n, w_}))
                                 .item<float>();
            test::require(std::abs((lu - ld) / (2 * h) - analytic[k]) < 2e-3f,
                          "finite difference of layer_norm");
        }
    }
}

} // namespace

int main() {
    return test::run("optimizer_normalization_test", [] {
        test_reference_gradients();
        test_optimizers_against_torch();
        test_optimizer_kernels();
        test_optimizer_api();
        test_norm_against_torch();
        test_norm_kernels();
        test_norm_api_and_numerics();
        test_batch_norm_against_torch();
        test_batch_norm_kernels();
        test_batch_norm_api();
        test_autograd();
    });
}
