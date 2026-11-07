// Tests data movement: the concat and pad kernels against the naive references (bit for bit, since
// they only copy) at several thread counts and at sizes around the parallel cutoff, then concat,
// split and pad through the public API (axes, views, empty pieces, errors, data on a GPU).
#include "backends/cpu/ops/movement/movement.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/movement.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;
using reference::Dims;

namespace {

// Distinct values, so a misplaced element is always caught.
std::vector<float> iota(std::int64_t n, float start) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = start + static_cast<float>(i);
    }
    return v;
}

std::string text(const Dims& d) { return to_string(Shape(d.begin(), d.end())); }

// ---- kernels -------------------------------------------------------------------------------

struct ConcatCase {
    std::vector<Dims> shapes;
    int axis;
};

// Small, around the parallel cutoff (262144 elements), one input row longer than a 16384-float
// piece, many thin rows, and inputs that are empty along the axis.
const ConcatCase concat_cases[] = {
    {{{3}, {5}}, 0},
    {{{2, 3}, {2, 4}, {2, 1}}, 1},
    {{{2, 3}, {4, 3}}, 0},
    {{{2, 3, 4}, {2, 5, 4}}, 1},
    {{{2, 3, 4}, {2, 3, 1}, {2, 3, 7}}, 2},
    {{{2, 0, 4}, {2, 3, 4}, {2, 0, 4}}, 1},
    {{{16000}, {16767}}, 0},
    {{{16384}, {16385}}, 0},
    {{{50000}, {1}, {30000}}, 0},
    {{{3, 40000}, {3, 7}}, 1},
    {{{20000, 3}, {20000, 2}}, 1},
    {{{8, 64, 64}, {8, 32, 64}, {8, 1, 64}}, 1},
    {{{131072}, {131072}}, 0},
    {{{131072}, {131073}}, 0},
    {{{2048, 64}, {2048, 65}}, 1},
};

void test_concat_kernel() {
    const int original_threads = cpu::max_threads();
    for (const ConcatCase& c : concat_cases) {
        std::vector<std::vector<float>> inputs;
        for (const Dims& s : c.shapes) {
            inputs.push_back(
                iota(reference::count(s), 1000.0f * static_cast<float>(inputs.size())));
        }
        const auto want = reference::concat(inputs, c.shapes, c.axis);
        std::int64_t outer = 1;
        for (int d = 0; d < c.axis; ++d) {
            outer *= c.shapes[0][static_cast<std::size_t>(d)];
        }
        std::vector<const float*> pointers;
        std::vector<std::int64_t> widths;
        for (std::size_t t = 0; t < c.shapes.size(); ++t) {
            pointers.push_back(inputs[t].data());
            widths.push_back(outer == 0 ? 0 : reference::count(c.shapes[t]) / outer);
        }
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<float> out(want.size(), -1.0f);
            cpu::concat_f32(pointers.data(), widths.data(),
                            static_cast<std::int64_t>(pointers.size()), out.data(), outer);
            test::require(test::matches_f32(out, want), "concat kernel " + text(c.shapes[0]) +
                                                            "... axis " + std::to_string(c.axis) +
                                                            " threads=" + std::to_string(threads));
        }
    }
    cpu::set_max_threads(original_threads);
}

struct PadCase {
    Dims shape, before, after;
};

const PadCase pad_cases[] = {
    {{5}, {2}, {3}},
    {{5}, {0}, {0}},
    {{3, 4}, {1, 2}, {0, 1}},
    {{3, 4}, {0, 0}, {2, 0}},
    {{2, 3, 4}, {1, 0, 2}, {1, 1, 0}},
    {{0, 4}, {1, 1}, {1, 1}},
    {{3, 0}, {0, 2}, {1, 2}},
    {{100, 330}, {1, 0}, {1, 1}},
    {{181, 181}, {0, 0}, {1, 1}},
    {{2, 50000}, {0, 3}, {0, 4}},
    {{40000, 1}, {0, 1}, {0, 1}},
    {{4, 5, 6, 7}, {1, 2, 0, 1}, {2, 0, 1, 3}},
    {{510, 510}, {1, 1}, {1, 1}},
    {{511, 511}, {1, 1}, {1, 1}},
};

void test_pad_kernel() {
    const int original_threads = cpu::max_threads();
    for (const PadCase& c : pad_cases) {
        const auto x = iota(reference::count(c.shape), 1.0f);
        const auto want = reference::pad(x, c.shape, c.before, c.after, -7.5f);
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<float> out(want.size(), 99.0f);
            cpu::pad_f32(x.data(), out.data(), c.shape.data(), c.before.data(), c.after.data(),
                         static_cast<int>(c.shape.size()), -7.5f);
            test::require(test::matches_f32(out, want),
                          "pad kernel " + text(c.shape) + " threads=" + std::to_string(threads));
        }
    }
    cpu::set_max_threads(original_threads);
}

// ---- public API ----------------------------------------------------------------------------

void test_concat_api(Device device) {
    const auto a = iota(6, 0.0f);
    const auto b = iota(9, 100.0f);
    const Tensor ta = Tensor::from_vector(a, {2, 3}, device);
    const Tensor tb = Tensor::from_vector(b, {3, 3}, device);
    const Tensor rows = concat({ta, tb});
    test::require(rows.shape() == Shape{5, 3}, "concat axis 0 shape");
    test::require(rows.device() == device, "concat result on the wrong device");
    test::require(
        test::matches_f32(rows.to_vector<float>(), reference::concat({a, b}, {{2, 3}, {3, 3}}, 0)),
        "concat axis 0 on " + to_string(device));

    const Tensor tc = Tensor::from_vector(iota(4, 50.0f), {2, 2}, device);
    const Tensor cols = concat({ta, tc}, -1);
    test::require(cols.shape() == Shape{2, 5}, "concat axis -1 shape");
    test::require(test::matches_f32(cols.to_vector<float>(),
                                    reference::concat({a, iota(4, 50.0f)}, {{2, 3}, {2, 2}}, 1)),
                  "concat axis -1 on " + to_string(device));
}

void test_concat_views_and_edges() {
    // A transposed view and a sliced view go in without the caller packing them.
    const Tensor m = Tensor::from_vector(iota(12, 0.0f), {3, 4});
    const Tensor t = m.transpose(0, 1); // [4, 3]
    const Tensor s = Tensor::from_vector(iota(10, 50.0f), {10}).slice(0, 1, 10, 3).unsqueeze(0);
    const Tensor joined = concat({t, s}); // [4, 3] and [1, 3]
    test::require(joined.to_vector<float>() ==
                      concat({t.contiguous(), s.contiguous()}).to_vector<float>(),
                  "concat of views");

    // One input still makes a copy.
    const Tensor one = concat({m});
    test::require(!one.shares_storage(m) && one.to_vector<float>() == m.to_vector<float>(),
                  "concat of one tensor copies it");

    // Empty pieces along the axis, and an all-empty result.
    const Tensor empty = Tensor::empty({3, 0});
    test::require(concat({empty, m, empty}, 1).to_vector<float>() == m.to_vector<float>(),
                  "concat with empty pieces");
    test::require(concat({empty, empty}, 1).shape() == Shape{3, 0}, "concat of empty tensors");

    using E = std::invalid_argument;
    test::require_throws<E>([] { concat({}); }, "no tensors");
    test::require_throws<E>([&] { concat({m, Tensor()}); }, "undefined tensor");
    test::require_throws<E>([&] { concat({m, Tensor::ones({3, 4, 1})}); }, "ranks differ");
    test::require_throws<E>([&] { concat({m, Tensor::ones({2, 5})}); }, "sizes differ off axis");
    test::require_throws<E>([&] { concat({m, m}, 2); }, "axis out of range");
    test::require_throws<E>([&] { concat({m, m}, -3); }, "negative axis out of range");
    test::require_throws<E>([&] { concat({Tensor::ones({}), Tensor::ones({})}); }, "0-d");
    test::require_throws<E>([&] { concat({m, Tensor::ones({3, 4}, DType::I32)}); }, "dtypes");
    for (Device device : list_devices()) {
        if (device.type != DeviceType::CPU) {
            test::require_throws<E>([&] { concat({m, m.to(device)}); }, "mixed devices");
        }
    }
}

void test_split() {
    const Tensor x = Tensor::from_vector(iota(10 * 3, 0.0f), {10, 3});
    const auto pieces = split(x, 4); // 4, 4, 2 rows
    test::require(pieces.size() == 3, "split into chunks: count");
    test::require(pieces[0].shape() == Shape{4, 3} && pieces[2].shape() == Shape{2, 3},
                  "split into chunks: shapes");
    for (const Tensor& p : pieces) {
        test::require(p.shares_storage(x), "split pieces are views");
    }
    test::require(concat(pieces).to_vector<float>() == x.to_vector<float>(), "concat undoes split");

    const auto cols = split(x, std::vector<std::int64_t>{1, 0, 2}, -1);
    test::require(cols.size() == 3 && cols[0].shape() == Shape{10, 1} &&
                      cols[1].shape() == Shape{10, 0} && cols[2].shape() == Shape{10, 2},
                  "split by sizes: shapes");
    test::require(concat(cols, 1).to_vector<float>() == x.to_vector<float>(),
                  "concat undoes split by sizes");
    test::require(split(x, 10).size() == 1 && split(x, 99).size() == 1, "one piece");
    test::require(split(Tensor::empty({0, 3}), 2).size() == 1, "empty axis gives one piece");

    using E = std::invalid_argument;
    test::require_throws<E>([&] { split(x, 0); }, "size 0");
    test::require_throws<E>([&] { split(x, std::vector<std::int64_t>{4, 4}); }, "sizes short");
    test::require_throws<E>([&] { split(x, std::vector<std::int64_t>{11, -1}); }, "negative size");
    test::require_throws<E>([&] { split(x, 2, 2); }, "axis out of range");
    test::require_throws<E>([&] { split(Tensor::ones({}), 1); }, "0-d");
}

void test_pad_api(Device device) {
    const auto values = iota(6, 1.0f);
    const Tensor x = Tensor::from_vector(values, {2, 3}, device);

    // {1, 2} pads only the last dimension.
    const Tensor last = pad(x, {1, 2});
    test::require(last.shape() == Shape{2, 6} && last.device() == device, "pad last: shape");
    test::require(test::matches_f32(last.to_vector<float>(),
                                    reference::pad(values, {2, 3}, {0, 1}, {0, 2}, 0.0f)),
                  "pad last dimension on " + to_string(device));

    // {1, 0, 2, 1}: last dimension (1, 0), the one before it (2, 1); value 9.
    const Tensor both = pad(x, {1, 0, 2, 1}, 9.0f);
    test::require(both.shape() == Shape{5, 4}, "pad both: shape");
    test::require(test::matches_f32(both.to_vector<float>(),
                                    reference::pad(values, {2, 3}, {2, 1}, {1, 0}, 9.0f)),
                  "pad both dimensions on " + to_string(device));
}

void test_pad_edges() {
    const Tensor x = Tensor::from_vector(iota(6, 1.0f), {2, 3});
    const Tensor none = pad(x, {});
    test::require(!none.shares_storage(x) && none.to_vector<float>() == x.to_vector<float>(),
                  "empty padding copies");
    test::require(pad(x, {0, 0}).to_vector<float>() == x.to_vector<float>(), "zero padding");
    test::require(pad(x.transpose(0, 1), {1, 1}).to_vector<float>() ==
                      pad(x.transpose(0, 1).contiguous(), {1, 1}).to_vector<float>(),
                  "pad of a transposed view");
    const Tensor filled = pad(Tensor::empty({0, 2}), {0, 0, 1, 1}, 4.0f);
    test::require(filled.shape() == Shape{2, 2} &&
                      filled.to_vector<float>() == std::vector<float>(4, 4.0f),
                  "pad of an empty tensor is all fill");

    using E = std::invalid_argument;
    test::require_throws<E>([&] { pad(x, {1}); }, "odd padding");
    test::require_throws<E>([&] { pad(x, {1, 1, 1, 1, 1, 1}); }, "too many dimensions");
    test::require_throws<E>([&] { pad(x, {-1, 0}); }, "negative padding");
    test::require_throws<E>([&] { pad(Tensor(), {1, 1}); }, "undefined");
}

} // namespace

int main() {
    return test::run("movement_test", [] {
        test_concat_kernel();
        test_pad_kernel();
        for (Device device : list_devices()) {
            test_concat_api(device);
            test_pad_api(device);
        }
        test_concat_views_and_edges();
        test_split();
        test_pad_edges();
    });
}
