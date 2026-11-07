// Tests batched GEMM: every CPU strategy (sequential, over_batch, within_matrix, parallel) against
// a float64 reference at several thread counts and sizes, then mensurflow::gemm_strided_batched
// through the public API (batch broadcasting, transposes, views, alpha and beta, empty shapes,
// errors, and data on a GPU).
#include "backends/cpu/ops/gemm/gemm_batched.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/blas.hpp"
#include "reference/movement.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;
using reference::Dims;

namespace {

// Multiples of 0.25 in [-4, 4], so products and short sums are exact in float.
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

void require_close(const std::vector<float>& got, const std::vector<double>& want,
                   const std::vector<double>& magnitude, const std::string& what) {
    test::require(got.size() == want.size(), what + ": " + std::to_string(got.size()) +
                                                 " values, expected " +
                                                 std::to_string(want.size()));
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(std::abs(static_cast<double>(got[i]) - want[i]) <=
                          reference::matmul_tolerance(magnitude[i]),
                      what + ": entry " + std::to_string(i) + " got " + std::to_string(got[i]) +
                          ", want " + std::to_string(want[i]));
    }
}

// ---- kernels -------------------------------------------------------------------------------

using Kernel = void (*)(const cpu::BatchedGemm&);

struct Size {
    std::int64_t batch, m, n, k;
};

// One matrix, fewer matrices than threads, many small ones, sizes that are not multiples of any
// tile, and matrices big enough for within_matrix to really split them.
const Size sizes[] = {{1, 1, 1, 1},  {1, 7, 5, 3},    {3, 17, 31, 13},  {16, 4, 4, 4},
                      {64, 9, 7, 5}, {2, 96, 80, 70}, {5, 65, 129, 33}, {33, 20, 1, 20}};

void test_kernels() {
    const int original_threads = cpu::max_threads();
    const Kernel kernels[] = {cpu::gemm_batched_f32_sequential, cpu::gemm_batched_f32_over_batch,
                              cpu::gemm_batched_f32_within_matrix, cpu::gemm_batched_f32_parallel};
    const char* names[] = {"sequential", "over_batch", "within_matrix", "parallel"};

    for (const Size& s : sizes) {
        for (const bool shared_b : {false, true}) { // shared_b: every matrix uses the same B
            for (const bool with_c : {false, true}) {
                const auto a = make_input(s.batch * s.m * s.k, 1);
                const auto b = make_input((shared_b ? 1 : s.batch) * s.k * s.n, 2);
                const auto c0 = make_input(s.batch * s.m * s.n, 3);
                const float alpha = with_c ? 0.5f : 1.0f;
                const float beta = with_c ? -2.0f : 0.0f;

                std::vector<double> want, magnitude;
                for (std::int64_t i = 0; i < s.batch; ++i) {
                    const auto r =
                        reference::matmul(a.data() + i * s.m * s.k,
                                          b.data() + (shared_b ? 0 : i) * s.k * s.n, s.m, s.n, s.k);
                    for (std::size_t j = 0; j < r.value.size(); ++j) {
                        const double initial = c0[static_cast<std::size_t>(i * s.m * s.n) + j];
                        want.push_back(alpha * r.value[j] + beta * initial);
                        magnitude.push_back(std::abs(alpha) * r.magnitude[j] +
                                            std::abs(beta * initial));
                    }
                }

                std::vector<std::int64_t> a_offsets, b_offsets;
                for (std::int64_t i = 0; i < s.batch; ++i) {
                    a_offsets.push_back(i * s.m * s.k);
                    b_offsets.push_back(shared_b ? 0 : i * s.k * s.n);
                }
                const std::string size = std::to_string(s.batch) + "x" + std::to_string(s.m) + "x" +
                                         std::to_string(s.n) + "x" + std::to_string(s.k) +
                                         (shared_b ? " shared b" : "") + (with_c ? " with c" : "");
                for (int threads : {1, 2, 3, original_threads}) {
                    cpu::set_max_threads(threads);
                    for (int v = 0; v < 4; ++v) {
                        // Start from garbage so a kernel that accumulates instead of overwriting
                        // is caught.
                        std::vector<float> out(want.size(), 123.0f);
                        cpu::BatchedGemm g;
                        g.a = a.data();
                        g.a_offsets = a_offsets.data();
                        g.rsa = s.k;
                        g.csa = 1;
                        g.b = b.data();
                        g.b_offsets = b_offsets.data();
                        g.rsb = s.n;
                        g.csb = 1;
                        g.c = out.data();
                        g.c0 = with_c ? c0.data() : nullptr;
                        g.batch = s.batch;
                        g.m = s.m;
                        g.n = s.n;
                        g.k = s.k;
                        g.alpha = alpha;
                        g.beta = beta;
                        kernels[v](g);
                        require_close(out, want, magnitude,
                                      std::string(names[v]) + " " + size +
                                          " threads=" + std::to_string(threads));
                    }
                }
            }
        }
    }
    cpu::set_max_threads(original_threads);
}

// ---- public API ----------------------------------------------------------------------------

// Reference for gemm_strided_batched on host data: a, b and c are contiguous in their given shapes,
// and every output element is computed from the broadcast batch index directly.
struct Expected {
    Dims shape;
    std::vector<double> value, magnitude;
};

Expected batched_reference(const std::vector<float>& a, const Dims& a_shape,
                           const std::vector<float>& b, const Dims& b_shape, float alpha,
                           const std::vector<float>* c, const Dims& c_shape, float beta,
                           bool transpose_a, bool transpose_b, const Dims& out_shape) {
    Expected e;
    e.shape = out_shape;
    const std::size_t nd = out_shape.size();
    const std::int64_t m = out_shape[nd - 2], n = out_shape[nd - 1];
    const std::int64_t k = transpose_a ? a_shape[a_shape.size() - 2] : a_shape.back();
    // Index of a broadcast operand of shape `shape` for output batch index `out`, matrix (r, q).
    auto element = [&](const Dims& shape, const Dims& out, std::int64_t r, std::int64_t q) {
        Dims index(shape.size());
        const std::size_t lead = nd - shape.size();
        for (std::size_t d = 0; d + 2 < shape.size(); ++d) {
            index[d] = shape[d] == 1 ? 0 : out[lead + d];
        }
        index[shape.size() - 2] = r;
        index[shape.size() - 1] = q;
        return reference::ravel(index, shape);
    };
    for (std::int64_t flat = 0; flat < reference::count(out_shape); ++flat) {
        const Dims out = reference::unravel(flat, out_shape);
        const std::int64_t i = out[nd - 2], j = out[nd - 1];
        double sum = 0.0, magnitude = 0.0;
        for (std::int64_t p = 0; p < k; ++p) {
            const float x = a[static_cast<std::size_t>(transpose_a ? element(a_shape, out, p, i)
                                                                   : element(a_shape, out, i, p))];
            const float y = b[static_cast<std::size_t>(transpose_b ? element(b_shape, out, j, p)
                                                                   : element(b_shape, out, p, j))];
            sum += static_cast<double>(x) * y;
            magnitude += std::abs(static_cast<double>(x) * y);
        }
        double initial = 0.0;
        if (c != nullptr) {
            // c broadcasts over every dimension, including the matrix ones.
            Dims index(c_shape.size());
            const std::size_t lead = nd - c_shape.size();
            for (std::size_t d = 0; d < c_shape.size(); ++d) {
                index[d] = c_shape[d] == 1 ? 0 : out[lead + d];
            }
            initial = (*c)[static_cast<std::size_t>(reference::ravel(index, c_shape))];
        }
        e.value.push_back(alpha * sum + beta * initial);
        e.magnitude.push_back(std::abs(alpha) * magnitude + std::abs(beta * initial));
    }
    (void)m;
    (void)n;
    return e;
}

struct ApiCase {
    const char* name;
    Dims a_shape, b_shape;
    bool transpose_a, transpose_b;
    float alpha;
    Dims c_shape; // empty: no c
    float beta;
    Dims out_shape;
};

const ApiCase api_cases[] = {
    {"plain batch", {3, 4, 5}, {3, 5, 2}, false, false, 1.0f, {}, 0.0f, {3, 4, 2}},
    {"shared right-hand side", {4, 3, 5}, {5, 2}, false, false, 1.0f, {}, 0.0f, {4, 3, 2}},
    {"shared left-hand side", {3, 5}, {4, 5, 2}, false, false, 1.0f, {}, 0.0f, {4, 3, 2}},
    {"broadcast batch dims", {2, 1, 3, 4}, {5, 4, 6}, false, false, 1.0f, {}, 0.0f, {2, 5, 3, 6}},
    {"plain matrices", {7, 9}, {9, 4}, false, false, 1.0f, {}, 0.0f, {7, 4}},
    {"transpose a", {2, 5, 3}, {2, 5, 4}, true, false, 1.0f, {}, 0.0f, {2, 3, 4}},
    {"transpose b", {2, 3, 5}, {2, 4, 5}, false, true, 1.0f, {}, 0.0f, {2, 3, 4}},
    {"transpose both", {2, 5, 3}, {1, 4, 5}, true, true, 1.0f, {}, 0.0f, {2, 3, 4}},
    {"alpha only", {2, 3, 4}, {2, 4, 5}, false, false, -1.5f, {}, 0.0f, {2, 3, 5}},
    {"alpha and full c", {2, 3, 4}, {2, 4, 5}, false, false, 2.0f, {2, 3, 5}, 0.5f, {2, 3, 5}},
    {"c broadcast over batch", {4, 3, 4}, {4, 5}, false, false, 1.0f, {3, 5}, -1.0f, {4, 3, 5}},
    {"c as a row", {2, 3, 4}, {2, 4, 5}, false, false, 1.0f, {5}, 1.0f, {2, 3, 5}},
    {"c given, beta 0", {2, 3, 4}, {2, 4, 5}, false, false, 1.0f, {2, 3, 5}, 0.0f, {2, 3, 5}},
    {"empty batch", {0, 3, 4}, {0, 4, 5}, false, false, 1.0f, {}, 0.0f, {0, 3, 5}},
    {"empty rows", {2, 0, 4}, {2, 4, 5}, false, false, 1.0f, {}, 0.0f, {2, 0, 5}},
    {"k is 0", {2, 3, 0}, {2, 0, 5}, false, false, 1.0f, {}, 0.0f, {2, 3, 5}},
    {"k is 0 with c", {2, 3, 0}, {2, 0, 5}, false, false, 1.0f, {2, 3, 5}, 3.0f, {2, 3, 5}},
    {"many small", {100, 2, 3}, {100, 3, 2}, false, false, 1.0f, {}, 0.0f, {100, 2, 2}},
    {"larger", {3, 70, 90}, {3, 90, 60}, false, false, 1.0f, {}, 0.0f, {3, 70, 60}},
};

void test_api(Device device) {
    const Execution on = test::execution_on(device);
    for (const ApiCase& t : api_cases) {
        const auto a = make_input(reference::count(t.a_shape), 4);
        const auto b = make_input(reference::count(t.b_shape), 5);
        const auto c = make_input(reference::count(t.c_shape), 6);
        const bool has_c = !t.c_shape.empty();
        const Tensor ta = Tensor::from_vector(a, t.a_shape, device);
        const Tensor tb = Tensor::from_vector(b, t.b_shape, device);
        const Tensor tc = has_c ? Tensor::from_vector(c, t.c_shape, device) : Tensor();
        const std::string what = std::string(t.name) + " on " + to_string(device);

        // Auto runs on the CPU (copying from a GPU and back); on the CPU also try it explicitly.
        for (const Execution execution : {Execution::Auto, on}) {
            if (execution != Execution::Auto && device.type != DeviceType::CPU) {
                continue; // no GPU kernel yet; checked in test_errors
            }
            const Tensor out = gemm_strided_batched(ta, tb, t.alpha, has_c ? &tc : nullptr, t.beta,
                                                    t.transpose_a, t.transpose_b, execution);
            test::require(out.shape() == t.out_shape, what + ": shape " + to_string(out.shape()) +
                                                          ", expected " + to_string(t.out_shape));
            test::require(out.device() == device, what + ": result on the wrong device");
            test::require(out.is_contiguous(), what + ": result not contiguous");
            const Expected e =
                batched_reference(a, t.a_shape, b, t.b_shape, t.alpha, has_c ? &c : nullptr,
                                  t.c_shape, t.beta, t.transpose_a, t.transpose_b, t.out_shape);
            require_close(out.to_vector<float>(), e.value, e.magnitude, what);
        }
    }
}

// Views go in as they are: a batch sliced with a step, a transposed view without the flag, and an
// expanded batch.
void test_views() {
    const auto big = make_input(6 * 4 * 5, 7);
    const Tensor full = Tensor::from_vector(big, {6, 4, 5});
    const Tensor every_other = full.slice(0, 0, 6, 2); // [3, 4, 5], batch stride 40
    const Tensor b = Tensor::from_vector(make_input(3 * 5 * 2, 8), {3, 5, 2});
    const Tensor packed = every_other.contiguous();
    test::require(gemm_strided_batched(every_other, b).to_vector<float>() ==
                      gemm_strided_batched(packed, b).to_vector<float>(),
                  "sliced batch differs from its packed copy");

    const Tensor bt = Tensor::from_vector(make_input(3 * 2 * 5, 9), {3, 2, 5});
    test::require(
        gemm_strided_batched(packed, bt.transpose(1, 2)).to_vector<float>() ==
            gemm_strided_batched(packed, bt, 1.0f, nullptr, 0.0f, false, true).to_vector<float>(),
        "transposed view differs from transpose_b");

    const Tensor one = Tensor::from_vector(make_input(4 * 5, 10), {1, 4, 5});
    const Tensor expanded = one.expand({3, 4, 5});
    test::require(gemm_strided_batched(expanded, b).to_vector<float>() ==
                      gemm_strided_batched(expanded.contiguous(), b).to_vector<float>(),
                  "expanded batch differs from its packed copy");

    // The 2-D case agrees with matmul.
    const Tensor x = Tensor::from_vector(make_input(8 * 6, 11), {8, 6});
    const Tensor y = Tensor::from_vector(make_input(6 * 3, 12), {6, 3});
    test::require(gemm_strided_batched(x, y).to_vector<float>() == matmul(x, y).to_vector<float>(),
                  "2-D gemm_strided_batched differs from matmul");
}

void test_errors() {
    const Tensor a = Tensor::ones({2, 3, 4});
    const Tensor b = Tensor::ones({2, 4, 5});
    using E = std::invalid_argument;
    test::require_throws<E>([&] { gemm_strided_batched(Tensor::ones({4}), b); }, "1-D a");
    test::require_throws<E>([&] { gemm_strided_batched(a, Tensor::ones({4})); }, "1-D b");
    test::require_throws<E>([&] { gemm_strided_batched(Tensor(), b); }, "undefined a");
    test::require_throws<E>([&] { gemm_strided_batched(a, Tensor::ones({2, 3, 5})); },
                            "inner dimensions differ");
    test::require_throws<E>([&] { gemm_strided_batched(a, Tensor::ones({3, 4, 5})); },
                            "batch 2 and 3 do not broadcast");
    test::require_throws<E>([&] { gemm_strided_batched(a, b, 1.0f, nullptr, 1.0f); },
                            "beta without c");
    const Tensor bad_c = Tensor::ones({4, 5});
    test::require_throws<E>([&] { gemm_strided_batched(a, b, 1.0f, &bad_c, 1.0f); },
                            "c that does not broadcast");
    const Tensor big_c = Tensor::ones({1, 2, 3, 5});
    test::require_throws<E>([&] { gemm_strided_batched(a, b, 1.0f, &big_c, 1.0f); },
                            "c with more dimensions than the result");
    test::require_throws<E>([&] { gemm_strided_batched(a, Tensor::ones({2, 4, 5}, DType::I32)); },
                            "dtypes differ");
    for (Device device : list_devices()) {
        if (device.type == DeviceType::CPU) {
            continue;
        }
        test::require_throws<E>([&] { gemm_strided_batched(a, b.to(device)); }, "mixed devices");
        test::require_throws<std::runtime_error>(
            [&] {
                gemm_strided_batched(a.to(device), b.to(device), 1.0f, nullptr, 0.0f, false, false,
                                     Execution::GPU);
            },
            "no GPU kernel yet");
    }
}

} // namespace

int main() {
    return test::run("gemm_batched_test", [] {
        test_kernels();
        for (Device device : list_devices()) {
            test_api(device);
        }
        test_views();
        test_errors();
    });
}
