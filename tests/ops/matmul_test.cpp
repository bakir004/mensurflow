// Tests matrix multiplication: the sequential and parallel CPU kernels against a float64 reference
// (including sizes that are not multiples of the block sizes, and several thread counts), and
// mensurflow::matmul through the public API on every device.
#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/blas.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

using Kernel = void (*)(const float*, const float*, float*, std::int64_t, std::int64_t,
                        std::int64_t);

// Multiples of 0.25 in [-4, 4].
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

// Every entry of `got` must be within float rounding of the float64 reference.
void check_against_reference(const std::vector<float>& got, const std::vector<float>& a,
                             const std::vector<float>& b, std::int64_t m, std::int64_t n,
                             std::int64_t k, const std::string& what) {
    const auto want = reference::matmul(a.data(), b.data(), m, n, k);
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(std::abs(static_cast<double>(got[i]) - want.value[i]) <=
                          reference::matmul_tolerance(want.magnitude[i]),
                      what + ": entry " + std::to_string(i) + " got " + std::to_string(got[i]) +
                          ", want " + std::to_string(want.value[i]));
    }
}

struct Shape3 {
    std::int64_t m, n, k;
};

// Sizes below, at, and above the block sizes (256 columns, 128 depth), thin and wide, and 1.
const Shape3 shapes[] = {{1, 1, 1},      {1, 5, 1},       {5, 1, 3},    {3, 5, 7},
                         {16, 16, 16},   {17, 31, 13},    {64, 64, 64}, {65, 129, 130},
                         {100, 300, 50}, {200, 100, 300}, {2, 1000, 3}, {1000, 2, 3},
                         {257, 129, 257}};

void test_kernels() {
    const int original_threads = cpu::max_threads();
    for (const Shape3& s : shapes) {
        const auto a = make_input(s.m * s.k, 1);
        const auto b = make_input(s.k * s.n, 2);
        const std::string size =
            std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k);

        // Start from garbage so a kernel that accumulates instead of overwriting is caught.
        std::vector<float> sequential(static_cast<std::size_t>(s.m * s.n), 123.0f);
        cpu::matmul_f32_sequential(a.data(), b.data(), sequential.data(), s.m, s.n, s.k);
        check_against_reference(sequential, a, b, s.m, s.n, s.k, "sequential " + size);

        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<float> parallel(sequential.size(), 123.0f);
            cpu::matmul_f32_parallel(a.data(), b.data(), parallel.data(), s.m, s.n, s.k);
            check_against_reference(parallel, a, b, s.m, s.n, s.k,
                                    "parallel " + size + " threads=" + std::to_string(threads));
        }
    }
    cpu::set_max_threads(original_threads);
}

void test_kernel_edges() {
    float out[4] = {9, 9, 9, 9};
    // k == 0: the product of an m x 0 and a 0 x n matrix is an m x n matrix of zeros.
    cpu::matmul_f32_parallel(nullptr, nullptr, out, 2, 2, 0);
    test::require(out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0, "k == 0 gives zeros");
    out[0] = 9;
    cpu::matmul_f32_sequential(nullptr, nullptr, out, 2, 2, 0);
    test::require(out[0] == 0, "sequential k == 0 gives zeros");
    // m == 0 or n == 0: nothing to do, nothing touched.
    cpu::matmul_f32_parallel(nullptr, nullptr, nullptr, 0, 3, 3);
    cpu::matmul_f32_parallel(nullptr, nullptr, nullptr, 3, 0, 3);
}

bool supported(Device device) {
    return detail::Registry::instance().dispatcher().find(
               {detail::OpId::Matmul, device.type, DType::F32}) != nullptr;
}

void test_known_values(Device device) {
    const Execution on = test::execution_on(device);
    const Tensor a = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
    const Tensor b = Tensor::from_vector(std::vector<float>{7, 8, 9, 10, 11, 12}, {3, 2}, device);
    const Tensor c = matmul(a, b, on);
    test::require(c.shape() == (Shape{2, 2}) && c.dtype() == DType::F32 && c.device() == device,
                  "shape, dtype and device of the result");
    test::require(c.is_contiguous(), "result is contiguous");
    test::require(c.to_vector<float>() == (std::vector<float>{58, 64, 139, 154}),
                  "[[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]]");

    // Multiplying by the identity gives the matrix back exactly.
    const std::int64_t m = 5, k = 7;
    const auto values = make_input(m * k, 3);
    std::vector<float> eye(static_cast<std::size_t>(k * k), 0.0f);
    for (std::int64_t i = 0; i < k; ++i) {
        eye[static_cast<std::size_t>(i * k + i)] = 1.0f;
    }
    const Tensor id = matmul(Tensor::from_vector(values, {m, k}, device),
                             Tensor::from_vector(eye, {k, k}, device), on);
    test::require(id.to_vector<float>() == values, "A x I == A");
}

void test_sizes(Device device) {
    const Execution on = test::execution_on(device);
    for (const Shape3& s : shapes) {
        const auto a = make_input(s.m * s.k, 4);
        const auto b = make_input(s.k * s.n, 5);
        const Tensor c = matmul(Tensor::from_vector(a, {s.m, s.k}, device),
                                Tensor::from_vector(b, {s.k, s.n}, device), on);
        check_against_reference(c.to_vector<float>(), a, b, s.m, s.n, s.k,
                                "matmul " + std::to_string(s.m) + "x" + std::to_string(s.n) + "x" +
                                    std::to_string(s.k));
    }
}

void test_views(Device device) {
    const Execution on = test::execution_on(device);
    const std::int64_t m = 6, k = 5, n = 4;
    const auto a = make_input(m * k, 6);
    const auto b = make_input(k * n, 7);

    // A stored as [k, m] and transposed to [m, k]; B stored as [n, k] and transposed to [k, n].
    std::vector<float> a_stored(static_cast<std::size_t>(k * m)),
        b_stored(static_cast<std::size_t>(n * k));
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t p = 0; p < k; ++p) {
            a_stored[static_cast<std::size_t>(p * m + i)] = a[static_cast<std::size_t>(i * k + p)];
        }
    }
    for (std::int64_t p = 0; p < k; ++p) {
        for (std::int64_t j = 0; j < n; ++j) {
            b_stored[static_cast<std::size_t>(j * k + p)] = b[static_cast<std::size_t>(p * n + j)];
        }
    }
    const Tensor ta = Tensor::from_vector(a_stored, {k, m}, device).transpose(0, 1);
    const Tensor tb = Tensor::from_vector(b_stored, {n, k}, device).transpose(0, 1);
    test::require(!ta.is_contiguous() && !tb.is_contiguous(), "inputs are strided views");
    check_against_reference(matmul(ta, tb, on).to_vector<float>(), a, b, m, n, k,
                            "transposed inputs");

    // Rows 1..4 of a larger matrix: contiguous, but starting partway into the allocation.
    const Tensor big = Tensor::from_vector(make_input(8 * k, 8), {8, k}, device);
    const auto rows = big.slice(0, 1, 5).to_vector<float>();
    const Tensor sliced = matmul(big.slice(0, 1, 5), Tensor::from_vector(b, {k, n}, device), on);
    check_against_reference(sliced.to_vector<float>(), rows, b, 4, n, k, "sliced rows");
}

// Stores a logical rows x cols matrix (row-major in `values`) as a view with the given kind of
// layout: 0 contiguous, 1 columns sliced out of a wider matrix (leading dimension larger than the
// width), 2 transposed, 3 transposed and sliced. The view always has shape {rows, cols}.
Tensor make_view(const std::vector<float>& values, std::int64_t rows, std::int64_t cols, int kind,
                 Device device) {
    const auto fill = [&](std::int64_t stored_rows, std::int64_t stored_cols, std::int64_t row0,
                          std::int64_t col0, bool transposed) {
        std::vector<float> stored(static_cast<std::size_t>(stored_rows * stored_cols), -77.0f);
        for (std::int64_t i = 0; i < rows; ++i) {
            for (std::int64_t j = 0; j < cols; ++j) {
                // Stored row r, column c holds the element: for the transposed layouts the logical
                // row i runs along the stored columns and the logical column j along the stored rows.
                const std::int64_t r = transposed ? row0 + j : row0 + i;
                const std::int64_t c = transposed ? col0 + i : col0 + j;
                stored[static_cast<std::size_t>(r * stored_cols + c)] =
                    values[static_cast<std::size_t>(i * cols + j)];
            }
        }
        return Tensor::from_vector(stored, {stored_rows, stored_cols}, device);
    };
    switch (kind) {
    case 0:
        return Tensor::from_vector(values, {rows, cols}, device);
    case 1:
        return fill(rows + 2, cols + 9, 1, 3, false).slice(0, 1, 1 + rows).slice(1, 3, 3 + cols);
    case 2:
        return fill(cols, rows, 0, 0, true).transpose(0, 1);
    default:
        return fill(cols + 3, rows + 5, 2, 4, true)
            .slice(0, 2, 2 + cols)
            .slice(1, 4, 4 + rows)
            .transpose(0, 1);
    }
}

// Products big enough to be split over threads, with every combination of layouts for A and B, through
// the public API.
void test_strided_large(Device device) {
    const Execution on = test::execution_on(device);
    const Shape3 sizes[] = {{70, 60, 90}, {130, 200, 150}, {300, 300, 300}};
    for (const Shape3& s : sizes) {
        const auto a = make_input(s.m * s.k, 21);
        const auto b = make_input(s.k * s.n, 22);
        for (int kind_a = 0; kind_a < 4; ++kind_a) {
            for (int kind_b = 0; kind_b < 4; ++kind_b) {
                if (s.m == 300 && (kind_a == 1 || kind_b == 1)) {
                    continue; // the biggest size only in the contiguous and transposed layouts
                }
                const Tensor va = make_view(a, s.m, s.k, kind_a, device);
                const Tensor vb = make_view(b, s.k, s.n, kind_b, device);
                check_against_reference(matmul(va, vb, on).to_vector<float>(), a, b, s.m, s.n, s.k,
                                        "strided " + std::to_string(s.m) + "x" + std::to_string(s.n) +
                                            "x" + std::to_string(s.k) + " A" + std::to_string(kind_a) +
                                            " B" + std::to_string(kind_b));
            }
        }
    }
}

// The backward pass multiplies by transposed views and by a broadcast gradient (stride 0). For
// loss = sum(a x b), the gradient of a[i][p] is the sum of row p of b, and of b[p][j] the sum of
// column p of a.
void test_gradient(Device device) {
    if (device.type != DeviceType::CPU) {
        return; // autograd is exercised on the CPU
    }
    const std::int64_t m = 70, k = 90, n = 60;
    const auto av = make_input(m * k, 31);
    const auto bv = make_input(k * n, 32);
    Tensor a = Tensor::from_vector(av, {m, k}, device);
    Tensor b = Tensor::from_vector(bv, {k, n}, device);
    a.set_requires_grad();
    b.set_requires_grad();
    sum(matmul(a, b)).backward();

    const auto grad_a = a.grad().to_vector<float>();
    const auto grad_b = b.grad().to_vector<float>();
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t p = 0; p < k; ++p) {
            float expected = 0.0f;
            for (std::int64_t j = 0; j < n; ++j) {
                expected += bv[static_cast<std::size_t>(p * n + j)];
            }
            test::require(grad_a[static_cast<std::size_t>(i * k + p)] == expected,
                          "gradient of a at (" + std::to_string(i) + "," + std::to_string(p) + ")");
        }
    }
    for (std::int64_t p = 0; p < k; ++p) {
        for (std::int64_t j = 0; j < n; ++j) {
            float expected = 0.0f;
            for (std::int64_t i = 0; i < m; ++i) {
                expected += av[static_cast<std::size_t>(i * k + p)];
            }
            test::require(grad_b[static_cast<std::size_t>(p * n + j)] == expected,
                          "gradient of b at (" + std::to_string(p) + "," + std::to_string(j) + ")");
        }
    }
}

void test_empty(Device device) {
    const Execution on = test::execution_on(device);
    const Tensor none_k = matmul(Tensor::empty({3, 0}, DType::F32, device),
                                 Tensor::empty({0, 4}, DType::F32, device), on);
    test::require(none_k.shape() == (Shape{3, 4}), "k == 0 shape");
    const auto zeros = none_k.to_vector<float>();
    test::require(zeros == std::vector<float>(12, 0.0f), "k == 0 gives zeros");

    const Tensor none_m = matmul(Tensor::empty({0, 5}, DType::F32, device),
                                 Tensor::empty({5, 2}, DType::F32, device), on);
    test::require(none_m.shape() == (Shape{0, 2}) && none_m.numel() == 0, "m == 0");
    const Tensor none_n = matmul(Tensor::empty({2, 5}, DType::F32, device),
                                 Tensor::empty({5, 0}, DType::F32, device), on);
    test::require(none_n.shape() == (Shape{2, 0}) && none_n.numel() == 0, "n == 0");
}

void test_errors(Device device) {
    const Execution on = test::execution_on(device);
    const Tensor m23 = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
    const Tensor m32 = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {3, 2}, device);
    const Tensor vec = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device);
    const Tensor ints =
        Tensor::from_vector(std::vector<std::int32_t>{1, 2, 3, 4, 5, 6}, {2, 3}, device);
    test::require_throws<std::invalid_argument>([&] { matmul(m23, m23, on); }, "inner mismatch");
    test::require_throws<std::invalid_argument>([&] { matmul(vec, m32, on); }, "1-D left");
    test::require_throws<std::invalid_argument>([&] { matmul(m23, vec, on); }, "1-D right");
    test::require_throws<std::invalid_argument>([&] { matmul(Tensor(), m32, on); }, "undefined");
    test::require_throws<std::invalid_argument>([&] { matmul(ints, m32, on); }, "dtype mismatch");
    // No int32 kernel exists on any backend yet; the failure must be a clear error.
    test::require_throws<std::runtime_error>([&] { matmul(ints, ints.transpose(0, 1), on); },
                                             "unsupported dtype");
}

void test_unsupported(Device device) {
    const Execution on = test::execution_on(device);
    const Tensor a = Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {2, 2}, device);
    test::require_throws<detail::NotImplemented>([&] { matmul(a, a, on); },
                                                 "matmul without a kernel on " + to_string(device));
}

} // namespace

int main() {
    return test::run("matmul_test", [] {
        test_kernels();
        test_kernel_edges();
        for (Device device : list_devices()) {
            std::cout << "  matmul on " << to_string(device) << '\n';
            if (!supported(device)) {
                // Only the CPU has a matmul kernel so far; the others must say so clearly.
                test_unsupported(device);
                continue;
            }
            test_known_values(device);
            test_sizes(device);
            test_views(device);
            test_strided_large(device);
            test_gradient(device);
            test_empty(device);
            test_errors(device);
        }
    });
}
