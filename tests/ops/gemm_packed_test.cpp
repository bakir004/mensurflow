// Tests the packed AVX2 GEMM (matmul_f32_packed) against a float64 reference on shapes around every
// tile and block boundary: the 6 x 16 tile, MC = 168, KC = 256, NC = 4080, and degenerate sizes.
#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

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

using Kernel = void (*)(const float*, const float*, float*, std::int64_t, std::int64_t,
                        std::int64_t);

void check_shape(Kernel kernel, const char* name, std::int64_t m, std::int64_t n, std::int64_t k) {
    const std::string what = std::string(name) + " m=" + std::to_string(m) +
                             " n=" + std::to_string(n) + " k=" + std::to_string(k);
    const auto a = make_input(m * k, 1);
    const auto b = make_input(k * n, 2);
    // Poison C to prove every entry is written, not accumulated into.
    std::vector<float> c(static_cast<std::size_t>(m * n), 12345.0f);
    kernel(a.data(), b.data(), c.data(), m, n, k);
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            double expected = 0.0;
            for (std::int64_t p = 0; p < k; ++p) {
                expected += static_cast<double>(a[i * k + p]) * static_cast<double>(b[p * n + j]);
            }
            const float got = c[static_cast<std::size_t>(i * n + j)];
            // Products are exact in float; only the summation order/rounding can differ.
            const double tolerance =
                1e-5 * (1.0 + std::abs(expected)) * (1.0 + static_cast<double>(k) / 64.0);
            test::require(std::abs(static_cast<double>(got) - expected) <= tolerance,
                          what + " wrong at (" + std::to_string(i) + "," + std::to_string(j) +
                              "): got " + std::to_string(got) + " expected " +
                              std::to_string(expected));
        }
    }
}

// ---- Strided operands ----------------------------------------------------------------------------

using StridedKernel = void (*)(const float*, std::int64_t, std::int64_t, const float*, std::int64_t,
                               std::int64_t, float*, std::int64_t, std::int64_t, std::int64_t,
                               std::int64_t);

// How a rows x cols matrix is stored: element (i, j) is at i * rs + j * cs.
//   0 row-major         1 row-major with padding        2 transposed (column-major)
//   3 transposed with padding     4 no unit stride at all (elements interleaved with another matrix)
void layout_strides(int kind, std::int64_t rows, std::int64_t cols, std::int64_t& rs,
                    std::int64_t& cs) {
    switch (kind) {
    case 0:
        rs = cols;
        cs = 1;
        break;
    case 1:
        rs = cols + 5;
        cs = 1;
        break;
    case 2:
        rs = 1;
        cs = rows;
        break;
    case 3:
        rs = 1;
        cs = rows + 3;
        break;
    default:
        rs = 2;
        cs = 2 * rows;
        break;
    }
}

// C = A * B with A (m x k) and B (k x n) stored as `kind_a` and `kind_b`, and C with a row stride
// larger than n (the unused columns must keep their value). Against a float64 reference.
void check_strided(StridedKernel kernel, const std::string& name, int kind_a, int kind_b,
                   std::int64_t m, std::int64_t n, std::int64_t k) {
    const std::string what = name + " A" + std::to_string(kind_a) + " B" + std::to_string(kind_b) +
                             " m=" + std::to_string(m) + " n=" + std::to_string(n) +
                             " k=" + std::to_string(k);
    std::int64_t rsa, csa, rsb, csb;
    layout_strides(kind_a, m, k, rsa, csa);
    layout_strides(kind_b, k, n, rsb, csb);
    const auto a = make_input((m - 1) * rsa + (k - 1) * csa + 1, 11);
    const auto b = make_input((k - 1) * rsb + (n - 1) * csb + 1, 12);
    const std::int64_t ldc = n + 7;
    constexpr float poison = 12345.0f;
    std::vector<float> c(static_cast<std::size_t>((m - 1) * ldc + n) + 16, poison);
    kernel(a.data(), rsa, csa, b.data(), rsb, csb, c.data(), ldc, m, n, k);
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < ldc; ++j) {
            const float got = c[static_cast<std::size_t>(i * ldc + j)];
            if (j >= n) {
                if (i * ldc + j < static_cast<std::int64_t>(c.size())) {
                    test::require(got == poison, what + ": wrote outside the matrix at (" +
                                                     std::to_string(i) + "," + std::to_string(j) +
                                                     ")");
                }
                continue;
            }
            double expected = 0.0;
            for (std::int64_t p = 0; p < k; ++p) {
                expected += static_cast<double>(a[static_cast<std::size_t>(i * rsa + p * csa)]) *
                            static_cast<double>(b[static_cast<std::size_t>(p * rsb + j * csb)]);
            }
            const double tolerance =
                1e-5 * (1.0 + std::abs(expected)) * (1.0 + static_cast<double>(k) / 64.0);
            test::require(std::abs(static_cast<double>(got) - expected) <= tolerance,
                          what + ": wrong at (" + std::to_string(i) + "," + std::to_string(j) +
                              "): got " + std::to_string(got) + " expected " +
                              std::to_string(expected));
        }
    }
    for (std::size_t i = c.size() - 16; i < c.size(); ++i) {
        test::require(c[i] == poison, what + ": wrote past the end of C");
    }
}

} // namespace

int main() {
    return test::run("gemm_packed_test", [] {
        std::cout << "AVX2 packed kernel: "
                  << (cpu::packed_gemm_supported() ? "yes" : "no (fallback)") << '\n';
        const std::int64_t shapes[][3] = {
            {1, 1, 1},      {1, 17, 3},     {5, 15, 7},    {6, 16, 1},     {6, 16, 256},
            {7, 17, 257},   {12, 32, 512},  {13, 33, 255}, {167, 31, 20},  {168, 16, 20},
            {169, 17, 300}, {337, 49, 513}, {40, 4080, 9}, {20, 4081, 17}, {9, 8200, 5},
            {64, 64, 64},   {100, 1, 100},  {1, 100, 100},
        };
        const std::int64_t extra[][3] = {{11, 7, 5},    {13, 9, 17},   {3, 23, 7},    {5, 25, 9},
                                         {24, 48, 300}, {25, 49, 301}, {50, 50, 1100}};
        // The threaded version needs shapes large enough to really split (the others fall back to
        // the sequential kernel).
        const std::int64_t big[][3] = {
            {200, 300, 100}, {500, 70, 400}, {37, 1000, 300}, {1000, 37, 300}};
        const int saved_threads = cpu::max_threads();
        const cpu::PackedGemmBlocking original = cpu::packed_gemm_blocking();

        // Every micro-kernel shape, with the default block sizes and with tiny ones, so partial
        // tiles and every block boundary (rows, depth, columns) are crossed on small matrices.
        const char* shape_names[] = {"6x16", "6x16 unrolled", "4x24", "12x8", "6x16 reuse x2", "6x16 reuse x4", "6x16 pipelined", "6x16 pipelined x2", "6x16 reuse x2 fenced", "6x16 reuse x4 fenced", "6x16 pipelined x2 fenced", "6x16 pipelined x4 fenced"};
        for (int shape = 0; shape < 12; ++shape) {
            for (int tiny = 0; tiny < 2; ++tiny) {
                cpu::PackedGemmBlocking blocking = original;
                blocking.shape = shape;
                if (tiny != 0) {
                    blocking.mc = 24;
                    blocking.kc = 20;
                    blocking.nc = 48;
                }
                cpu::set_packed_gemm_blocking(blocking);
                const std::string name =
                    std::string("packed ") + shape_names[shape] + (tiny != 0 ? " tiny blocks" : "");
                for (const auto& s : shapes) {
                    check_shape(cpu::matmul_f32_packed, name.c_str(), s[0], s[1], s[2]);
                }
                for (const auto& s : extra) {
                    check_shape(cpu::matmul_f32_packed, name.c_str(), s[0], s[1], s[2]);
                }
                // 24 threads for every shape, plus 1 to 8 threads for the default shape.
                std::vector<int> thread_counts = {3, 24};
                if (shape == 0 && tiny == 0) {
                    thread_counts = {1, 2, 3, 8, 24};
                }
                for (int threads : thread_counts) {
                    cpu::set_max_threads(threads);
                    for (const auto& s : shapes) {
                        check_shape(cpu::matmul_f32_packed_parallel, (name + " parallel").c_str(),
                                    s[0], s[1], s[2]);
                    }
                    for (const auto& s : big) {
                        check_shape(cpu::matmul_f32_packed_parallel, (name + " parallel").c_str(),
                                    s[0], s[1], s[2]);
                    }
                    for (const auto& s : extra) {
                        check_shape(cpu::matmul_f32_packed_parallel, (name + " parallel").c_str(),
                                    s[0], s[1], s[2]);
                    }
                }
                cpu::set_max_threads(saved_threads);
            }
        }
        cpu::set_packed_gemm_blocking(original);

        // Every prefetch mode of the kernel and between tiles, in both loop orders, with default and
        // tiny blocks (the between-tile modes run their own tile loop, including partial tiles).
        for (int prefetch = 0; prefetch <= 6; ++prefetch) {
            for (int prefetch_block = 0; prefetch_block <= 4; ++prefetch_block) {
                for (int loop = 0; loop < 2; ++loop) {
                    for (int tiny = 0; tiny < 2; ++tiny) {
                        cpu::PackedGemmBlocking blocking = original;
                        blocking.prefetch = prefetch;
                        blocking.prefetch_block = prefetch_block;
                        blocking.loop_order = loop;
                        if (tiny != 0) {
                            blocking.mc = 24;
                            blocking.kc = 20;
                            blocking.nc = 48;
                        }
                        cpu::set_packed_gemm_blocking(blocking);
                        const std::string name = "prefetch " + std::to_string(prefetch) + "/" +
                                                 std::to_string(prefetch_block) + " loop " +
                                                 std::to_string(loop) + (tiny != 0 ? " tiny" : "");
                        for (const auto& s : extra) {
                            check_shape(cpu::matmul_f32_packed, name.c_str(), s[0], s[1], s[2]);
                        }
                        cpu::set_max_threads(3);
                        for (const auto& s : big) {
                            check_shape(cpu::matmul_f32_packed_parallel, name.c_str(), s[0], s[1],
                                        s[2]);
                        }
                        cpu::set_max_threads(saved_threads);
                    }
                }
            }
        }
        cpu::set_packed_gemm_blocking(original);

        // The ways of parallelizing the loops (strategy 0 to 3), with and without pinning the threads
        // to cores, in both loop orders, with the default block sizes and with tiny ones, on a thread
        // count below and one above the number of cores. Two larger shapes are included because
        // pinning is only done for products of at least 8 million multiply-adds.
        const std::int64_t large[][3] = {{300, 300, 300}, {97, 257, 901}};
        for (int strategy = 0; strategy < 12; ++strategy) { // 6, 7: strategies 4, 5 on the thread pool;
            // 8, 9: 4, 5 with B packed privately per thread; 10, 11: both
            for (int pin : {0, 1}) {
                for (int loop : {0, 1}) {
                    for (int tiny = 0; tiny < 2; ++tiny) {
                        cpu::PackedGemmBlocking blocking = original;
                        blocking.parallel = strategy < 6 ? strategy : 4 + (strategy & 1);
                        blocking.pool = strategy == 6 || strategy == 7 || strategy >= 10 ? 1 : 0;
                        blocking.private_b = strategy >= 8 ? 1 : 0;
                        blocking.pin = pin;
                        blocking.loop_order = loop;
                        if (tiny != 0) {
                            blocking.mc = 24;
                            blocking.kc = 20;
                            blocking.nc = 48;
                            blocking.b_chunk = 2;
                        }
                        cpu::set_packed_gemm_blocking(blocking);
                        const std::string name = "packed parallel strategy " +
                                                 std::to_string(strategy) + " pin " +
                                                 std::to_string(pin) + " loop " +
                                                 std::to_string(loop) + (tiny != 0 ? " tiny" : "");
                        if (strategy >= 8 && strategy < 10 && pin == 0) {
                            // The single-thread kernel has the same private-B path.
                            for (const auto& s : big) {
                                check_shape(cpu::matmul_f32_packed, name.c_str(), s[0], s[1], s[2]);
                            }
                            for (const auto& s : extra) {
                                check_shape(cpu::matmul_f32_packed, name.c_str(), s[0], s[1], s[2]);
                            }
                            for (const auto& s : large) {
                                check_shape(cpu::matmul_f32_packed, name.c_str(), s[0], s[1], s[2]);
                            }
                        }
                        for (int threads : {3, 24}) {
                            cpu::set_max_threads(threads);
                            for (const auto& s : big) {
                                check_shape(cpu::matmul_f32_packed_parallel, name.c_str(), s[0],
                                            s[1], s[2]);
                            }
                            for (const auto& s : extra) {
                                check_shape(cpu::matmul_f32_packed_parallel, name.c_str(), s[0],
                                            s[1], s[2]);
                            }
                            for (const auto& s : large) {
                                check_shape(cpu::matmul_f32_packed_parallel, name.c_str(), s[0],
                                            s[1], s[2]);
                            }
                        }
                        cpu::set_max_threads(saved_threads);
                    }
                }
            }
        }
        cpu::set_packed_gemm_blocking(original);

        // Strided operands (transposed, padded, interleaved) and a padded C, with the single-thread
        // and the threaded kernel, with default and tiny block sizes.
        const std::int64_t strided_shapes[][3] = {{1, 1, 1},      {5, 15, 7},     {13, 33, 255},
                                                  {37, 50, 300},  {169, 17, 300}, {64, 64, 64},
                                                  {200, 100, 70}, {9, 200, 5}};
        for (int tiny = 0; tiny < 2; ++tiny) {
            cpu::PackedGemmBlocking blocking = original;
            if (tiny != 0) {
                blocking.mc = 24;
                blocking.kc = 20;
                blocking.nc = 48;
            }
            cpu::set_packed_gemm_blocking(blocking);
            const std::string suffix = tiny != 0 ? " tiny" : "";
            for (int kind_a = 0; kind_a < 5; ++kind_a) {
                for (int kind_b = 0; kind_b < 5; ++kind_b) {
                    for (const auto& s : strided_shapes) {
                        check_strided(cpu::matmul_f32_packed_strided, "strided" + suffix, kind_a,
                                      kind_b, s[0], s[1], s[2]);
                        for (int threads : {3, 24}) {
                            cpu::set_max_threads(threads);
                            check_strided(cpu::matmul_f32_packed_parallel_strided,
                                          "strided parallel" + suffix, kind_a, kind_b, s[0], s[1],
                                          s[2]);
                        }
                        cpu::set_max_threads(saved_threads);
                    }
                }
            }
        }
        // One product big enough to be pinned (at least 8 million multiply-adds) in the layouts that
        // matter: transposed A, transposed B, both, and a padded view.
        // Every parallel strategy, so each one honours the row stride of C.
        for (int strategy = 0; strategy < 12; ++strategy) { // 6, 7: strategies 4, 5 on the thread pool;
            // 8, 9: 4, 5 with B packed privately per thread; 10, 11: both
            cpu::PackedGemmBlocking blocking = original;
            blocking.parallel = strategy < 6 ? strategy : 4 + (strategy & 1);
            blocking.pool = strategy == 6 || strategy == 7 || strategy >= 10 ? 1 : 0;
            blocking.private_b = strategy >= 8 ? 1 : 0;
            cpu::set_packed_gemm_blocking(blocking);
            for (int kinds : {0, 1, 2, 3}) {
                cpu::set_max_threads(12);
                check_strided(cpu::matmul_f32_packed_parallel_strided,
                              "strided parallel large strategy " + std::to_string(strategy),
                              kinds == 1 || kinds == 3 ? 2 : 1, kinds == 2 || kinds == 3 ? 2 : 1,
                              300, 300, 300);
                cpu::set_max_threads(saved_threads);
            }
        }
        cpu::set_packed_gemm_blocking(original);

        // Degenerate sizes: k == 0 fills C with zeros, m or n == 0 touches nothing.
        std::vector<float> c(6, 7.0f);
        cpu::matmul_f32_packed(nullptr, nullptr, c.data(), 2, 3, 0);
        for (float x : c) {
            test::require(x == 0.0f, "k == 0 must zero C");
        }
        std::vector<float> untouched(4, 7.0f);
        cpu::matmul_f32_packed(nullptr, nullptr, untouched.data(), 0, 4, 5);
        cpu::matmul_f32_packed(nullptr, nullptr, untouched.data(), 4, 0, 5);
        for (float x : untouched) {
            test::require(x == 7.0f, "empty m or n must not write");
        }
    });
}
