// Strides in matrix multiplication, two questions:
//
//   1. What did copying a transposed operand to contiguous memory first (what matmul used to do for
//      every non-contiguous input) cost? "copy" times contiguous() followed by matmul, "in place" times
//      matmul on the view itself, which the packing step now reads directly. The backward pass of
//      matmul is made of exactly such products: grad_a = grad x b^T and grad_b = a^T x grad.
//   2. Does the leading dimension of a matrix matter? A row stride that is a large power of two (n =
//      2048 floats = 8 KB) makes the rows of a block map to the same cache sets. The kernel level
//      sweep multiplies matrices stored with row strides n, n + 16 and n + 1.
//
//   gemm_stride_benchmark [sizes...]      (default 512 1024 2048)
#include "bench.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(300)); }

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int64_t> sizes;
    for (int i = 1; i < argc; ++i) {
        sizes.push_back(std::atoll(argv[i]));
    }
    if (sizes.empty()) {
        sizes = {512, 1024, 2048};
    }
    bench::Config config;
    config.repeats = 7;
    bench::print_header("Strided operands in matmul", config,
                        {"part 1 through the public API (Tensor), part 2 on the kernel"});

    // ---- Part 1: transposed operands, copy first vs in place ------------------------------------
    std::printf(
        "\n== 1. Transposed operands: copy first (old) vs read in place (new), GFLOP/s ==\n");
    std::printf("%6s  %-14s %10s %10s %9s\n", "n", "operands", "copy", "in place", "speedup");
    for (std::int64_t n : sizes) {
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        std::vector<float> a_values(static_cast<std::size_t>(n * n)), b_values(a_values.size());
        bench::fill_a(a_values.data(), a_values.size());
        bench::fill_b(b_values.data(), b_values.size());
        const Tensor a = Tensor::from_vector(a_values, {n, n});
        const Tensor b = Tensor::from_vector(b_values, {n, n});
        const Tensor at = a.transpose(0, 1);
        const Tensor bt = b.transpose(0, 1);

        struct Case {
            const char* name;
            const Tensor* left;
            const Tensor* right;
        };

        const Case cases[] = {
            {"A B", &a, &b}, {"A^T B", &at, &b}, {"A B^T", &a, &bt}, {"A^T B^T", &at, &bt}};
        for (const Case& c : cases) {
            Tensor result;
            const auto observe = [&] { return static_cast<double>(result.to_vector<float>()[0]); };
            settle();
            const bench::Sample copy = bench::measure(
                config,
                [&] {
                    const Tensor x = c.left->is_contiguous() ? *c.left : c.left->contiguous();
                    const Tensor y = c.right->is_contiguous() ? *c.right : c.right->contiguous();
                    result = matmul(x, y);
                },
                observe);
            settle();
            const bench::Sample in_place =
                bench::measure(config, [&] { result = matmul(*c.left, *c.right); }, observe);
            std::printf("%6lld  %-14s %10.1f %10.1f %8.2fx\n", static_cast<long long>(n), c.name,
                        flops / copy.median_s / 1e9, flops / in_place.median_s / 1e9,
                        copy.median_s / in_place.median_s);
        }
    }

    // ---- Part 2: leading dimension of the matrices ---------------------------------------------
    std::printf("\n== 2. Row stride of A, B and C (kernel level, all threads), GFLOP/s ==\n");
    std::printf("%6s  %12s %12s %12s\n", "n", "ld = n", "ld = n + 16", "ld = n + 1");
    for (std::int64_t n : sizes) {
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        double rate[3] = {0, 0, 0};
        const std::int64_t pads[3] = {0, 16, 1};
        for (int variant = 0; variant < 3; ++variant) {
            const std::int64_t ld = n + pads[variant];
            const auto len = static_cast<std::size_t>(n * ld);
            bench::Arrays arrays(len, 3);
            float* a = arrays[0];
            float* b = arrays[1];
            float* c = arrays[2];
            bench::fill_a(a, len);
            bench::fill_b(b, len);
            settle();
            const bench::Sample sample = bench::measure(
                config,
                [&] {
                    cpu::matmul_f32_packed_parallel_strided(a, ld, 1, b, ld, 1, c, ld, n, n, n);
                },
                [&] { return bench::checksum(c, len); });
            rate[variant] = flops / sample.median_s / 1e9;
        }
        std::printf("%6lld  %12.1f %12.1f %12.1f\n", static_cast<long long>(n), rate[0], rate[1],
                    rate[2]);
    }
    return 0;
}
