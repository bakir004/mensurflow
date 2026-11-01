// Matrix multiplication, scalar vs sequential vs parallel, for square matrices.
//   scalar:     the plain i-k-j loop compiled without auto-vectorization (scalar.cpp).
//   sequential: the same loop with whatever SIMD the compiler finds (sequential.cpp).
//   parallel:   the cache-blocked kernel (matmul.cpp) with 1, 2, 4, ... threads. The 1-thread row
//               isolates what cache blocking alone buys; the rows after it add the threads.
// GFLOP/s counts 2*n^3 floating point operations. "vs scalar" and "vs seq" divide that baseline's
// median time by this row's. Kernels are called directly into preallocated matrices: no
// dispatcher, no allocation. Built on the shared harness: see benchmarks/bench.hpp and
// docs/BENCHMARKING.md.
#include "bench.hpp"

#include "backends/cpu/matmul.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/scalar.hpp"
#include "backends/cpu/sequential.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace seq = mensurflow::cpu::sequential;
namespace scalar = mensurflow::cpu::scalar;

int main() {
    const int max = cpu::max_threads();
    const std::vector<int> counts = bench::thread_counts(max);
    const bench::Config config;

    bench::print_header(
        "Matrix multiplication C = A * B, square float32 matrices", config,
        {"impl: scalar = plain i-k-j loop, no SIMD; sequential = the same loop with "
         "the compiler's SIMD; parallel = blocked kernel with N threads",
         "vs scalar / vs seq = that baseline's median time / this row's median time"});
    bench::Table table("GFLOP/s");
    table.section("matmul");

    for (std::int64_t n : {64, 128, 256, 512, 1024}) {
        const auto len = static_cast<std::size_t>(n * n);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        bench::Arrays arrays(len, 4);
        float* a = arrays[0];
        float* b = arrays[1];
        float* c = arrays[2];
        float* reference = arrays[3];
        bench::fill_a(a, len);
        bench::fill_b(b, len);

        // The sequential result is the reference. Each C entry sums its products in the same
        // order in every implementation, so they agree up to rounding differences between
        // compilers (fused multiply-adds), hence a tolerance instead of exact equality.
        auto run_sequential = [&] { seq::matmul_f32(a, b, reference, n, n, n); };
        auto run_scalar = [&] { scalar::matmul_f32(a, b, c, n, n, n); };
        run_sequential();
        run_scalar();
        bench::check_close(reference, c, len, 1e-4, "scalar matmul n=" + std::to_string(n));

        const bench::Sample scalar_sample =
            bench::measure(config, run_scalar, [&] { return bench::checksum(c, len); });
        const bench::Sample seq_sample =
            bench::measure(config, run_sequential, [&] { return bench::checksum(reference, len); });
        const std::string label = std::to_string(n);
        table.row(label, "scalar", "-", scalar_sample, flops, &scalar_sample, &seq_sample);
        table.row(label, "sequential", "-", seq_sample, flops, &scalar_sample, &seq_sample);

        for (int threads : counts) {
            cpu::set_max_threads(threads);
            auto run_parallel = [&] { cpu::matmul_f32(a, b, c, n, n, n); };
            run_parallel();
            bench::check_close(reference, c, len, 1e-4,
                               "parallel matmul n=" + std::to_string(n) +
                                   " threads=" + std::to_string(threads));
            const bench::Sample par =
                bench::measure(config, run_parallel, [&] { return bench::checksum(c, len); });
            table.row(label, "parallel", std::to_string(threads), par, flops, &scalar_sample,
                      &seq_sample);
        }
        table.blank();
    }
    cpu::set_max_threads(max);
    return 0;
}
