// Matrix multiplication, sequential vs parallel, for square matrices.
//   sequential: matmul_f32_sequential, the plain i-k-j loop with whatever SIMD the compiler finds.
//   parallel:   matmul_f32_parallel, the cache-blocked loop with 1, 2, 4, ... threads. The
//               1-thread row isolates what cache blocking alone buys; the rows after it add the
//               threads.
// GFLOP/s counts 2*n^3 floating point operations. "vs seq" divides the sequential median time by
// this row's. Kernels are called directly into preallocated matrices: no
// dispatcher, no allocation. Built on the shared harness: see benchmarks/bench.hpp and
// docs/BENCHMARKING.md.
#include "bench.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;

int main() {
    const int max = cpu::max_threads();
    const std::vector<int> counts = bench::thread_counts(max);
    const bench::Config config;

    bench::print_header(
        "Matrix multiplication C = A * B, square float32 matrices", config,
        {"impl: sequential = plain i-k-j loop with the compiler's SIMD; parallel = blocked "
         "kernel with N threads",
         "vs seq = the sequential median time / this row's median time"});
    bench::Table table("GFLOP/s", 1e9, "vs seq", "");
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
        // order in both implementations, so they agree up to rounding differences between
        // compilers (fused multiply-adds), hence a tolerance instead of exact equality.
        auto run_sequential = [&] { cpu::matmul_f32_sequential(a, b, reference, n, n, n); };
        run_sequential();
        const bench::Sample seq_sample =
            bench::measure(config, run_sequential, [&] { return bench::checksum(reference, len); });
        const std::string label = std::to_string(n);
        table.row(label, "sequential", "-", seq_sample, flops, &seq_sample);

        for (int threads : counts) {
            cpu::set_max_threads(threads);
            auto run_parallel = [&] { cpu::matmul_f32_parallel(a, b, c, n, n, n); };
            run_parallel();
            bench::check_close(reference, c, len, 1e-4,
                               "parallel matmul n=" + std::to_string(n) +
                                   " threads=" + std::to_string(threads));
            const bench::Sample par =
                bench::measure(config, run_parallel, [&] { return bench::checksum(c, len); });
            table.row(label, "parallel", std::to_string(threads), par, flops, &seq_sample);
        }
        table.blank();
    }
    cpu::set_max_threads(max);
    return 0;
}
