// Matrix multiplication against OpenBLAS: the reference our own kernel has to approach.
//   sequential: matmul_f32_sequential, the plain i-k-j loop (the speedup baseline).
//   parallel:   matmul_f32_parallel with N threads.
//   openblas:   cblas_sgemm with N threads (openblas_set_num_threads).
// Both libraries run at the same thread counts. "vs seq" divides the sequential median time by this
// row's; "vs obl" divides the OpenBLAS median time (same n, same threads) by this row's, so 1.00
// means as fast as OpenBLAS and 0.25 means four times slower. GFLOP/s counts 2*n^3. Kernels are
// called directly into preallocated matrices: no dispatcher, no allocation.
//
//   matmul_openblas_benchmark [n ...]    default 64 128 256 512 1024 2048
// From n = 4096 the sequential and old blocked kernels are left out (too slow) and OpenBLAS is the
// reference for the correctness checks.
//
// Needs OpenBLAS at build time: see docs/BENCHMARKING.md ("Comparing against OpenBLAS").
#include "bench.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"

#include <cblas.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

// Both OpenBLAS's worker threads and OpenMP's keep spinning for a while after a parallel call, and
// on an SMT CPU they take cores and execution units from whatever is timed next (our packed kernel
// read 376 instead of 620 GFLOP/s at n = 1024, 8 threads). Pause before every measurement so each
// library starts from idle threads.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

} // namespace

int main(int argc, char** argv) {
    // Sizes can be given on the command line (default 64 to 2048; 4096 and above skip the old kernels).
    std::vector<std::int64_t> sizes;
    for (int i = 1; i < argc; ++i) {
        sizes.push_back(std::atoll(argv[i]));
    }
    if (sizes.empty()) {
        sizes = {64, 128, 256, 512, 1024, 2048};
    }
    const int max = cpu::max_threads();
    const std::vector<int> counts = bench::thread_counts(max);
    const bench::Config config;

    bench::print_header(
        "Matrix multiplication C = A * B vs OpenBLAS, square float32 matrices", config,
        {std::string("OpenBLAS: ") + openblas_get_config() + ", core " + openblas_get_corename(),
         "packed blocking (mc, kc, nc, tiles/thread): " +
             std::to_string(cpu::packed_gemm_blocking().mc) + ", " +
             std::to_string(cpu::packed_gemm_blocking().kc) + ", " +
             std::to_string(cpu::packed_gemm_blocking().nc) + ", " +
             std::to_string(cpu::packed_gemm_blocking().tiles_per_thread),
         "impl: sequential = plain i-k-j loop; parallel = our blocked kernel; openblas = "
         "cblas_sgemm",
         "vs seq = sequential median time / this row's; vs obl = OpenBLAS median time at the same "
         "n and threads / this row's (1.00 = as fast as OpenBLAS)"});
    bench::Table table("GFLOP/s", 1e9, "vs seq", "vs obl");
    table.section("matmul vs openblas");

    for (std::int64_t n : sizes) {
        // From 4096 up the old kernels are left out: the sequential loop takes about 6 s per call at
        // 4096 and the old blocked kernel on one thread about 25 s. OpenBLAS's own result is then the
        // reference for the correctness checks, and the "vs seq" column is empty.
        const bool big = n >= 4096;
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
        const std::string label = std::to_string(n);
        const int ni = static_cast<int>(n);

        bench::Sample seq_sample;
        const bench::Sample* seq = nullptr; // baseline for the "vs seq" column
        if (!big) {
            auto run_sequential = [&] { cpu::matmul_f32_sequential(a, b, reference, n, n, n); };
            run_sequential();
            seq_sample = bench::measure(config, run_sequential,
                                        [&] { return bench::checksum(reference, len); });
            seq = &seq_sample;
            table.row(label, "sequential", "-", seq_sample, flops, seq);
        } else {
            openblas_set_num_threads(max);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, ni, ni, 1.0f, a, ni, b, ni,
                        0.0f, reference, ni);
        }

        // Single-thread packed AVX2 kernel (gemm_packed.cpp) against single-thread OpenBLAS.
        {
            openblas_set_num_threads(1);
            auto run_obl1 = [&] {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, ni, ni, 1.0f, a, ni, b,
                            ni, 0.0f, c, ni);
            };
            settle();
            run_obl1();
            const bench::Sample obl1 =
                bench::measure(config, run_obl1, [&] { return bench::checksum(c, len); });
            auto run_packed = [&] { cpu::matmul_f32_packed(a, b, c, n, n, n); };
            settle();
            run_packed();
            bench::check_close(reference, c, len, 1e-3, "packed matmul n=" + label);
            const bench::Sample packed =
                bench::measure(config, run_packed, [&] { return bench::checksum(c, len); });
            table.row(label, "packed", "1", packed, flops, seq, &obl1);
            table.row(label, "openblas", "1", obl1, flops, seq);
        }

        for (int threads : counts) {
            cpu::set_max_threads(threads);
            openblas_set_num_threads(threads);

            auto run_openblas = [&] {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, ni, ni, 1.0f, a, ni, b,
                            ni, 0.0f, c, ni);
            };
            settle();
            run_openblas();
            bench::check_close(reference, c, len, 1e-3,
                               "openblas n=" + label + " threads=" + std::to_string(threads));
            const bench::Sample obl =
                bench::measure(config, run_openblas, [&] { return bench::checksum(c, len); });

            bench::Sample par;
            if (!big) {
                auto run_parallel = [&] { cpu::matmul_f32_parallel(a, b, c, n, n, n); };
                settle();
                run_parallel();
                bench::check_close(reference, c, len, 1e-3,
                                   "parallel matmul n=" + label +
                                       " threads=" + std::to_string(threads));
                par = bench::measure(config, run_parallel, [&] { return bench::checksum(c, len); });
            }

            auto run_packed_parallel = [&] { cpu::matmul_f32_packed_parallel(a, b, c, n, n, n); };
            settle();
            run_packed_parallel();
            bench::check_close(reference, c, len, 1e-3,
                               "packed parallel n=" + label +
                                   " threads=" + std::to_string(threads));
            const bench::Sample packed_par = bench::measure(
                config, run_packed_parallel, [&] { return bench::checksum(c, len); });

            if (!big) {
                table.row(label, "parallel", std::to_string(threads), par, flops, seq, &obl);
            }
            table.row(label, "packed", std::to_string(threads), packed_par, flops, seq, &obl);
            table.row(label, "openblas", std::to_string(threads), obl, flops, seq);
        }
        table.blank();
    }
    cpu::set_max_threads(max);
    return 0;
}
