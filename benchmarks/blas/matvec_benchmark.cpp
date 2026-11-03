// Matrix-vector multiplication: sequential versus the row-parallel CPU kernel.
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
    bench::print_header("CPU matrix-vector multiplication, float32", config,
                        {"sequential = one thread; parallel splits rows across N OpenMP threads",
                         "rate is GFLOP/s for 2*M*K floating-point operations"});
    bench::Table table("GFLOP/s");
    table.section("matvec");

    for (const std::int64_t m : {64, 256, 1024, 4096}) {
        const std::int64_t k = 1024;
        const auto matrix_len = static_cast<std::size_t>(m * k);
        bench::Arrays arrays(matrix_len + static_cast<std::size_t>(k) + static_cast<std::size_t>(m),
                             3);
        float* a = arrays[0];
        float* x = a + matrix_len;
        float* y = x + k;
        bench::fill_a(a, matrix_len);
        bench::fill_b(x, static_cast<std::size_t>(k));
        const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(k);
        auto run = [&] { cpu::matvec_f32(a, x, y, m, k); };
        cpu::set_max_threads(1);
        run();
        std::vector<float> expected(y, y + m);
        const bench::Sample sequential = bench::measure(
            config, run, [&] { return bench::checksum(y, static_cast<std::size_t>(m)); });
        table.row(std::to_string(m) + "x" + std::to_string(k), "sequential", "-", sequential,
                  flops);
        for (int threads : counts) {
            if (threads == 1) {
                continue; // that is the sequential row
            }
            cpu::set_max_threads(threads);
            run();
            bench::check_close(expected.data(), y, static_cast<std::size_t>(m), 1e-5,
                               "matvec threads=" + std::to_string(threads));
            const bench::Sample parallel = bench::measure(
                config, run, [&] { return bench::checksum(y, static_cast<std::size_t>(m)); });
            table.row(std::to_string(m) + "x" + std::to_string(k), "parallel",
                      std::to_string(threads), parallel, flops, &sequential);
        }
        table.blank();
    }
    cpu::set_max_threads(max);
    return 0;
}
