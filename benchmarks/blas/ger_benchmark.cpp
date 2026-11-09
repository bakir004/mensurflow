// Rank-1 update out = alpha * x * y^T + a: sequential versus the row-parallel CPU kernel.
// It is memory-bound (one multiply-add per element of a read and of out written), so the rate is
// GB/s of a read plus out written. "vs seq" divides the sequential median time by this row's.
//
//   ger_benchmark
#include "bench.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header("CPU rank-1 update (ger), float32", config,
                        {"sequential = one thread; parallel splits rows across N OpenMP threads",
                         "rate is GB/s: a read once and out written once"});
    bench::Table table("GB/s", 1e9, "vs seq", "");
    table.section("ger, square [n, n]");

    for (const std::int64_t n : {32, 181, 1024, 4096}) {
        const auto len = static_cast<std::size_t>(n * n);
        bench::Arrays arrays(len, 4);
        float* x = arrays[0];
        float* y = arrays[1];
        float* a = arrays[2];
        float* out = arrays[3];
        bench::fill_a(x, static_cast<std::size_t>(n));
        bench::fill_b(y, static_cast<std::size_t>(n));
        bench::fill_a(a, len);
        const double bytes = 8.0 * static_cast<double>(len);
        auto run = [&] { cpu::ger_f32(0.5f, x, y, a, out, n, n); };
        auto observe = [&] { return bench::checksum(out, len); };
        const std::string label = std::to_string(n) + "x" + std::to_string(n);

        cpu::set_max_threads(1);
        run();
        const std::vector<float> expected(out, out + len);
        const bench::Sample sequential = bench::measure(config, run, observe);
        table.row(label, "sequential", "1", sequential, bytes, &sequential);
        for (int threads : thread_counts) {
            if (threads == 1) {
                continue; // that is the sequential row
            }
            cpu::set_max_threads(threads);
            run();
            bench::check_bits(expected.data(), out, len,
                              "ger " + label + " threads=" + std::to_string(threads));
            const bench::Sample parallel = bench::measure(config, run, observe);
            table.row(label, "parallel", std::to_string(threads), parallel, bytes, &sequential);
        }
        table.blank();
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}
