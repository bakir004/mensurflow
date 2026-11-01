// Last-axis softmax: sequential versus row-parallel CPU kernels.
#include "bench.hpp"

#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential_softmax.hpp"
#include "backends/cpu/softmax.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace sequential = mensurflow::cpu::sequential;

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header(
        "CPU last-axis softmax (float32)", config,
        {"parallel splits rows across the configured OpenMP thread count",
         "rate is elements processed per second"});
    bench::Table table("elements/s", 1e9);
    table.section("softmax");

    for (const std::int64_t rows : {64, 256, 1024, 4096}) {
        const std::int64_t width = 1024;
        const std::size_t length = static_cast<std::size_t>(rows * width);
        bench::Arrays arrays(length, 3);
        float* input = arrays[0];
        float* output = arrays[1];
        float* expected = arrays[2];
        bench::fill_a(input, length);
        auto run_sequential = [&] {
            sequential::softmax_f32(input, expected, rows, width);
        };
        auto run_parallel = [&] {
            cpu::softmax_f32(input, output, rows, width);
        };
        run_sequential();
        run_parallel();
        bench::check_close(expected, output, length, 1e-5,
                           "softmax rows=" + std::to_string(rows));
        const bench::Sample serial =
            bench::measure(config, run_sequential, [&] { return bench::checksum(expected, length); });
        table.row(std::to_string(rows) + "x" + std::to_string(width), "sequential", "-", serial,
                  static_cast<double>(length), &serial);
        for (int threads : thread_counts) {
            cpu::set_max_threads(threads);
            run_parallel();
            bench::check_close(expected, output, length, 1e-5,
                               "softmax threads=" + std::to_string(threads));
            const bench::Sample parallel =
                bench::measure(config, run_parallel, [&] { return bench::checksum(output, length); });
            table.row(std::to_string(rows) + "x" + std::to_string(width), "parallel",
                      std::to_string(threads), parallel, static_cast<double>(length), &serial);
        }
        table.blank();
    }
    cpu::set_max_threads(maximum_threads);
}
