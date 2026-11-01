// Last-axis argmax: sequential versus row-parallel CPU kernels.
#include "bench.hpp"

#include "backends/cpu/argmax.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential_argmax.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace sequential = mensurflow::cpu::sequential;

double checksum(const std::int64_t* values, std::size_t n) {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total = total * 1315423911 + values[i];
    }
    return static_cast<double>(total);
}

void check_equal(const std::int64_t* expected, const std::int64_t* actual, std::size_t n,
                 const std::string& label) {
    for (std::size_t i = 0; i < n; ++i) {
        if (expected[i] != actual[i]) {
            throw std::runtime_error(label + ": result mismatch");
        }
    }
}

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header(
        "CPU last-axis argmax (float32 input, int64 output)", config,
        {"parallel splits rows across the configured OpenMP thread count",
         "rate is rows processed per second"});
    bench::Table table("M rows/s", 1e6);
    table.section("argmax");

    for (const std::int64_t rows : {1024, 4096, 16384}) {
        const std::int64_t width = 1024;
        const std::size_t input_length = static_cast<std::size_t>(rows * width);
        const std::size_t output_length = static_cast<std::size_t>(rows);
        bench::Arrays arrays(input_length, 1);
        float* input = arrays[0];
        std::vector<std::int64_t> output(output_length);
        std::vector<std::int64_t> expected(output_length);
        bench::fill_a(input, input_length);
        auto run_sequential = [&] {
            sequential::argmax_f32(input, expected.data(), rows, width);
        };
        auto run_parallel = [&] {
            cpu::argmax_f32(input, output.data(), rows, width);
        };
        run_sequential();
        run_parallel();
        check_equal(expected.data(), output.data(), output_length,
                    "argmax rows=" + std::to_string(rows));
        const bench::Sample serial =
            bench::measure(config, run_sequential,
                           [&] { return checksum(expected.data(), output_length); });
        const std::string label = std::to_string(rows) + "x" + std::to_string(width);
        table.row(label, "sequential", "-", serial, static_cast<double>(rows), &serial);
        for (int threads : thread_counts) {
            cpu::set_max_threads(threads);
            run_parallel();
            check_equal(expected.data(), output.data(), output_length,
                        "argmax threads=" + std::to_string(threads));
            const bench::Sample parallel =
                bench::measure(config, run_parallel,
                               [&] { return checksum(output.data(), output_length); });
            table.row(label, "parallel", std::to_string(threads), parallel,
                      static_cast<double>(rows), &serial);
        }
        table.blank();
    }
    cpu::set_max_threads(maximum_threads);
}
