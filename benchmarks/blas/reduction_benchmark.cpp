#include "bench.hpp"

#include "backends/cpu/ops/reduction/reduction.hpp"
#include "backends/cpu/threads.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct Reduction {
    const char* name;
    float (*parallel)(const float*, std::int64_t);
    float (*serial)(const float*, std::int64_t); // the same function when it parallelizes itself
};

} // namespace

int main() {
    const bench::Config config;
    const Reduction reductions[] = {
        {"sum", cpu::sum_f32, cpu::sum_f32},
        {"mean", cpu::mean_f32, cpu::mean_f32},
        {"max", cpu::max_f32_parallel, cpu::max_f32_sequential},
    };
    const std::int64_t sizes[] = {1, 16, 32768, 32769, 100000, 1000003, 10000000};

    const int configured_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(configured_threads);
    bench::print_header(
        "CPU full-tensor reductions (float32)", config,
        {"direct kernels only; values are validated before timing",
         "rate is elements per second; sequential = one thread (sum, mean) or the plain "
         "loop (max)",
         "thread-count sweep uses the configured maximum of " +
             std::to_string(configured_threads)});

    bench::Table table("elements/s", 1e9, "vs sequential", "");
    for (const Reduction& reduction : reductions) {
        table.section(reduction.name);
        for (std::int64_t n : sizes) {
            std::vector<float> values(static_cast<std::size_t>(n));
            bench::fill_a(values.data(), values.size());
            cpu::set_max_threads(1);
            const float expected = reduction.serial(values.data(), n);
            float result = 0.0f;
            auto run_serial = [&] { result = reduction.serial(values.data(), n); };
            auto run_parallel = [&] { result = reduction.parallel(values.data(), n); };
            const bench::Sample serial =
                bench::measure(config, run_serial, [&] { return static_cast<double>(result); });
            const std::string label = std::to_string(n);
            table.row(label, "sequential", "-", serial, static_cast<double>(n), &serial);
            const bool same_kernel = reduction.serial == reduction.parallel;
            for (int threads : thread_counts) {
                if (same_kernel && threads == 1) {
                    continue; // that is the sequential row
                }
                cpu::set_max_threads(threads);
                run_parallel();
                bench::check_value(expected, result, 2e-4,
                                   std::string(reduction.name) + " parallel (" +
                                       std::to_string(threads) + " threads)",
                                   std::abs(static_cast<double>(expected)) + 1.0);
                const float first = result;
                run_parallel();
                bench::check_value(first, result, 2e-4,
                                   std::string(reduction.name) + " repeat (" +
                                       std::to_string(threads) + " threads)",
                                   std::abs(static_cast<double>(expected)) + 1.0);
                const bench::Sample parallel = bench::measure(
                    config, run_parallel, [&] { return static_cast<double>(result); });
                table.row(label, "parallel", std::to_string(threads), parallel,
                          static_cast<double>(n), &serial);
            }
            table.blank();
        }
    }
    cpu::set_max_threads(configured_threads);
}
