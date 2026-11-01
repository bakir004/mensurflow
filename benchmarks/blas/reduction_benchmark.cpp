#include "bench.hpp"

#include "backends/cpu/parallel.hpp"
#include "backends/cpu/reduction.hpp"
#include "backends/cpu/sequential_reduction.hpp"


#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace sequential = mensurflow::cpu::sequential;

namespace {

struct Reduction {
    const char* name;
    float (*parallel)(const float*, std::int64_t);
    float (*serial)(const float*, std::int64_t);
};

} // namespace

int main() {
    const bench::Config config;
    const Reduction reductions[] = {
        {"sum", cpu::sum_f32, sequential::sum_f32},
        {"mean", cpu::mean_f32, sequential::mean_f32},
        {"max", cpu::max_f32, sequential::max_f32},
    };
    const std::int64_t sizes[] = {1, 16, 32768, 32769, 100000, 1000003, 10000000};

    bench::print_header(
        "CPU full-tensor reductions (float32)", config,
        {"direct kernels only; values are validated before timing",
         "rate is elements per second; parallel uses the configured OpenMP thread count"});

    bench::Table table("elements/s", 1e9, "vs sequential", "");
    for (const Reduction& reduction : reductions) {
        table.section(reduction.name);
        for (std::int64_t n : sizes) {
            std::vector<float> values(static_cast<std::size_t>(n));
            bench::fill_a(values.data(), values.size());
            const float expected = reduction.serial(values.data(), n);
            float result = 0.0f;
            auto run_serial = [&] { result = reduction.serial(values.data(), n); };
            auto run_parallel = [&] { result = reduction.parallel(values.data(), n); };
            run_parallel();
            bench::check_value(expected, result, 2e-4, std::string(reduction.name) + " parallel",
                               std::abs(static_cast<double>(expected)) + 1.0);
            const bench::Sample serial =
                bench::measure(config, run_serial, [&] { return static_cast<double>(result); });
            const bench::Sample parallel =
                bench::measure(config, run_parallel, [&] { return static_cast<double>(result); });
            const std::string label = std::to_string(n);
            table.row(label, "parallel", std::to_string(cpu::max_threads()), parallel,
                      static_cast<double>(n), &serial);
            table.row(label, "sequential", "-", serial, static_cast<double>(n), &serial);
            table.blank();
        }
    }
}
