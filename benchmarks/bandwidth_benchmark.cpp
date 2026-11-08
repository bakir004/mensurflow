#include "bench.hpp"
#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"

#include <cstdint>
#include <iostream>

namespace {

using namespace mensurflow::cpu;

} // namespace

int main() {
    constexpr std::int64_t n = 64 * 1024 * 1024;
    bench::Config config;
    config.min_sample_s = 0.1;
    bench::print_header("Memory bandwidth: float32 copy", config,
                        {"64M floats; source and destination exceed typical L3 cache"});
    bench::Table table("copy");
    bench::Arrays arrays(static_cast<std::size_t>(n), 2);
    bench::fill_a(arrays[0], static_cast<std::size_t>(n));
    for (int threads : {1, max_threads()}) {
        set_max_threads(threads);
        const auto sample = bench::measure(
            config, [&] { copy_f32(arrays[0], arrays[1], n); },
            [&] { return bench::checksum(arrays[1], static_cast<std::size_t>(n)); });
        table.row("copy", "memcpy-like", std::to_string(threads), sample,
                  static_cast<double>(n) * sizeof(float) * 2.0);
    }
    return 0;
}
