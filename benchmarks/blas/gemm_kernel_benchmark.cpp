// The micro-kernel alone, one thread, no packing: every 6 x 16 variant (and the other tile shapes) on
// packed panels that stay in cache, so differences in the loop body (unrolling, where the broadcasts
// go) are not hidden by packing, cache blocking or threads.
//
//   L1   kc = 256, one A panel and one B panel (6 KB + 16 KB, both in the 32 KB L1), the same tile
//        accumulated over and over
//   L2   kc = 768, one A panel (18 KB, L1) swept over 8 B panels (8 x 48 KB, L2): the loop order of
//        the real kernel (A panel held, B streaming)
//
// GFLOP/s = 2 * mr * nr * kc per call; the median over rounds in which the variants alternate. The %
// column is against the measured FMA peak of one core (run fma_peak_benchmark; 112 GFLOP/s on the
// Ryzen 9 5900X under AVX2).
//
//   gemm_kernel_benchmark [shapes=0,1,...,11] [rounds=7] [peak=112]
#include "bench.hpp"

#include "backends/cpu/affinity.hpp"
#include "backends/cpu/ops/blas/gemm_packed.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::atoi(item.c_str()));
        }
    }
    return values;
}

const char* shape_name(int shape) {
    static const char* names[] = {"6x16 (default)",       "6x16 unrolled x4 (old)",
                                  "4x24",                 "12x8",
                                  "6x16 reuse x2",        "6x16 reuse x4",
                                  "6x16 pipelined",       "6x16 pipelined x2",
                                  "6x16 reuse x2 fence",  "6x16 reuse x4 fence",
                                  "6x16 pipe x2 fence",   "6x16 pipe x4 fence"};
    return shape >= 0 && shape < 12 ? names[shape] : "?";
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> shapes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int rounds = 7;
    double peak = 112.0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("shapes=", 0) == 0) {
            shapes = parse_list(arg.substr(7));
        } else if (arg.rfind("rounds=", 0) == 0) {
            rounds = std::atoi(arg.substr(7).c_str());
        } else if (arg.rfind("peak=", 0) == 0) {
            peak = std::atof(arg.substr(5).c_str());
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (!cpu::packed_gemm_supported()) {
        std::printf("no AVX2/FMA on this CPU: nothing to measure\n");
        return 0;
    }
    cpu::set_thread_pin(0, 1); // one core, so the OS cannot move the thread mid-measurement

    bench::Config config;
    bench::print_header("Micro-kernel alone (one thread, panels in cache)", config,
                        {"GFLOP/s are medians over " + std::to_string(rounds) +
                         " rounds; % of a measured core peak of " + std::to_string(peak) +
                         " GFLOP/s"});

    struct Mode {
        const char* name;
        std::int64_t kc;
        std::int64_t b_panels;
    };
    const Mode modes[] = {{"L1 (kc 256, 1 B panel)", 256, 1}, {"L2 (kc 768, 8 B panels)", 768, 8}};

    for (const Mode& mode : modes) {
        const std::int64_t max_floats = mode.kc * 24; // the widest panel: 24 floats per step
        std::vector<float> storage(static_cast<std::size_t>(max_floats * (mode.b_panels + 1) + 64));
        // Panels start on 64-byte boundaries, as the GEMM's packing buffers do.
        const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        float* pa = storage.data() + ((64 - address % 64) % 64) / sizeof(float);
        float* pb = pa + ((max_floats + 15) / 16) * 16;
        for (std::int64_t i = 0; i < max_floats * (mode.b_panels + 1); ++i) {
            pa[i] = 0.001f * static_cast<float>(i % 97);
        }
        alignas(64) float c[12 * 24] = {};

        std::vector<std::vector<double>> rates(shapes.size());
        for (int round = 0; round < rounds; ++round) {
            for (std::size_t v = 0; v < shapes.size(); ++v) {
                std::int64_t mr = 0;
                std::int64_t nr = 0;
                if (!cpu::packed_microkernel_tile(shapes[v], mr, nr)) {
                    continue;
                }
                const std::int64_t panel = mode.kc * nr;
                const auto sweep = [&] {
                    for (std::int64_t j = 0; j < mode.b_panels; ++j) {
                        cpu::run_packed_microkernel(shapes[v], mode.kc, pa, pb + j * panel, c, nr,
                                                    true);
                    }
                };
                for (int w = 0; w < 200; ++w) {
                    sweep();
                }
                // Enough sweeps for about 30 ms.
                const std::int64_t sweeps =
                    std::max<std::int64_t>(1, static_cast<std::int64_t>(
                                                  30e-3 * peak * 1e9 /
                                                  (2.0 * mr * nr * mode.kc * mode.b_panels)));
                const auto t0 = std::chrono::steady_clock::now();
                for (std::int64_t s = 0; s < sweeps; ++s) {
                    sweep();
                }
                const double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                rates[v].push_back(2.0 * static_cast<double>(mr * nr * mode.kc * mode.b_panels) *
                                   static_cast<double>(sweeps) / seconds / 1e9);
            }
        }
        bench::consume(bench::checksum(c, 12 * 24));

        std::printf("\n== %s ==\n%-24s %10s %10s %8s\n", mode.name, "kernel", "median", "best",
                    "% peak");
        for (std::size_t v = 0; v < shapes.size(); ++v) {
            if (rates[v].empty()) {
                continue;
            }
            const double med = median_of(rates[v]);
            const double best = *std::max_element(rates[v].begin(), rates[v].end());
            std::printf("%-24s %10.1f %10.1f %7.1f%%\n", shape_name(shapes[v]), med, best,
                        100.0 * med / peak);
        }
    }
    cpu::set_thread_pin(0, 0);
    return 0;
}
