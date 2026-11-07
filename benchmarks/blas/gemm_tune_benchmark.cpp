// Cache-blocking sweep for the packed GEMM (gemm_packed.cpp). Times square float32 products for every
// combination of the listed block sizes, then re-times the best few and the default alternately
// (so a quiet or noisy moment does not pick the winner) and prints them next to the CPU's cache sizes.
//
//   gemm_tune_benchmark [key=v1,v2,...] ...
//     threads=   thread counts            (default 1,<max>)
//     sizes=     square matrix sizes      (default 512,1024,2048)
//     mc=        rows of A kept in L2     (default: the current default)
//     kc=        depth step               (default: the current default)
//     nc=        columns of B kept in L3  (default: the current default)
//     tiles=     tiles of C per thread, threaded version only (default: the current default)
//     prefetch=  6x16 kernel only: 0 none, 1 C tile, 2 C tile + A/B ahead (default 0)
//     shape=     micro-kernel tile: 0 6x16, 1 6x16 unrolled x4, 2 4x24, 3 12x8 (default 0)
//     pack=      1 = scalar packers instead of the AVX2 ones (default 0)
//   e.g. gemm_tune_benchmark threads=1,12 sizes=2048 mc=240,336,480 kc=384,512,768
//
// mc and nc are rounded down to whole tiles. Not pass/fail: it only tells you which constants to put
// in gemm_packed.cpp.
#include "bench.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct Result {
    cpu::PackedGemmBlocking blocking;
    double gflops = 0.0;
};

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::atoll(item.c_str()));
        }
    }
    return values;
}

} // namespace

int main(int argc, char** argv) {
    const int max = cpu::max_threads();
    const cpu::PackedGemmBlocking original = cpu::packed_gemm_blocking();

    std::map<std::string, std::vector<std::int64_t>> options = {
        {"threads", {1, max}}, {"sizes", {512, 1024, 2048}}, {"mc", {original.mc}},
        {"kc", {original.kc}}, {"nc", {original.nc}},        {"tiles", {original.tiles_per_thread}},
        {"prefetch", {0}},     {"shape", {original.shape}},  {"pack", {0}},
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq == std::string::npos || options.find(arg.substr(0, eq)) == options.end()) {
            std::fprintf(stderr, "unknown argument '%s' (see the comment at the top of the file)\n",
                         argv[i]);
            return 2;
        }
        options[arg.substr(0, eq)] = parse_list(arg.substr(eq + 1));
    }

    bench::Config sweep_config;
    sweep_config.repeats = 5;
    bench::Config confirm_config;
    confirm_config.repeats = 9;
    const cpu::CacheSizes caches = cpu::cpu_cache_sizes();
    bench::print_header(
        "Packed GEMM cache-blocking sweep, square float32 matrices", sweep_config,
        {"caches from CPUID: L1d " + std::to_string(caches.l1d / 1024) + " KB, L2 " +
             std::to_string(caches.l2 / 1024) + " KB (per core), L3 " +
             std::to_string(caches.l3 / 1024 / 1024) + " MB (shared)",
         "sweep: 5 samples per combination; the best few are then re-timed alternately with the "
         "default, 3 rounds, median reported"});

    for (std::int64_t threads : options["threads"]) {
        if (threads < 1 || threads > max) {
            continue;
        }
        cpu::set_max_threads(static_cast<int>(threads));
        for (std::int64_t n : options["sizes"]) {
            const auto len = static_cast<std::size_t>(n * n);
            const double flops =
                2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
            bench::Arrays arrays(len, 3);
            float* a = arrays[0];
            float* b = arrays[1];
            float* c = arrays[2];
            bench::fill_a(a, len);
            bench::fill_b(b, len);

            auto time_it = [&](const cpu::PackedGemmBlocking& blocking, const bench::Config& cfg) {
                cpu::set_packed_gemm_blocking(blocking);
                auto run = [&] {
                    if (threads == 1) {
                        cpu::matmul_f32_packed(a, b, c, n, n, n);
                    } else {
                        cpu::matmul_f32_packed_parallel(a, b, c, n, n, n);
                    }
                };
                const bench::Sample sample =
                    bench::measure(cfg, run, [&] { return bench::checksum(c, len); });
                return flops / sample.median_s / 1e9;
            };

            std::vector<Result> results;
            for (std::int64_t pack : options["pack"]) {
                for (std::int64_t shape : options["shape"]) {
                    for (std::int64_t prefetch : options["prefetch"]) {
                        for (std::int64_t t : options["tiles"]) {
                            if (threads == 1 && t != options["tiles"].front()) {
                                continue; // tile granularity does nothing without threads
                            }
                            for (std::int64_t nc : options["nc"]) {
                                for (std::int64_t kc : options["kc"]) {
                                    for (std::int64_t mc : options["mc"]) {
                                        cpu::PackedGemmBlocking blocking;
                                        blocking.mc = mc;
                                        blocking.kc = kc;
                                        blocking.nc = nc;
                                        blocking.tiles_per_thread = t;
                                        blocking.prefetch = prefetch;
                                        blocking.shape = shape;
                                        blocking.scalar_pack = pack;
                                        results.push_back(
                                            {blocking, time_it(blocking, sweep_config)});
                                    }
                                }
                            }
                        }
                    }
                }
            }
            std::sort(results.begin(), results.end(),
                      [](const Result& x, const Result& y) { return x.gflops > y.gflops; });

            // Re-time the top 5 and the default alternately, three rounds each, and keep medians.
            std::vector<Result> finalists(
                results.begin(), results.begin() + std::min<std::size_t>(5, results.size()));
            finalists.push_back({original, 0.0});
            std::vector<std::vector<double>> rounds(finalists.size());
            for (int round = 0; round < 3; ++round) {
                for (std::size_t i = 0; i < finalists.size(); ++i) {
                    rounds[i].push_back(time_it(finalists[i].blocking, confirm_config));
                }
            }
            for (std::size_t i = 0; i < finalists.size(); ++i) {
                std::sort(rounds[i].begin(), rounds[i].end());
                finalists[i].gflops = rounds[i][1];
            }
            const double default_gflops = finalists.back().gflops;
            finalists.pop_back();
            std::sort(finalists.begin(), finalists.end(),
                      [](const Result& x, const Result& y) { return x.gflops > y.gflops; });

            std::printf(
                "\n== n=%lld, %lld thread%s: default (mc %lld, kc %lld, nc %lld, tiles %lld, "
                "prefetch %lld, shape %lld, scalar_pack %lld) = %.1f GFLOP/s ==\n",
                static_cast<long long>(n), static_cast<long long>(threads), threads == 1 ? "" : "s",
                static_cast<long long>(original.mc), static_cast<long long>(original.kc),
                static_cast<long long>(original.nc),
                static_cast<long long>(original.tiles_per_thread),
                static_cast<long long>(original.prefetch), static_cast<long long>(original.shape),
                static_cast<long long>(original.scalar_pack), default_gflops);
            std::printf("   confirmed best:    mc   kc    nc tiles pf\n");
            for (const Result& r : finalists) {
                std::printf(
                    "                   %4lld %4lld %5lld %5lld %2lld %5lld %6lld   %8.1f GFLOP/s  "
                    "(%.2fx default)\n",
                    static_cast<long long>(r.blocking.mc), static_cast<long long>(r.blocking.kc),
                    static_cast<long long>(r.blocking.nc),
                    static_cast<long long>(r.blocking.tiles_per_thread),
                    static_cast<long long>(r.blocking.prefetch),
                    static_cast<long long>(r.blocking.shape),
                    static_cast<long long>(r.blocking.scalar_pack), r.gflops,
                    r.gflops / default_gflops);
            }
            std::fflush(stdout);
        }
    }
    cpu::set_packed_gemm_blocking(original);
    cpu::set_max_threads(max);
    return 0;
}
