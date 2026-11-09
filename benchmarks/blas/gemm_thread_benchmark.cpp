// How the threaded packed GEMM spends its time, and which way of parallelizing the loops is fastest.
// Times square float32 products for every combination of the listed settings, in several rounds with
// the combinations alternating inside each round (the median over the rounds is shown, so one noisy
// moment cannot pick a winner). Three ways of calling, because they behave differently:
//
//   median  the product repeated with a checksum of C between the calls (bench::measure)
//   burst   5 calls in a row with nothing in between (a loop that multiplies repeatedly)
//   slept   each call after `sleep` ms of idle time (a program that does other work between products)
//
// and for a warm (burst) and a cold (slept) call where the threads were busy:
//
//   pack     mean over the threads of the time spent copying A and B into panels
//   comp     mean / max over the threads of the time spent in the micro-kernel
//   wait     wall time minus (pack + comp) of the average thread: barriers, uneven work, start-up
//
//   gemm_thread_benchmark [key=v1,v2,...] ...
//     sizes=     square matrix sizes                   (default 1024,2048)
//     threads=   thread counts                         (default 4, half the hardware threads, all)
//     parallel=  0 two loops per step, 1 one region, 2 one region + private A (static rows),
//                3 dynamic tiles + private A, 4 next step packed during this one (double
//                buffered), 5 as 4 with the team split by L3 cache   (default 0,1,2,3)
//     pin=       0 threads not pinned, 1 pinned to a physical core, 2 to one hardware thread
//     loop=      0 B panel outer, 1 A panel outer      (default 0)
//     tiles=     tiles per thread, strategies 0, 1, 3  (default: the current default)
//     kc= mc=    block sizes                           (default: the current default)
//     rounds=    passes over all combinations          (default 5)
//     sleep=     idle time in ms before each "slept" call (default 5)
//     pool=      strategies 4, 5: 0 OpenMP region, 1 the library's thread pool (default 0)
//     spin=      pool only: microseconds an idle worker spins before sleeping (default 2000)
#include "bench.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/thread_pool.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

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

struct Variant {
    cpu::PackedGemmBlocking blocking;
    std::int64_t spin = 2000; // pool spin time, microseconds
    std::vector<double> median;
    std::vector<double> burst;
    std::vector<double> slept;
    cpu::PackedGemmProfile warm; // profile of the last burst call
    cpu::PackedGemmProfile cold; // profile of the last slept call
};

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double wait_percent(const cpu::PackedGemmProfile& p) {
    const double busy = p.pack_mean_s + p.compute_mean_s;
    return p.wall_s > 0.0 ? 100.0 * (1.0 - busy / p.wall_s) : 0.0;
}

} // namespace

int main(int argc, char** argv) {
    const int max = cpu::max_threads();
    const cpu::PackedGemmBlocking original = cpu::packed_gemm_blocking();
    std::map<std::string, std::vector<std::int64_t>> options = {
        {"sizes", {1024, 2048}},
        {"threads", {4, max / 2, max}},
        {"parallel", {0, 1, 2, 3}},
        {"pin", {0}},
        {"pool", {0}},
        {"spin", {2000}},
        {"loop", {0}},
        {"tiles", {original.tiles_per_thread}},
        {"kc", {original.kc}},
        {"mc", {original.mc}},
        {"rounds", {5}},
        {"sleep", {5}},
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
    const std::int64_t rounds = options["rounds"][0];
    const auto sleep_ms = std::chrono::milliseconds(options["sleep"][0]);

    bench::Config config;
    config.repeats = 7;
    bench::print_header("Threaded packed GEMM: parallelization strategies", config,
                        {"GFLOP/s are medians over " + std::to_string(rounds) +
                         " rounds; slept = " + std::to_string(options["sleep"][0]) +
                         " ms idle before each call"});

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

            std::vector<Variant> variants;
            for (std::int64_t parallel : options["parallel"]) {
                for (std::int64_t pin : options["pin"]) {
                for (std::int64_t pool : options["pool"]) {
                for (std::int64_t spin : options["spin"]) {
                    for (std::int64_t loop : options["loop"]) {
                        for (std::int64_t tiles : options["tiles"]) {
                            if (parallel == 2 && tiles != options["tiles"].front()) {
                                continue; // strategy 2 does not use tiles
                            }
                            for (std::int64_t kc : options["kc"]) {
                                for (std::int64_t mc : options["mc"]) {
                                    Variant v;
                                    v.blocking = original;
                                    v.blocking.parallel = parallel;
                                    v.blocking.pin = pin;
                                    v.blocking.pool = pool;
                                    v.spin = spin;
                                    v.blocking.loop_order = loop;
                                    v.blocking.tiles_per_thread = tiles;
                                    v.blocking.kc = kc;
                                    v.blocking.mc = mc;
                                    variants.push_back(v);
                                }
                            }
                        }
                    }
                }
                }
                }
            }

            auto run = [&] { cpu::matmul_f32_packed_parallel(a, b, c, n, n, n); };
            const auto seconds_of_run = [&] {
                const auto t0 = std::chrono::steady_clock::now();
                run();
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            };

            for (std::int64_t round = 0; round < rounds; ++round) {
                for (Variant& v : variants) {
                    // 1. bench::measure, with a checksum of C between the calls.
                    cpu::set_packed_gemm_blocking(v.blocking);
                    cpu::set_pool_spin_us(v.spin);
                    const bench::Sample sample =
                        bench::measure(config, run, [&] { return bench::checksum(c, len); });
                    v.median.push_back(flops / sample.median_s / 1e9);

                    // The next two are profiled, which costs a few timer reads per tile.
                    cpu::PackedGemmBlocking profiled = v.blocking;
                    profiled.profile = 1;
                    cpu::set_packed_gemm_blocking(profiled);

                    // 2. Back to back: 5 calls in a row, the first one dropped.
                    std::vector<double> rates;
                    for (int call = 0; call < 6; ++call) {
                        const double s = seconds_of_run();
                        if (call > 0) {
                            rates.push_back(flops / s / 1e9);
                        }
                    }
                    v.warm = cpu::packed_gemm_profile();
                    v.burst.push_back(median_of(rates));
                    bench::consume(bench::checksum(c, len));

                    // 3. After idle time: 5 calls, each preceded by a sleep.
                    rates.clear();
                    for (int call = 0; call < 5; ++call) {
                        std::this_thread::sleep_for(sleep_ms);
                        rates.push_back(flops / seconds_of_run() / 1e9);
                    }
                    v.cold = cpu::packed_gemm_profile();
                    v.slept.push_back(median_of(rates));
                }
            }

            std::printf("\n== n=%lld, %lld threads ==\n", static_cast<long long>(n),
                        static_cast<long long>(threads));
            std::printf("%-12s %-8s %-8s %5s %5s %5s | %8s %8s %8s | warm: %7s %7s %6s | cold: %7s %7s "
                        "%6s\n",
                        "pool/spin", "strategy", "loop/pin", "tiles", "kc", "mc", "median", "burst", "slept",
                        "comp", "pack", "wait%", "comp", "pack", "wait%");
            double best = 0.0;
            for (const Variant& v : variants) {
                best = std::max(best, median_of(v.median));
            }
            for (const Variant& v : variants) {
                std::printf(
                    "%lld/%-10lld %-8lld %3lld/%-4lld %5lld %5lld %5lld | %8.1f %8.1f %8.1f | "
                    "      %5.1fms %5.1fms %5.1f%% |       %5.1fms %5.1fms %5.1f%%%s\n",
                    static_cast<long long>(v.blocking.pool), static_cast<long long>(v.spin),
                    static_cast<long long>(v.blocking.parallel),
                    static_cast<long long>(v.blocking.loop_order),
                    static_cast<long long>(v.blocking.pin),
                    static_cast<long long>(v.blocking.tiles_per_thread),
                    static_cast<long long>(v.blocking.kc), static_cast<long long>(v.blocking.mc),
                    median_of(v.median), median_of(v.burst), median_of(v.slept),
                    v.warm.compute_mean_s * 1e3, v.warm.pack_mean_s * 1e3, wait_percent(v.warm),
                    v.cold.compute_mean_s * 1e3, v.cold.pack_mean_s * 1e3, wait_percent(v.cold),
                    median_of(v.median) == best ? "   <- best median" : "");
            }
            std::fflush(stdout);
        }
    }
    cpu::set_packed_gemm_blocking(original);
    cpu::set_max_threads(max);
    return 0;
}
