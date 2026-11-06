// What the CPU can actually do with AVX2 FMAs: a loop of independent fused multiply-adds on values
// that stay in registers (no loads, no stores), compiled by the same compiler as the GEMM kernels.
// That is the ceiling any matmul micro-kernel can reach on this machine at its real clock, so
// "GFLOP/s of the kernel / GFLOP/s here" is its efficiency.
//
//   part 1: 1 thread, N independent accumulators. An FMA has a latency of about 4 cycles and the
//           core can start 2 per cycle, so it takes about 8 independent chains to keep both units
//           busy; more than 14 (16 registers minus the two FMA inputs) would spill.
//   part 2: 12 accumulators on 1, 2, 4, ... threads. Shows how the rate scales over physical cores
//           and what the second SMT thread of a core adds.
#include "bench.hpp"

#include "backends/cpu/threads.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

#if defined(_MSC_VER)
#define FMA_TARGET
#else
#define FMA_TARGET __attribute__((target("avx2,fma")))
#endif

template <int N, std::size_t... I>
FMA_TARGET inline void step(__m256 (&c)[N], __m256 a, __m256 b, std::index_sequence<I...>) {
    ((c[I] = _mm256_fmadd_ps(a, b, c[I])), ...);
}

// iters x N independent FMAs on registers; returns a value that depends on all of them.
template <int N> FMA_TARGET float fma_chains(std::int64_t iters, float seed) {
    __m256 c[N];
    for (int i = 0; i < N; ++i) {
        c[i] = _mm256_set1_ps(seed + static_cast<float>(i));
    }
    const __m256 a = _mm256_set1_ps(1.0000001f);
    const __m256 b = _mm256_set1_ps(0.0000001f);
    for (std::int64_t it = 0; it < iters; ++it) {
        step<N>(c, a, b, std::make_index_sequence<N>{});
    }
    __m256 sum = c[0];
    for (int i = 1; i < N; ++i) {
        sum = _mm256_add_ps(sum, c[i]);
    }
    alignas(32) float out[8];
    _mm256_store_ps(out, sum);
    return out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7];
}

constexpr std::int64_t iterations = 2'000'000;

template <int N> bench::Sample time_chains(const bench::Config& config, float& sink) {
    return bench::measure(
        config, [&] { sink = fma_chains<N>(iterations, 1.0f); }, [&] { return double(sink); });
}

template <int N> void row(const bench::Config& config) {
    float sink = 0.0f;
    const bench::Sample sample = time_chains<N>(config, sink);
    const double flops = double(iterations) * N * 16.0; // 8 floats x (multiply + add)
    std::printf("%12d %12.1f %14.2f\n", N, flops / sample.median_s / 1e9,
                flops / sample.median_s / 1e9 / N);
}

} // namespace

int main() {
    const bench::Config config;
    bench::print_header("AVX2 FMA throughput without memory traffic", config,
                        {"GFLOP/s counts 16 flops per FMA (8 floats, multiply and add)"});

    std::printf("\n== 1 thread, N independent accumulators ==\n");
    std::printf("%12s %12s %14s\n", "accumulators", "GFLOP/s", "per accumulator");
    row<1>(config);
    row<2>(config);
    row<4>(config);
    row<6>(config);
    row<8>(config);
    row<10>(config);
    row<12>(config);
    row<14>(config);

    std::printf("\n== 12 accumulators on several threads ==\n");
    std::printf("%12s %12s %14s\n", "threads", "GFLOP/s", "per thread");
    const int max = cpu::max_threads();
    for (int threads : bench::thread_counts(max)) {
        cpu::set_max_threads(threads);
        std::vector<float> sinks(static_cast<std::size_t>(threads) * 16, 0.0f);
        auto run = [&] {
            MENSURFLOW_PARALLEL_FOR(schedule(static, 1) num_threads(threads))
            for (int t = 0; t < threads; ++t) {
                sinks[static_cast<std::size_t>(t) * 16] = fma_chains<12>(iterations, 1.0f);
            }
        };
        const bench::Sample sample =
            bench::measure(config, run, [&] { return double(sinks[0]); });
        const double flops = double(iterations) * 12 * 16.0 * threads;
        std::printf("%12d %12.1f %14.1f\n", threads, flops / sample.median_s / 1e9,
                    flops / sample.median_s / 1e9 / threads);
    }
    cpu::set_max_threads(max);
    return 0;
}
