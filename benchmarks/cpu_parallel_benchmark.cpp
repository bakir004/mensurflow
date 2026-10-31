// Sequential vs parallel on the CPU.
//   sequential: the plain single-loop kernels in sequential.cpp (no OpenMP, no parallel_for).
//   parallel:   the kernels in elementwise.cpp / blas.cpp (parallel_for / parallel_sum) run with
//               1, 2, 4, ... threads. The 1-thread row shows what the parallel machinery costs
//               when there is nothing to gain.
// Speedup is always sequential time / this time. Kernels are called directly into preallocated
// buffers: no dispatcher, no allocation. GB/s counts the bytes each op must touch (add/mul: 12
// per element, dot: 8 per element).
#include "backends/cpu/blas.hpp"
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace seq = mensurflow::cpu::sequential;

namespace {

using Clock = std::chrono::steady_clock;

const char* compiler() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

// Median seconds per call; fast calls are repeated until a sample lasts at least ~2 ms.
template <typename F> double median_seconds(F&& call) {
    call(); // warm up caches and the thread pool
    int reps = 1;
    for (;;) {
        const auto start = Clock::now();
        for (int r = 0; r < reps; ++r) {
            call();
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        if (elapsed >= 2e-3 || reps >= (1 << 20)) {
            break;
        }
        reps *= 2;
    }
    std::vector<double> samples;
    for (int s = 0; s < 9; ++s) {
        const auto start = Clock::now();
        for (int r = 0; r < reps; ++r) {
            call();
        }
        samples.push_back(std::chrono::duration<double>(Clock::now() - start).count() / reps);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

volatile float sink = 0; // keeps dot's result from being optimized away

using Kernel = void (*)(const float*, const float*, float*, std::int64_t);

struct Op {
    const char* name;
    double bytes_per_element;
    Kernel sequential; // nullptr for dot, which returns a value instead of filling a buffer
    Kernel parallel;
};

void row(const char* impl, std::int64_t n, const std::string& threads, double seconds,
         double bytes_per_element, double sequential_seconds) {
    std::cout << std::setw(12) << n << std::setw(12) << impl << std::setw(9) << threads
              << std::setw(12) << seconds * 1e3 << std::setw(10)
              << bytes_per_element * static_cast<double>(n) / seconds / 1e9 << std::setw(10)
              << sequential_seconds / seconds << "x\n";
}

} // namespace

int main() {
    const int max = cpu::max_threads();
    // 1, a few powers of two, max / 2 (the physical core count when the CPU has two hardware
    // threads per core), and max.
    std::set<int> wanted{1, 2, 4, 8, 16, max / 2, max};
    std::vector<int> counts;
    for (int t : wanted) {
        if (t >= 1 && t <= max) {
            counts.push_back(t);
        }
    }

    std::cout << "Sequential vs parallel on the CPU (float32)\n"
              << "compiler: " << compiler() << ", OpenMP: " << (cpu::has_openmp() ? "yes" : "no")
              << ", max threads: " << max
              << ", hardware threads: " << std::thread::hardware_concurrency()
              << ", parallel cutoff: " << cpu::default_grain << " elements\n"
              << "sequential = plain loop, no OpenMP. parallel = parallel kernel with N threads.\n"
              << "Speedup = sequential time / this time. Median of 9 samples.\n"
              << std::fixed << std::setprecision(3);

    const Op ops[] = {{"add", 12.0, seq::add_f32, cpu::add_f32},
                      {"mul", 12.0, seq::mul_f32, cpu::mul_f32},
                      {"dot", 8.0, nullptr, nullptr}};
    const std::int64_t sizes[] = {1000, 10000, 100000, 1000000, 10000000, 33554432};

    for (const Op& op : ops) {
        const bool is_dot = op.sequential == nullptr;
        std::cout << "\n== " << op.name << " ==\n"
                  << std::setw(12) << "n" << std::setw(12) << "impl" << std::setw(9) << "threads"
                  << std::setw(12) << "ms" << std::setw(10) << "GB/s" << std::setw(11) << "speedup"
                  << '\n';
        for (std::int64_t n : sizes) {
            // The four arrays start 1 KB apart modulo 4 KB. Equally sized arrays tend to start at
            // the same offset within a page, and addresses that agree in their low 12 bits make
            // loads and stores falsely depend on each other ("4K aliasing"). That can slow one op
            // by 2x and distort the comparison.
            constexpr std::size_t stagger = 256; // floats, 1 KB
            const auto len = static_cast<std::size_t>(n);
            std::vector<float> pool_a(len + 4 * stagger), pool_b(len + 4 * stagger),
                pool_out(len + 4 * stagger), pool_ref(len + 4 * stagger);
            float* a = pool_a.data();
            float* b = pool_b.data() + stagger;
            float* out = pool_out.data() + 2 * stagger;
            float* reference = pool_ref.data() + 3 * stagger;
            for (std::size_t i = 0; i < len; ++i) {
                a[i] = static_cast<float>(i % 13) / 8 - 0.75f;
                b[i] = static_cast<float>(i % 17) / 8 - 1.0f;
            }

            // Sequential baseline.
            auto run_sequential = [&] {
                if (is_dot) {
                    sink = seq::dot_f32(a, b, n);
                } else {
                    op.sequential(a, b, reference, n);
                }
            };
            run_sequential();
            const float dot_reference = sink;
            const double sequential = median_seconds(run_sequential);
            row("sequential", n, "-", sequential, op.bytes_per_element, sequential);

            // Parallel kernel at each thread count.
            for (int threads : counts) {
                cpu::set_max_threads(threads);
                auto run_parallel = [&] {
                    if (is_dot) {
                        sink = cpu::dot_f32(a, b, n);
                    } else {
                        op.parallel(a, b, out, n);
                    }
                };
                const double seconds = median_seconds(run_parallel);

                // Checked after timing. Elementwise results must match the sequential ones bit
                // for bit; dot may differ slightly because partial sums are added in a
                // different order.
                bool ok;
                if (is_dot) {
                    ok = std::abs(sink - dot_reference) <= 1e-4f * std::abs(dot_reference) + 1e-3f;
                } else {
                    ok = std::memcmp(out, reference, len * sizeof(float)) == 0;
                }
                if (!ok) {
                    std::cerr << op.name << " n=" << n << " threads=" << threads
                              << ": parallel result differs from the sequential one\n";
                    return 1;
                }
                row("parallel", n, std::to_string(threads), seconds, op.bytes_per_element,
                    sequential);
            }
            std::cout << '\n';
        }
    }
    cpu::set_max_threads(max);
    return 0;
}
