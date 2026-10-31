// Sequential vs parallel on the CPU. The same kernels run with 1 thread (sequential: parallel_for
// takes its serial path) and with more threads, so the only difference is the parallelism.
// Kernels are called directly into preallocated buffers: no dispatcher, no allocation.
// GB/s counts the bytes each op must touch (add/mul: 12 per element, dot: 8 per element).
#include "backends/cpu/blas.hpp"
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <thread>
#include <vector>

using namespace mensurflow::cpu;

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

} // namespace

int main() {
    const int max = max_threads();
    // 1 (sequential), a few powers of two, max / 2 (the physical core count when the CPU has
    // two hardware threads per core), and max.
    std::set<int> wanted{1, 2, 4, 8, 16, max / 2, max};
    std::vector<int> counts;
    for (int t : wanted) {
        if (t >= 1 && t <= max) {
            counts.push_back(t);
        }
    }

    std::cout << "Sequential vs parallel on the CPU (float32)\n"
              << "compiler: " << compiler() << ", OpenMP: " << (has_openmp() ? "yes" : "no")
              << ", max threads: " << max
              << ", hardware threads: " << std::thread::hardware_concurrency()
              << ", parallel cutoff: " << default_grain << " elements\n"
              << "threads = 1 is the sequential run. Median of 9 samples.\n"
              << std::fixed << std::setprecision(3);

    struct Op {
        const char* name;
        double bytes_per_element;
    };

    const Op ops[] = {{"add", 12.0}, {"mul", 12.0}, {"dot", 8.0}};
    const std::int64_t sizes[] = {1000, 10000, 100000, 1000000, 10000000, 33554432};

    for (const Op& op : ops) {
        std::cout << "\n== " << op.name << " ==\n"
                  << std::setw(12) << "n" << std::setw(9) << "threads" << std::setw(12) << "ms"
                  << std::setw(10) << "GB/s" << std::setw(11) << "speedup" << '\n';
        for (std::int64_t n : sizes) {
            std::vector<float> a(static_cast<std::size_t>(n)), b(a.size()), out(a.size());
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] = static_cast<float>(i % 13) / 8 - 0.75f;
                b[i] = static_cast<float>(i % 17) / 8 - 1.0f;
            }
            auto run = [&] {
                if (op.name[0] == 'a') {
                    add_f32(a.data(), b.data(), out.data(), n);
                } else if (op.name[0] == 'm') {
                    mul_f32(a.data(), b.data(), out.data(), n);
                } else {
                    sink = dot_f32(a.data(), b.data(), n);
                }
            };

            // The sequential result is the reference for the parallel ones.
            set_max_threads(1);
            run();
            const std::vector<float> reference = out;
            const float dot_reference = sink;

            double sequential = 0;
            for (int threads : counts) {
                set_max_threads(threads);
                const double seconds = median_seconds(run);
                if (threads == 1) {
                    sequential = seconds;
                }
                // Checked after timing. Elementwise ops must match bit for bit; dot may differ
                // slightly because partial sums are added in a different order.
                bool ok = true;
                if (op.name[0] == 'd') {
                    ok = std::abs(sink - dot_reference) <= 1e-4f * std::abs(dot_reference) + 1e-3f;
                } else {
                    ok = std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0;
                }
                if (!ok) {
                    std::cerr << op.name << " n=" << n << " threads=" << threads
                              << ": result differs from the sequential run\n";
                    return 1;
                }
                std::cout << std::setw(12) << n << std::setw(9) << threads << std::setw(12)
                          << seconds * 1e3 << std::setw(10)
                          << op.bytes_per_element * static_cast<double>(n) / seconds / 1e9
                          << std::setw(10) << sequential / seconds << "x\n";
            }
            std::cout << '\n';
        }
    }
    set_max_threads(max);
    return 0;
}
