// Matrix multiplication, sequential vs parallel, for square matrices.
//   sequential: the plain i-k-j loop (sequential.cpp), no blocking, no threads.
//   parallel:   the cache-blocked kernel (matmul.cpp) with 1, 2, 4, ... threads. The 1-thread row
//               isolates what cache blocking alone buys; the rows after it add the threads.
// GFLOP/s counts 2*n^3 floating point operations. Speedup = sequential time / this time.
// Kernels are called directly into preallocated matrices: no dispatcher, no allocation.
#include "backends/cpu/matmul.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
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

double seconds_for(int reps, const std::function<void()>& call) {
    const auto start = Clock::now();
    for (int r = 0; r < reps; ++r) {
        call();
    }
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Median seconds per call. One warm-up call tells how long a call takes: slow ones (big matrices)
// get 3 samples of one call, fast ones are repeated until a sample lasts ~2 ms and get 9 samples.
double median_seconds(const std::function<void()>& call) {
    const double first = seconds_for(1, call);
    int reps = 1;
    int samples = 3;
    if (first <= 0.2) {
        samples = 9;
        while (seconds_for(reps, call) < 2e-3 && reps < (1 << 20)) {
            reps *= 2;
        }
    }
    std::vector<double> times;
    for (int s = 0; s < samples; ++s) {
        times.push_back(seconds_for(reps, call) / reps);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void row(const char* impl, std::int64_t n, const std::string& threads, double seconds,
         double sequential_seconds) {
    const double flops =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    std::cout << std::setw(8) << n << std::setw(12) << impl << std::setw(9) << threads
              << std::setw(12) << seconds * 1e3 << std::setw(11) << flops / seconds / 1e9
              << std::setw(10) << sequential_seconds / seconds << "x\n";
}

} // namespace

int main() {
    const int max = cpu::max_threads();
    std::set<int> wanted{1, 2, 4, 8, 16, max / 2, max};
    std::vector<int> counts;
    for (int t : wanted) {
        if (t >= 1 && t <= max) {
            counts.push_back(t);
        }
    }

    std::cout << "Matrix multiplication C = A * B, square float32 matrices\n"
              << "compiler: " << compiler() << ", OpenMP: " << (cpu::has_openmp() ? "yes" : "no")
              << ", max threads: " << max
              << ", hardware threads: " << std::thread::hardware_concurrency() << '\n'
              << "sequential = plain i-k-j loop. parallel = blocked kernel with N threads.\n"
              << "Speedup = sequential time / this time.\n\n"
              << std::setw(8) << "n" << std::setw(12) << "impl" << std::setw(9) << "threads"
              << std::setw(12) << "ms" << std::setw(11) << "GFLOP/s" << std::setw(11) << "speedup"
              << '\n'
              << std::fixed << std::setprecision(3);

    for (std::int64_t n : {64, 128, 256, 512, 1024}) {
        // The three matrices start 1 KB apart modulo 4 KB, so they do not alias in the cache.
        constexpr std::size_t stagger = 256;
        const auto len = static_cast<std::size_t>(n * n);
        std::vector<float> pool_a(len + 3 * stagger), pool_b(len + 3 * stagger),
            pool_c(len + 3 * stagger), pool_ref(len + 3 * stagger);
        float* a = pool_a.data();
        float* b = pool_b.data() + stagger;
        float* c = pool_c.data() + 2 * stagger;
        float* reference = pool_ref.data() + 3 * stagger;
        for (std::size_t i = 0; i < len; ++i) {
            a[i] = static_cast<float>(i % 13) / 8 - 0.75f;
            b[i] = static_cast<float>(i % 17) / 8 - 1.0f;
        }

        auto run_sequential = [&] { seq::matmul_f32(a, b, reference, n, n, n); };
        const double sequential = median_seconds(run_sequential);
        row("sequential", n, "-", sequential, sequential);

        for (int threads : counts) {
            cpu::set_max_threads(threads);
            const double seconds = median_seconds([&] { cpu::matmul_f32(a, b, c, n, n, n); });
            // Checked after timing: the blocked kernel sums each entry in the same order as the
            // sequential loop, so the results agree up to compiler differences in rounding.
            const float tolerance = 1e-5f * static_cast<float>(n);
            for (std::size_t i = 0; i < len; ++i) {
                if (std::abs(c[i] - reference[i]) > tolerance) {
                    std::cerr << "n=" << n << " threads=" << threads
                              << ": parallel result differs from the sequential one at " << i
                              << '\n';
                    return 1;
                }
            }
            row("parallel", n, std::to_string(threads), seconds, sequential);
        }
        std::cout << '\n';
    }
    cpu::set_max_threads(max);
    return 0;
}
