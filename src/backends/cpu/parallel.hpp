// The one place that knows about OpenMP. Kernels call parallel_for and never use pragmas
// themselves, so builds without OpenMP (Apple clang, minimal toolchains) run serially.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mensurflow::cpu {

// Below this many elements the cost of waking threads exceeds the work saved.
// Measure with benchmarks/elementwise_benchmark.cpp before changing it.
constexpr std::int64_t default_grain = 1 << 15;

constexpr bool has_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

// Threads the next parallel_for may use; 1 without OpenMP.
inline int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Ignored without OpenMP. OMP_NUM_THREADS sets the initial value.
inline void set_max_threads(int threads) {
#ifdef _OPENMP
    omp_set_num_threads(std::max(threads, 1));
#else
    (void)threads;
#endif
}

// Block [begin, end) of [0, n) for thread `id` of `threads`. Edges are multiples of `align`; the
// default of 16 floats is one 64-byte cache line, so two threads never write the same line. Use
// align = 1 when each index is a large unit of work (a matrix row) and a fine split matters more.
inline void block_range(std::int64_t n, std::int64_t threads, std::int64_t id, std::int64_t& begin,
                        std::int64_t& end, std::int64_t align = 16) {
    const std::int64_t chunk = ((n + threads - 1) / threads + align - 1) / align * align;
    begin = std::min(id * chunk, n);
    end = std::min(begin + chunk, n);
}

// Calls body(begin, end) on disjoint ranges that together cover [0, n). Each thread gets one
// contiguous block, so inner loops stay vectorizable. Runs serially when n <= grain or only one
// thread is available.
// body must not throw: exceptions cannot cross an OpenMP region.
template <typename Body>
void parallel_for(std::int64_t n, Body&& body, std::int64_t grain = default_grain,
                  std::int64_t align = 16) {
    if (n <= 0) {
        return;
    }
#ifdef _OPENMP
    if (n > grain && omp_get_max_threads() > 1) {
#pragma omp parallel
        {
            std::int64_t begin = 0, end = 0;
            block_range(n, omp_get_num_threads(), omp_get_thread_num(), begin, end, align);
            if (begin < end) {
                body(begin, end);
            }
        }
        return;
    }
#endif
    (void)grain;
    (void)align; // only used by the OpenMP path
    body(std::int64_t{0}, n);
}

// Sum of body(begin, end) over the same blocks as parallel_for. Each thread returns a partial
// sum and the partials are added in thread order, so the result depends on the thread count
// (float addition is not associative) but not on scheduling: reruns with the same count match.
template <typename Body>
double parallel_sum(std::int64_t n, Body&& body, std::int64_t grain = default_grain) {
    if (n <= 0) {
        return 0.0;
    }
#ifdef _OPENMP
    if (n > grain && omp_get_max_threads() > 1) {
        std::vector<double> partial(static_cast<std::size_t>(omp_get_max_threads()), 0.0);
#pragma omp parallel
        {
            const int id = omp_get_thread_num();
            std::int64_t begin = 0, end = 0;
            block_range(n, omp_get_num_threads(), id, begin, end);
            if (begin < end) {
                partial[static_cast<std::size_t>(id)] = body(begin, end);
            }
        }
        double total = 0.0;
        for (double p : partial) {
            total += p;
        }
        return total;
    }
#endif
    (void)grain;
    return body(std::int64_t{0}, n);
}

template <typename Body>
float parallel_max(std::int64_t n, Body&& body, std::int64_t grain = default_grain) {
    if (n <= 0) {
        return 0.0f;
    }
#ifdef _OPENMP
    if (n > grain && omp_get_max_threads() > 1) {
        std::vector<float> partial(static_cast<std::size_t>(omp_get_max_threads()),
                                   -std::numeric_limits<float>::infinity());
#pragma omp parallel
        {
            const int id = omp_get_thread_num();
            std::int64_t begin = 0, end = 0;
            block_range(n, omp_get_num_threads(), id, begin, end);
            if (begin < end) {
                partial[static_cast<std::size_t>(id)] = body(begin, end);
            }
        }
        float result = partial[0];
        for (float value : partial) {
            if (std::isnan(value)) {
                return value;
            }
            if (std::isnan(result)) {
                return result;
            }
            result = std::fmax(result, value);
        }
        return result;
    }
#endif
    (void)grain;
    return body(std::int64_t{0}, n);
}

} // namespace mensurflow::cpu
