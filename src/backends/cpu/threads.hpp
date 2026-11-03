// The few OpenMP facts the CPU backend, tests and benchmarks need.
#pragma once

#include <algorithm>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// MENSURFLOW_PARALLEL_FOR(clauses): put it on the line before a `for` loop to split the loop's
// iterations across threads. With OpenMP it becomes `#pragma omp parallel for clauses`; without
// OpenMP it is nothing, and the loop runs on the calling thread. For example:
//
//     MENSURFLOW_PARALLEL_FOR(if (n > parallel_threshold) reduction(+ : sum))
//     for (std::int64_t i = 0; i < n; ++i) { sum += x[i]; }
#ifdef _MSC_VER
#define MENSURFLOW_PRAGMA(...) __pragma(__VA_ARGS__)
#else
#define MENSURFLOW_PRAGMA(...) _Pragma(#__VA_ARGS__)
#endif

#ifdef _OPENMP
#define MENSURFLOW_PARALLEL_FOR(...) MENSURFLOW_PRAGMA(omp parallel for __VA_ARGS__)
#else
#define MENSURFLOW_PARALLEL_FOR(...)
#endif

namespace mensurflow::cpu {

// Below this many elements, waking threads costs more than the work they would share. Parallel
// kernels pass it to their pragma's `if` clause, so smaller inputs run on the calling thread.
constexpr std::int64_t parallel_threshold = 1 << 15;

// Below this many multiply-adds a matrix kernel (matvec, gemv, matmul, axis reductions) stays on
// one thread.
constexpr std::int64_t parallel_work_threshold = 1 << 18;

constexpr bool has_openmp() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

// Threads the next parallel region may use; 1 without OpenMP.
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

} // namespace mensurflow::cpu
