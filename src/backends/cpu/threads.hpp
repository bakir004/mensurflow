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

// For kernels that run several loops inside one team of threads (so the threads are woken once, not
// once per loop). Put MENSURFLOW_PARALLEL_REGION(clauses) on the line before a `{ ... }` block; inside
// it, MENSURFLOW_FOR(clauses) before a `for` loop shares that loop's iterations among the team (and
// waits for all of them at the end, unless the clauses say `nowait`), and MENSURFLOW_BARRIER() makes
// every thread wait for the others. Without OpenMP the block runs once on the calling thread and the
// other two are nothing.
#ifdef _OPENMP
#define MENSURFLOW_PARALLEL_REGION(...) MENSURFLOW_PRAGMA(omp parallel __VA_ARGS__)
#define MENSURFLOW_FOR(...) MENSURFLOW_PRAGMA(omp for __VA_ARGS__)
#define MENSURFLOW_BARRIER() MENSURFLOW_PRAGMA(omp barrier)
#else
#define MENSURFLOW_PARALLEL_REGION(...)
#define MENSURFLOW_FOR(...)
#define MENSURFLOW_BARRIER()
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

// True when called from inside a parallel region (the calling thread is one of several workers).
inline bool in_parallel_region() {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

// Inside a parallel region: this thread's number (0 .. team size - 1) and the size of the team.
inline int thread_index() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline int team_size() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

// Wall-clock seconds from an arbitrary starting point, for timing inside kernels.
inline double wall_seconds() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
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
