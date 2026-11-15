// parallel_for and parallel_sum: the two loops parallel kernels are built from, so the kernels
// themselves contain no pragmas. Both split work into pieces that never overlap; neither takes a
// lock or an atomic.
//
// Determinism: parallel_for hands each item to exactly one call of the body, so a body that writes
// only its own items gives the same bits at every thread count. parallel_sum (and parallel_reduce)
// cut the range into blocks whose size the caller fixes, independent of the thread count, and
// combine the block results in block order on the calling thread; the result is therefore the same
// at every thread count too, though it can differ in the last bits from a plain sequential loop.
#pragma once

#include "backends/cpu/affinity.hpp"
#include "backends/cpu/thread_pool.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace mensurflow::cpu {

// Calls body(begin, end) on contiguous pieces of [0, n) that together cover it once. Uses up to
// max_threads() threads, but never gives a thread fewer than `grain` items, so small ranges (fewer
// than 2 * grain items), single-thread builds, and calls from inside a parallel region or a pool
// body run body(0, n) on the calling thread.
//
// The pieces run on the thread pool the packed GEMM uses (thread_pool.hpp), so a kernel that
// alternates its own loops with GEMM calls (im2col, then a product, then a copy) keeps one set of
// threads: OpenMP's workers and the pool's workers both spin for a while after a job, and when the
// two alternate each set steals the cores the other one is about to need (conv2d at batch 1 lost
// more than half its time to this). OpenMP is the fallback when the pool is busy.
//
// parallel_for_pinned is the same on at most `threads` threads (and still at most max_threads()),
// each piece on its own physical core (while there are no more pieces than cores): piece p runs on
// core p, the calling thread (piece 0) pinned to core 0 for the duration of its piece and pool
// worker p pinned to core p (where it stays, as the GEMM leaves its workers). With more pieces than
// cores, piece p shares core p modulo the core count. For bodies that run one long
// single-threaded computation per piece
// (whole images, tiles of a convolution): unpinned, the OS now and then keeps two of them on one
// core for a whole process, and the piece there runs at half speed and holds up the call (batch-1
// conv1 in tiles: 0.33 ms in some processes, 0.59 ms in others). Pinning costs a few system calls
// per call, so short loops use parallel_for.
template <typename Body>
void parallel_for_placed(int threads, bool pinned, std::int64_t n, std::int64_t grain,
                         Body&& body) {
    if (n <= 0) {
        return;
    }
    const std::int64_t by_grain = n / std::max<std::int64_t>(grain, 1);
    const std::int64_t pieces =
        std::min<std::int64_t>(std::min(threads, max_threads()), by_grain);
    if (pieces <= 1 || in_parallel_region() || pool_thread_index() >= 0) {
        body(std::int64_t{0}, n);
        return;
    }
    auto piece = [&](int p) {
        if (pinned) {
            set_thread_pin(p, 1);
        }
        body(n * p / pieces, n * (p + 1) / pieces);
        if (pinned && p == 0) {
            set_thread_pin(0, 0); // the caller gets its old placement back
        }
    };
    if (pool_run(static_cast<int>(pieces), piece)) {
        return;
    }
    MENSURFLOW_PARALLEL_FOR(num_threads(static_cast<int>(pieces)) schedule(static, 1))
    for (std::int64_t p = 0; p < pieces; ++p) {
        body(n * p / pieces, n * (p + 1) / pieces);
    }
}

template <typename Body>
void parallel_for_pinned(int threads, std::int64_t n, std::int64_t grain, Body&& body) {
    parallel_for_placed(threads, true, n, grain, body);
}

template <typename Body> void parallel_for(std::int64_t n, std::int64_t grain, Body&& body) {
    parallel_for_placed(max_threads(), false, n, grain, body);
}

// The number of items that holds at least `work` units when one item costs `cost` units: the
// grain to pass parallel_for for a cutoff given in units of work (elements, multiply-adds).
inline std::int64_t grain_for(std::int64_t work, std::int64_t cost) {
    return std::max<std::int64_t>(1, (work + std::max<std::int64_t>(cost, 1) - 1) /
                                         std::max<std::int64_t>(cost, 1));
}

// Splits [0, n) into blocks of `block` items (the last may be shorter), computes
// map(begin, end) for every block in parallel, then folds the results in block order with
// combine(accumulated, next), starting from `identity`. `min_blocks_per_thread` keeps tiny inputs
// on one thread.
template <typename T, typename Map, typename Combine>
T parallel_reduce(std::int64_t n, std::int64_t block, T identity, Map&& map, Combine&& combine,
                  std::int64_t min_blocks_per_thread = 1) {
    if (n <= 0) {
        return identity;
    }
    block = std::max<std::int64_t>(block, 1);
    const std::int64_t blocks = (n + block - 1) / block;
    std::vector<T> partial(static_cast<std::size_t>(blocks), identity);
    parallel_for(blocks, min_blocks_per_thread, [&](std::int64_t b0, std::int64_t b1) {
        for (std::int64_t b = b0; b < b1; ++b) {
            partial[static_cast<std::size_t>(b)] = map(b * block, std::min(n, (b + 1) * block));
        }
    });
    T result = identity;
    for (const T& value : partial) {
        result = combine(result, value);
    }
    return result;
}

// parallel_reduce with +: the sum of map(begin, end) over fixed blocks.
template <typename T, typename Map>
T parallel_sum(std::int64_t n, std::int64_t block, Map&& map,
               std::int64_t min_blocks_per_thread = 1) {
    return parallel_reduce(
        n, block, T{}, map, [](const T& a, const T& b) { return a + b; }, min_blocks_per_thread);
}

} // namespace mensurflow::cpu
