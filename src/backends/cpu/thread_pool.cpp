#include "backends/cpu/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define MENSURFLOW_CPU_RELAX() _mm_pause()
#else
#define MENSURFLOW_CPU_RELAX() std::this_thread::yield()
#endif

namespace mensurflow::cpu {

namespace {

// The job word: a job counter in the high bits and the team size in the low 16, so a worker reads both
// with one load and never pairs a new counter with an old team size.
constexpr std::uint64_t team_bits = 16;
constexpr std::uint64_t team_mask = (std::uint64_t{1} << team_bits) - 1;

// What a sleeping worker waits on. One per worker, so waking the team wakes exactly its members and
// they do not queue up behind one mutex.
struct alignas(64) Sleeper {
    std::atomic<bool> asleep{false};
    std::mutex mutex;
    std::condition_variable wake;
};

struct Pool {
    std::atomic<std::uint64_t> job{0};
    PoolBody body{nullptr, nullptr};          // valid while the participating workers run
    std::atomic<int> remaining{0};            // participating workers that have not finished
    std::atomic<bool> busy{false};            // a team is running
    std::atomic<int> started{0};              // worker threads started
    std::atomic<std::int64_t> spin_us{2000};
    // sleepers[w - 1] belongs to worker w. Only the thread that holds `busy` reads or grows the
    // vector; a worker keeps a pointer to its own entry.
    std::vector<std::unique_ptr<Sleeper>> sleepers;

    // Barrier: threads that have arrived in the current phase, and the phase number.
    alignas(64) std::atomic<int> arrived{0};
    alignas(64) std::atomic<std::uint32_t> phase{0};
    int team = 1;
};

// Never destroyed: the workers are detached and still blocked in it when the process exits.
Pool& pool() {
    static Pool* instance = new Pool();
    return *instance;
}

thread_local int t_tid = -1;

// `seen` is the job word before the worker was started: a worker that starts running only after the
// next job was published must still run that job, so it must not read its starting point itself.
void worker_loop(int w, std::uint64_t seen, Sleeper* self) {
    Pool& p = pool();
    bool member = true; // of the job `seen` (a new worker is about to get its first job)

    for (;;) {
        std::uint64_t now = p.job.load(std::memory_order_acquire);
        if (now == seen) {
            // A member of the last team spins for spin_us (the next call usually wants it again),
            // checking the clock every 256 pauses, then sleeps. A worker the last team did not need
            // sleeps at once: spinning, it would only take the other hardware thread of a core from
            // a thread that is working.
            const auto start = std::chrono::steady_clock::now();
            const auto budget = std::chrono::microseconds(p.spin_us.load(std::memory_order_relaxed));
            for (int i = member ? 1 : 256; (now = p.job.load(std::memory_order_acquire)) == seen; ++i) {
                if ((i & 255) == 0 && (!member || std::chrono::steady_clock::now() - start > budget)) {
                    std::unique_lock<std::mutex> lock(self->mutex);
                    self->asleep.store(true, std::memory_order_seq_cst);
                    // Woken only for a job it is a member of; other jobs pass it by.
                    self->wake.wait(lock, [&] {
                        const std::uint64_t j = p.job.load(std::memory_order_seq_cst);
                        return j != seen && w < static_cast<int>(j & team_mask);
                    });
                    self->asleep.store(false, std::memory_order_relaxed);
                } else {
                    MENSURFLOW_CPU_RELAX();
                }
            }
        }
        seen = now;
        member = w < static_cast<int>(now & team_mask);
        if (member) {
            t_tid = w;
            p.body.call(p.body.object, w);
            t_tid = -1;
            p.remaining.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
}

} // namespace

void set_pool_spin_us(std::int64_t microseconds) {
    pool().spin_us.store(microseconds < 0 ? 0 : microseconds, std::memory_order_relaxed);
}

std::int64_t pool_spin_us() {
    return pool().spin_us.load(std::memory_order_relaxed);
}

int pool_worker_count() {
    return pool().started.load(std::memory_order_relaxed);
}

int pool_thread_index() {
    return t_tid;
}

bool pool_run(int team, PoolBody body) {
    Pool& p = pool();
    if (team < 1 || team > static_cast<int>(team_mask) || t_tid >= 0 ||
        p.busy.exchange(true, std::memory_order_acquire)) {
        return false;
    }
    // Start the workers this team needs (worker w is always tid w).
    for (int w = p.started.load(std::memory_order_relaxed) + 1; w < team; ++w) {
        const std::uint64_t before = p.job.load(std::memory_order_relaxed);
        p.sleepers.push_back(std::make_unique<Sleeper>());
        Sleeper* const self = p.sleepers.back().get();
        std::thread([w, before, self] { worker_loop(w, before, self); }).detach();
        p.started.store(w, std::memory_order_relaxed);
    }
    p.body = body;
    p.team = team;
    p.arrived.store(0, std::memory_order_relaxed);
    p.remaining.store(team - 1, std::memory_order_relaxed);
    const std::uint64_t counter = (p.job.load(std::memory_order_relaxed) >> team_bits) + 1;
    p.job.store((counter << team_bits) | static_cast<std::uint64_t>(team), std::memory_order_seq_cst);
    // Wake the members that are asleep. A worker sets `asleep` before it checks the job word and we
    // publish the job word before we check `asleep` (both sequentially consistent), so one of the two
    // sees the other.
    for (int w = 1; w < team; ++w) {
        Sleeper& sleeper = *p.sleepers[static_cast<std::size_t>(w) - 1];
        if (sleeper.asleep.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lock(sleeper.mutex);
            sleeper.wake.notify_one();
        }
    }

    t_tid = 0;
    body.call(body.object, 0);
    t_tid = -1;
    while (p.remaining.load(std::memory_order_acquire) > 0) {
        MENSURFLOW_CPU_RELAX();
    }
    p.busy.store(false, std::memory_order_release);
    return true;
}

void pool_barrier() {
    Pool& p = pool();
    if (p.team <= 1) {
        return;
    }
    const std::uint32_t phase = p.phase.load(std::memory_order_acquire);
    if (p.arrived.fetch_add(1, std::memory_order_acq_rel) == p.team - 1) {
        p.arrived.store(0, std::memory_order_relaxed);
        p.phase.store(phase + 1, std::memory_order_release);
        return;
    }
    for (int i = 1; p.phase.load(std::memory_order_acquire) == phase; ++i) {
        MENSURFLOW_CPU_RELAX();
        if ((i & 4095) == 0) {
            std::this_thread::yield(); // a long wait: let the other hardware thread of the core run
        }
    }
}

} // namespace mensurflow::cpu
