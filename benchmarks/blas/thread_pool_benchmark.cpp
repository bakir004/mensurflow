// What the thread pool itself costs (src/backends/cpu/thread_pool.hpp): the time of a call whose
// threads do nothing, of one barrier, and of a call whose threads each do a fixed amount of work
// (so the difference to that work is what starting and joining the team adds), per team size.
// A small product takes tens of microseconds, so these few microseconds decide how many threads are
// worth using for it.
//
//   thread_pool_benchmark [team sizes ...]     default 2 4 6 8 12 and all hardware threads
#include "backends/cpu/affinity.hpp"
#include "backends/cpu/thread_pool.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// About `units` * 1 ns of arithmetic that the compiler cannot remove.
double spin_work(int units) {
    volatile double x = 1.0;
    for (int i = 0; i < units; ++i) {
        x = x * 1.0000001 + 0.1;
    }
    return x;
}

template <typename F> double per_call_us(int calls, F&& f) {
    for (int i = 0; i < calls / 10 + 1; ++i) {
        f();
    }
    std::vector<double> rounds;
    for (int r = 0; r < 5; ++r) {
        const double t0 = seconds();
        for (int i = 0; i < calls; ++i) {
            f();
        }
        rounds.push_back((seconds() - t0) / calls * 1e6);
    }
    std::sort(rounds.begin(), rounds.end());
    return rounds[rounds.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> teams;
    for (int i = 1; i < argc; ++i) {
        teams.push_back(std::atoi(argv[i]));
    }
    if (teams.empty()) {
        teams = {2, 4, 6, 8, 12, cpu::max_threads()};
    }
    const int work_units = 20000;
    const double alone = per_call_us(2000, [&] { spin_work(work_units); });
    std::printf("thread pool: microseconds per call (median of 5 rounds); the work alone takes %.2f us\n",
                alone);
    std::printf("%-6s %-8s | %10s %10s %12s %12s\n", "team", "pinned", "empty", "barrier",
                "with work", "overhead");
    for (int pinned = 0; pinned < 2; ++pinned) {
        for (int team : teams) {
            if (team < 1) {
                continue;
            }
            auto place = [&](int tid) {
                if (tid != 0) {
                    cpu::set_thread_pin(tid, pinned);
                }
            };
            cpu::pool_run(team, place);
            auto empty = [](int) {};
            auto barrier = [](int) { cpu::pool_barrier(); };
            auto work = [&](int) { spin_work(work_units); };
            const double t_empty = per_call_us(20000, [&] { cpu::pool_run(team, empty); });
            const double t_barrier = per_call_us(20000, [&] { cpu::pool_run(team, barrier); });
            const double t_work = per_call_us(5000, [&] { cpu::pool_run(team, work); });
            std::printf("%-6d %-8s | %10.2f %10.2f %12.2f %12.2f\n", team, pinned ? "yes" : "no",
                        t_empty, t_barrier - t_empty, t_work, t_work - alone);
            std::fflush(stdout);
        }
    }
    return 0;
}
