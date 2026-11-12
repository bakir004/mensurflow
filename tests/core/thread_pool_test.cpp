// The thread pool (src/backends/cpu/thread_pool.hpp): every thread of a team runs the body exactly
// once with its own index, barriers separate phases, team sizes can change between calls, sleeping
// workers wake up, and a second team (a concurrent caller or a call from inside a body) is refused.
#include "backends/cpu/thread_pool.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

// Runs a team of `team` threads through `phases` barrier-separated phases. In phase p every thread
// adds its tid + 1 to a per-phase slot; after the barrier, thread 0 checks the slot of phase p is
// complete (no thread got ahead of the barrier).
void check_team(int team, int phases) {
    std::vector<std::atomic<std::int64_t>> sums(static_cast<std::size_t>(phases));
    for (auto& s : sums) {
        s.store(0);
    }
    std::vector<std::atomic<int>> runs(static_cast<std::size_t>(team));
    for (auto& r : runs) {
        r.store(0);
    }
    std::atomic<bool> bad_index{false};
    std::atomic<bool> barrier_broken{false};
    const std::int64_t expected = static_cast<std::int64_t>(team) * (team + 1) / 2;
    auto body = [&](int tid) {
        if (tid < 0 || tid >= team || cpu::pool_thread_index() != tid) {
            bad_index = true;
            return;
        }
        runs[static_cast<std::size_t>(tid)].fetch_add(1);
        for (int p = 0; p < phases; ++p) {
            sums[static_cast<std::size_t>(p)].fetch_add(tid + 1);
            cpu::pool_barrier();
            if (sums[static_cast<std::size_t>(p)].load() != expected) {
                barrier_broken = true;
            }
            cpu::pool_barrier(); // nobody starts phase p + 1 before everyone has checked phase p
        }
    };
    test::require(cpu::pool_run(team, body), "pool_run accepts a team of " + std::to_string(team));
    test::require(!bad_index, "thread indices 0 .. team - 1, matching pool_thread_index()");
    test::require(!barrier_broken, "barriers separate the phases (team " + std::to_string(team) + ")");
    for (int t = 0; t < team; ++t) {
        test::require(runs[static_cast<std::size_t>(t)].load() == 1,
                      "every thread runs the body once (team " + std::to_string(team) + ")");
    }
    test::require(cpu::pool_thread_index() == -1, "the caller is not in a team after pool_run");
}

} // namespace

int main() {
    return test::run("thread_pool_test", [] {
        const int hardware = std::max(2, static_cast<int>(std::thread::hardware_concurrency()));
        test::require(cpu::pool_thread_index() == -1, "no team outside pool_run");

        // Growing, shrinking and oversubscribed teams, with many phases.
        for (int team : {1, 2, 3, hardware, 5, hardware + 3, 2, 1, hardware}) {
            check_team(team, 50);
        }
        test::require(cpu::pool_worker_count() >= hardware + 2, "workers are started on demand");

        // Many short calls in a row (the spinning path).
        for (int i = 0; i < 2000; ++i) {
            check_team(4, 1);
        }

        // Workers that went to sleep (no spinning at all) must still wake up.
        cpu::set_pool_spin_us(0);
        for (int i = 0; i < 50; ++i) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            check_team(i % 2 == 0 ? hardware : 3, 3);
        }
        cpu::set_pool_spin_us(2000);

        // A call from inside a body is refused (one team at a time), and the outer team is unaffected.
        std::atomic<int> nested_accepted{0};
        auto outer = [&](int) {
            auto inner = [](int) {};
            if (cpu::pool_run(2, inner)) {
                nested_accepted.fetch_add(1);
            }
        };
        test::require(cpu::pool_run(3, outer), "outer team runs");
        test::require(nested_accepted.load() == 0, "pool_run from inside a body is refused");

        // A team with one worker left out: every tid except that one runs once, the barrier counts
        // only the members, and the left-out worker is back in the next ordinary team.
        for (int skip : {1, 2, 4}) {
            for (int repeat = 0; repeat < 200; ++repeat) {
                std::vector<std::atomic<int>> runs(6);
                for (auto& r : runs) {
                    r.store(0);
                }
                std::atomic<int> before{0};
                std::atomic<bool> early{false};
                auto body = [&](int tid) {
                    runs[static_cast<std::size_t>(tid)].fetch_add(1);
                    before.fetch_add(1);
                    cpu::pool_barrier();
                    if (before.load() != 5) {
                        early = true;
                    }
                };
                test::require(cpu::pool_run(6, body, skip), "a team with a worker left out runs");
                for (int t = 0; t < 6; ++t) {
                    test::require(runs[static_cast<std::size_t>(t)].load() == (t == skip ? 0 : 1),
                                  "every member but the left-out worker runs once");
                }
                test::require(!early, "the barrier waits for the five members");
            }
            check_team(6, 3);
        }
        auto nothing = [](int) {};
        test::require(!cpu::pool_run(4, nothing, 4) && !cpu::pool_run(4, nothing, -1),
                      "a skip outside the team is refused");

        // Concurrent callers: each call either runs on the pool or is refused, never both teams at
        // once, and every accepted call completes.
        std::atomic<int> inside{0};
        std::atomic<bool> overlap{false};
        std::atomic<int> accepted{0};
        std::atomic<int> refused{0};
        auto worker = [&] {
            for (int i = 0; i < 300; ++i) {
                auto body = [&](int tid) {
                    if (tid == 0 && inside.fetch_add(1) != 0) {
                        overlap = true;
                    }
                    cpu::pool_barrier();
                    if (tid == 0) {
                        inside.fetch_sub(1);
                    }
                };
                if (cpu::pool_run(3, body)) {
                    accepted.fetch_add(1);
                } else {
                    refused.fetch_add(1);
                }
            }
        };
        std::thread t1(worker);
        std::thread t2(worker);
        std::thread t3(worker);
        t1.join();
        t2.join();
        t3.join();
        test::require(!overlap, "two teams never run at the same time");
        test::require(accepted.load() + refused.load() == 900, "every call returns");
        test::require(accepted.load() > 0, "some concurrent calls run on the pool");
        std::cout << "workers: " << cpu::pool_worker_count() << ", concurrent calls accepted "
                  << accepted.load() << ", refused " << refused.load() << '\n';
    });
}
