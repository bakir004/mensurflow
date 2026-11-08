// Tests thread pinning (src/backends/cpu/affinity.hpp): pinning restricts the calling thread to one
// physical core, different thread numbers go to different cores, the logical mode narrows it to a
// single hardware thread, the ideal mode restricts nothing, and unpinning gives back exactly what the
// thread had. Runs on a fresh thread so the test runner's own thread is never touched. On a platform
// where the topology cannot be read it only checks that every call is a harmless no-op.
#include "backends/cpu/affinity.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <thread>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace cpu = mensurflow::cpu;

namespace {

// The set of hardware threads the calling thread may run on, as a bit set (group 0 on Windows).
std::bitset<1024> current_affinity() {
    std::bitset<1024> set;
#if defined(_WIN32)
    GROUP_AFFINITY affinity{};
    if (GetThreadGroupAffinity(GetCurrentThread(), &affinity) != 0) {
        for (unsigned bit = 0; bit < 64; ++bit) {
            if (((affinity.Mask >> bit) & 1ULL) != 0) {
                set.set(affinity.Group * 64u + bit);
            }
        }
    }
#elif defined(__linux__)
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0) {
        for (int cpu = 0; cpu < 1024; ++cpu) {
            if (CPU_ISSET(cpu, &cpus)) {
                set.set(static_cast<std::size_t>(cpu));
            }
        }
    }
#endif
    return set;
}

void run_checks() {
    const int cores = cpu::physical_core_count();
    const auto original = current_affinity();
    if (cores <= 0 || original.none()) {
        // No topology on this platform: calls must do nothing and not fail.
        cpu::set_thread_pin(0, 1);
        cpu::set_thread_pin(0, 0);
        return;
    }

    // Pinned to a core: one or two hardware threads (two with SMT), inside the original set.
    cpu::set_thread_pin(0, 1);
    const auto core0 = current_affinity();
    test::require(core0.any() && core0.count() <= 2, "pinning to a core leaves 1 or 2 hardware threads");
    test::require((core0 & ~original).none(), "the pinned set is inside the original set");

    // The same call again changes nothing; another thread number goes to another core.
    cpu::set_thread_pin(0, 1);
    test::require(current_affinity() == core0, "pinning twice is the same as once");
    if (cores >= 2) {
        cpu::set_thread_pin(1, 1);
        const auto core1 = current_affinity();
        test::require(core1.any() && (core1 & core0).none(), "thread 1 is on a different core than 0");
        // Thread numbers wrap around: thread `cores` shares thread 0's core.
        cpu::set_thread_pin(cores, 1);
        test::require(current_affinity() == core0, "thread number cores wraps to core 0");
    }

    // Logical mode narrows to a single hardware thread of that core.
    cpu::set_thread_pin(0, 2);
    const auto logical = current_affinity();
    test::require(logical.count() == 1, "logical pinning leaves exactly one hardware thread");
    test::require((logical & ~core0).none(), "the logical thread belongs to the same core");

    // Unpinning gives back exactly the original set, and doing it twice is harmless.
    cpu::set_thread_pin(0, 0);
    test::require(current_affinity() == original, "unpinning restores the original affinity");
    cpu::set_thread_pin(0, 0);
    test::require(current_affinity() == original, "unpinning twice is harmless");

    // Ideal mode only expresses a preference: the allowed set does not change.
    cpu::set_thread_pin(0, 3);
    test::require(current_affinity() == original, "ideal mode does not restrict the thread");
    cpu::set_thread_pin(0, 0);
    test::require(current_affinity() == original, "unpinning after ideal mode keeps the original");

    // Going from one mode straight to another and back still restores the original.
    cpu::set_thread_pin(0, 1);
    cpu::set_thread_pin(0, 2);
    cpu::set_thread_pin(0, 3);
    cpu::set_thread_pin(0, 0);
    test::require(current_affinity() == original, "switching modes and unpinning restores the original");
}

} // namespace

int main() {
    return test::run("affinity_test", [] {
        std::string failure;
        std::thread worker([&failure] {
            try {
                run_checks();
            } catch (const std::exception& error) {
                failure = error.what();
            }
        });
        worker.join();
        test::require(failure.empty(), failure);
        std::cout << "physical cores: " << cpu::physical_core_count() << '\n';

        // L3 cache domains: numbered from 0 in core order, each one used, the same for i and i + cores.
        const int domains = cpu::cache_domain_count();
        const int cores = cpu::physical_core_count();
        test::require(domains >= 1, "at least one cache domain");
        std::vector<bool> used(static_cast<std::size_t>(domains), false);
        int highest = -1;
        std::cout << "L3 domains: " << domains << " (";
        for (int i = 0; i < std::max(cores, 1); ++i) {
            const int d = cpu::core_cache_domain(i);
            test::require(d >= 0 && d < domains, "domain in range");
            test::require(d <= highest + 1, "domains numbered in order of first appearance");
            highest = std::max(highest, d);
            used[static_cast<std::size_t>(d)] = true;
            test::require(cpu::core_cache_domain(i + cores) == d, "domain wraps around the cores");
            std::cout << d << (i + 1 < cores ? " " : "");
        }
        std::cout << ")\n";
        for (bool u : used) {
            test::require(u || cores == 0, "every domain has a core");
        }
    });
}
