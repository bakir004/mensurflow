#include "backends/cpu/affinity.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <string>
#endif

namespace mensurflow::cpu {

namespace {

// One physical core: the hardware threads it contains, best kind of core first.
struct Core {
#if defined(_WIN32)
    GROUP_AFFINITY affinity{};      // all hardware threads of the core
    GROUP_AFFINITY first_thread{};  // the first one of them
    int efficiency_class = 0;       // higher = faster kind of core (hybrid CPUs)
#elif defined(__linux__)
    std::vector<int> cpus; // sorted logical processor numbers
#endif
};

std::vector<Core> read_cores() {
    std::vector<Core> cores;
#if defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) {
        return cores;
    }
    std::vector<char> buffer(bytes);
    auto* first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &bytes)) {
        return cores;
    }
    for (char* p = buffer.data(); p < buffer.data() + bytes;) {
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1) {
            Core core;
            core.affinity = info->Processor.GroupMask[0];
            core.first_thread = core.affinity;
            // Keep only the lowest set bit.
            core.first_thread.Mask = core.affinity.Mask & (~core.affinity.Mask + 1);
            core.efficiency_class = info->Processor.EfficiencyClass;
            cores.push_back(core);
        }
        p += info->Size;
    }
    std::stable_sort(cores.begin(), cores.end(), [](const Core& x, const Core& y) {
        return x.efficiency_class > y.efficiency_class;
    });
#elif defined(__linux__)
    // Cores are identified by the list of hardware threads that share them, for example "0,12" or
    // "0-1"; the first thread of each list names the core.
    std::vector<std::vector<int>> seen;
    for (int cpu = 0; cpu < 4096; ++cpu) {
        std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/topology/thread_siblings_list");
        if (!file) {
            break;
        }
        std::string text;
        std::getline(file, text);
        std::vector<int> cpus;
        std::stringstream parts(text);
        std::string part;
        while (std::getline(parts, part, ',')) {
            const auto dash = part.find('-');
            if (dash == std::string::npos) {
                cpus.push_back(std::stoi(part));
            } else {
                for (int i = std::stoi(part.substr(0, dash)); i <= std::stoi(part.substr(dash + 1));
                     ++i) {
                    cpus.push_back(i);
                }
            }
        }
        std::sort(cpus.begin(), cpus.end());
        if (!cpus.empty() && cpus.front() == cpu) {
            Core core;
            core.cpus = cpus;
            cores.push_back(core);
        }
    }
#endif
    return cores;
}

const std::vector<Core>& cores() {
    static const std::vector<Core> list = read_cores();
    return list;
}

} // namespace

int physical_core_count() {
    return static_cast<int>(cores().size());
}

bool pin_current_thread(int index, PinMode mode) {
    const std::vector<Core>& list = cores();
    if (list.empty() || index < 0) {
        return false;
    }
    const Core& core = list[static_cast<std::size_t>(index) % list.size()];
#if defined(_WIN32)
    if (mode == PinMode::Ideal) {
        // Only a preference: the first hardware thread of the core.
        PROCESSOR_NUMBER preferred{};
        preferred.Group = core.first_thread.Group;
        unsigned long long mask = core.first_thread.Mask;
        BYTE bit = 0;
        while (bit < 63 && ((mask >> bit) & 1ULL) == 0) {
            ++bit;
        }
        preferred.Number = bit;
        return SetThreadIdealProcessorEx(GetCurrentThread(), &preferred, nullptr) != 0;
    }
    const GROUP_AFFINITY& target = mode == PinMode::Core ? core.affinity : core.first_thread;
    GROUP_AFFINITY previous{};
    return SetThreadGroupAffinity(GetCurrentThread(), &target, &previous) != 0;
#elif defined(__linux__)
    if (mode == PinMode::Ideal) {
        return false; // Linux has no "preferred processor" hint: leave the thread unrestricted
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    if (mode == PinMode::Core) {
        for (int cpu : core.cpus) {
            CPU_SET(cpu, &set);
        }
    } else {
        CPU_SET(core.cpus.front(), &set);
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)mode;
    return false;
#endif
}


namespace {

// What this thread was restricted to before we pinned it, and what we did.
struct PinState {
    int applied = 0;        // 0 none, 1 core, 2 logical, 3 ideal
    int applied_index = -1; // the index it was pinned for
    bool have_original = false;
#if defined(_WIN32)
    GROUP_AFFINITY original{};
    PROCESSOR_NUMBER original_ideal{};
    bool have_original_ideal = false;
#elif defined(__linux__)
    cpu_set_t original;
#endif
};

PinState& pin_state() {
    thread_local PinState state;
    return state;
}

} // namespace

void set_thread_pin(int index, int wanted) {
    PinState& state = pin_state();
    if (wanted == state.applied && (wanted == 0 || index == state.applied_index)) {
        return;
    }

    // Give back what the thread had before, for whichever kind of restriction is in force now.
    if (state.applied != 0 && state.have_original) {
#if defined(_WIN32)
        if (state.applied == 3) {
            if (state.have_original_ideal) {
                SetThreadIdealProcessorEx(GetCurrentThread(), &state.original_ideal, nullptr);
            }
        } else {
            SetThreadGroupAffinity(GetCurrentThread(), &state.original, nullptr);
        }
#elif defined(__linux__)
        pthread_setaffinity_np(pthread_self(), sizeof(state.original), &state.original);
#endif
    }
    state.applied = 0;
    if (wanted == 0) {
        return;
    }

    if (!state.have_original) {
#if defined(_WIN32)
        state.have_original = GetThreadGroupAffinity(GetCurrentThread(), &state.original) != 0;
        state.have_original_ideal =
            GetThreadIdealProcessorEx(GetCurrentThread(), &state.original_ideal) != 0;
#elif defined(__linux__)
        state.have_original =
            pthread_getaffinity_np(pthread_self(), sizeof(state.original), &state.original) == 0;
#endif
    }
    const PinMode mode = wanted == 1 ? PinMode::Core : wanted == 2 ? PinMode::Logical : PinMode::Ideal;
    if (pin_current_thread(index, mode)) {
        state.applied = wanted;
        state.applied_index = index;
    }
}

} // namespace mensurflow::cpu
