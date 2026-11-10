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
    long long l3_key = -1; // identifies the L3 cache of the core (-1: unknown)
    int domain = 0;        // compact L3 domain number, see core_cache_domain
};

// Numbers the L3 caches 0, 1, ... in the order of the cores.
void number_domains(std::vector<Core>& cores) {
    std::vector<long long> keys;
    for (Core& core : cores) {
        const auto found = std::find(keys.begin(), keys.end(), core.l3_key);
        core.domain = static_cast<int>(found - keys.begin());
        if (found == keys.end()) {
            keys.push_back(core.l3_key);
        }
    }
}

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
    // The L3 caches: each core belongs to the one whose processor mask overlaps its own.
    DWORD cache_bytes = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &cache_bytes);
    if (cache_bytes > 0) {
        std::vector<char> caches(cache_bytes);
        auto* head = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(caches.data());
        if (GetLogicalProcessorInformationEx(RelationCache, head, &cache_bytes)) {
            long long number = 0;
            for (char* p = caches.data(); p < caches.data() + cache_bytes;) {
                auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
                if (info->Relationship == RelationCache && info->Cache.Level == 3) {
                    const GROUP_AFFINITY& mask = info->Cache.GroupMask;
                    for (Core& core : cores) {
                        if (core.affinity.Group == mask.Group && (core.affinity.Mask & mask.Mask) != 0) {
                            core.l3_key = number;
                        }
                    }
                    ++number;
                }
                p += info->Size;
            }
        }
    }
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
            // The L3 cache: the "id" of the level-3 entry among the core's caches.
            for (int index = 0; index < 8; ++index) {
                const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                        "/cache/index" + std::to_string(index) + "/";
                std::ifstream level_file(dir + "level");
                if (!level_file) {
                    break;
                }
                int level = 0;
                level_file >> level;
                if (level == 3) {
                    std::ifstream id_file(dir + "id");
                    long long id = -1;
                    if (id_file >> id) {
                        core.l3_key = id;
                    }
                    break;
                }
            }
            cores.push_back(core);
        }
    }
#endif
    number_domains(cores);
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

int core_cache_domain(int index) {
    const std::vector<Core>& list = cores();
    if (list.empty() || index < 0) {
        return 0;
    }
    return list[static_cast<std::size_t>(index) % list.size()].domain;
}


int current_core_index() {
    const std::vector<Core>& list = cores();
#if defined(_WIN32)
    PROCESSOR_NUMBER now{};
    GetCurrentProcessorNumberEx(&now);
    const KAFFINITY bit = static_cast<KAFFINITY>(1) << now.Number;
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (list[i].affinity.Group == now.Group && (list[i].affinity.Mask & bit) != 0) {
            return static_cast<int>(i);
        }
    }
#elif defined(__linux__)
    const int cpu = sched_getcpu();
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (std::find(list[i].cpus.begin(), list[i].cpus.end(), cpu) != list[i].cpus.end()) {
            return static_cast<int>(i);
        }
    }
#endif
    return -1;
}
int cache_domain_count() {
    int count = 1;
    for (const Core& core : cores()) {
        count = std::max(count, core.domain + 1);
    }
    return count;
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
