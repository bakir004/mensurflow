// The auto-tuner's internals: the results file, the hook dispatch() calls, and the search. The
// public side is include/mensurflow/tuning.hpp.
//
// The results file is plain text, one result per line under the machine it was measured on, and can
// be read and edited by hand:
//
//   # mensurflow tuning results
//   machine Apple M3 Pro | Apple M3 Pro
//   matmul metal v1 m=64 n=256 k=784 -> kernel=1 bm=32 bn=64 bk=16 tm=4 tn=4 sg_m=2 sg_n=2 | 0.0410 ms (default 0.0470 ms) tested
//   matmul auto v1 data=cpu f32 256x1024 1024x4096 -> gpu=1 | 0.8000 ms (default 8.4000 ms) tested
//
// Every line is   <op> <device> v<version> <key> -> <settings> | <best> ms (default <ms> ms) <mode>
//   - kernel settings: <device> is the device type the kernel ran on, <key> the kernel's key (for
//     matmul m, n and k, plus the thread count on the CPU), <settings> the kernel's parameters;
//   - device choice for an op called with Execution::Auto: <device> is "auto", <key> the device the
//     data lives on, the dtype and the input shapes, and the one setting is gpu=0 or gpu=1. Times
//     include copying the inputs to the device and the result back; the default is the CPU.
// A result is applied only to a call with exactly its machine, op, device, version and key. Blank
// lines and lines starting with '#' are ignored.
#pragma once

#include "core/dispatcher.hpp"
#include "mensurflow/tuning.hpp"

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace mensurflow::detail {

// The <device> of a device-choice line.
inline const char* const kAutoDevice = "auto";

// One line of the results file.
struct TuningEntry {
    std::string machine;
    std::string op;     // TunableKernel::name, or the op's name for a device choice
    std::string device; // device type name (cpu, cuda, metal), or "auto"
    int version = 1;
    std::string key;     // the call, as text: "m=64 n=256 k=784" or "data=cpu f32 8x8 8x8"
    TuningValues config; // parameter name = value
    double best_ms = 0;
    double default_ms = 0;
    std::string mode; // how it was found: "tested" (exact shapes), later "range" and "online"
};

class TuningCache {
  public:
    // Reads `path`; a missing file gives an empty cache. Throws runtime_error naming the line of a
    // malformed file.
    void load(const std::string& path);

    // Writes every entry, grouped by machine. Throws runtime_error if the file cannot be written.
    void save(const std::string& path) const;
    void write(std::ostream& out) const; // the file's text

    // The entry for exactly this call, or nullptr.
    const TuningEntry* find(const std::string& machine, const std::string& op,
                            const std::string& device, int version, const std::string& key) const;

    // Adds the entry, replacing one for the same call.
    void store(TuningEntry entry);

    // Removes the entry for the same call; false if there was none.
    bool erase(const TuningEntry& entry);

    const std::vector<TuningEntry>& entries() const { return entries_; }

  private:
    std::vector<TuningEntry> entries_;
};

// "CPU model | first GPU name": the machine results are stored under.
const std::string& machine_name();

// The key of a kernel call: the kernel's key values, plus the thread count when it runs on the CPU.
std::string call_key(const TunableKernel& tunable, const OpArgs& args, Device device);

// The key of a device choice: where the data lives, the dtype and the input shapes.
std::string placement_key(const OpArgs& args, Device data_device);

// Settings by name as the kernel's TuningConfig (the kernel's default for a parameter the settings
// lack; unknown names are ignored), and back.
TuningConfig to_config(const TunableKernel& tunable, const TuningValues& settings);
TuningValues from_config(const TunableKernel& tunable, const TuningConfig& config);

// ---------------------------------------------------------------------------- the search

// Every point of a region (the product of its axes) that tunable.allowed accepts for the call, the
// last axis changing fastest.
std::vector<TuningConfig> region_points(const TunableKernel& tunable, const SearchRegion& region,
                                        const OpArgs& args);

// The objective: the call's time in ms at each of a batch of points (infinity for a point that is
// unusable, for example because its result is wrong), in the order asked. It may be asked again
// about a point it already measured, so it should remember results.
using Measure = std::function<std::vector<double>(const std::vector<TuningConfig>& points)>;

// Searches every region of `space` for the point with the lowest `measure`, as `method` says
// (mensurflow::SearchMethod). Binary sampling samples each line down to `levels`. The caller keeps
// the best point measure() saw.
void search(const TunableKernel& tunable, const OpArgs& args,
            const std::vector<SearchRegion>& space, SearchMethod method, int max_sweeps, int levels,
            const Measure& measure);

// ---------------------------------------------------------------------------- dispatch side

// One op call seen while an algorithm ran: the execution asked for, where the data lives, where it
// ran, and its arguments (on the data device; the tensors are shared, so they stay alive).
struct RecordedCall {
    OpId op;
    Execution execution;
    Device data_device;
    Device device;
    OpArgs args;
};

// What dispatch() consults, set on this thread by a TuningScope.
struct TuningContext {
    const TuningCache* cache = nullptr;          // stored results to apply, if set
    std::vector<RecordedCall>* record = nullptr; // every op call is appended here, if set
};

class TuningScope {
  public:
    explicit TuningScope(TuningContext context);
    ~TuningScope();
    TuningScope(const TuningScope&) = delete;
    TuningScope& operator=(const TuningScope&) = delete;

  private:
    TuningContext* previous_;
    TuningContext context_;
};

// True while a TuningScope is active on this thread.
bool tuning_active();

// Called by dispatch() for every op while a scope is active. Returns the device to run on (the
// stored choice for an Execution::Auto call, if there is one), records the call, and sets
// args.tuning to the stored settings of the kernel that will run.
Device apply_tuning(OpId op, Execution execution, Device data_device, OpArgs& args);

} // namespace mensurflow::detail
