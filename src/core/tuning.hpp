// The auto-tuner's internals: the results file, the context dispatch() consults, and the
// measurement of one kernel call. The public side is include/mensurflow/tuning.hpp.
//
// A tuned call is identified by the machine, the kernel's name and device, the kernel's version,
// and its key (for matmul m, n and k, plus the thread count on the CPU). The results file is plain
// text, one result per line, and can be read and edited by hand:
//
//   # mensurflow tuning results
//   machine Apple M3 Pro | Apple M3 Pro
//   matmul metal v1 m=64 n=256 k=784 -> kernel=1 bm=32 bn=64 bk=16 tm=4 tn=4 sg_m=2 sg_n=2 | 0.0410 ms (default 0.0470 ms) tested
//
// Device choice for ops called with Execution::Auto has its own line kind: the op, its dtype, the
// device the data lives on, and the input shapes; then the chosen execution and every device's time
// (copies to the device and back included):
//
//   place matmul f32 cpu 256x1024 1024x4096 -> gpu | 0.8000 ms (cpu 8.4000 ms, gpu 0.8000 ms) tested
//
// Every result line belongs to the machine line above it, so one file can hold several machines.
// Blank lines and lines starting with '#' are ignored.
#pragma once

#include "core/dispatcher.hpp"
#include "mensurflow/tuning.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mensurflow::detail {

using NamedValues = std::vector<std::pair<std::string, std::int64_t>>;

struct TuningEntry {
    std::string machine;
    std::string op;     // TunableKernel::name
    std::string device; // device type name: cpu, cuda, metal
    int version = 1;
    NamedValues key;    // the kernel's key values, then "threads" on the CPU
    NamedValues config; // parameter name = value
    double best_ms = 0;
    double default_ms = 0;
    std::string mode; // how it was found: "tested" (exact shapes), later "range" and "online"
};

// Where an op called with Execution::Auto runs best, for one op, dtype, data device and input shapes.
struct PlacementEntry {
    std::string machine;
    std::string op;          // op_name, e.g. "matmul"
    std::string dtype;       // dtype of the first input, e.g. "f32"
    std::string data_device; // device type the inputs live on: cpu, cuda, metal
    std::vector<std::vector<std::int64_t>> shapes; // input shapes
    std::string chosen;                            // execution name: "cpu" or "gpu"
    double best_ms = 0;
    std::vector<std::pair<std::string, double>> times; // execution name -> ms (copies included)
    std::string mode;
};

bool same_placement(const PlacementEntry& a, const PlacementEntry& b);

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
                            const std::string& device, int version, const NamedValues& key) const;

    // Adds the entry, replacing one with the same machine, op, device and key.
    void store(TuningEntry entry);

    // Removes the entry with the same machine, op, device and key; false if there was none.
    bool erase(const TuningEntry& entry);

    const std::vector<TuningEntry>& entries() const { return entries_; }

    // The stored device choice for the same op, dtype, data device and shapes on this machine.
    const PlacementEntry* find_placement(const PlacementEntry& key) const;
    void store_placement(PlacementEntry entry); // replaces the same call
    bool erase_placement(const PlacementEntry& entry);

    const std::vector<PlacementEntry>& placements() const { return placements_; }

  private:
    std::vector<TuningEntry> entries_;
    std::vector<PlacementEntry> placements_;
};

// The placement key of a call (everything but the results), for this machine.
PlacementEntry placement_key(OpId op, const OpArgs& args, Device data_device);

// "CPU model | first GPU name": the machine results are stored under.
const std::string& machine_name();

// The key of a call: the kernel's key values, plus the thread count when it runs on the CPU.
NamedValues call_key(const TunableKernel& tunable, const OpArgs& args, Device device);

// The entry's parameters as the kernel's TuningConfig: values in the kernel's order, with the
// kernel's default for any parameter the entry lacks. Unknown names are ignored.
TuningConfig to_config(const TunableKernel& tunable, const NamedValues& config);
NamedValues from_config(const TunableKernel& tunable, const TuningConfig& config);

// ---------------------------------------------------------------------------- the search

// A point of a kernel's settings: TuningConfig's values.
using TuningPoint = std::array<std::int64_t, TuningConfig::capacity>;

TuningConfig with_values(const TuningConfig& base, const TuningPoint& values);

// Every point of a region (the product of its axes) that tunable.allowed accepts for the call, the
// last axis changing fastest.
std::vector<TuningPoint> region_points(const TunableKernel& tunable, const SearchRegion& region,
                                       const OpArgs& args);

// One point the search asks to be measured. `region` and `step` say why (the region's name;
// "start", an argument's name, or "grid"), `sweep` which pass over the axes and `level` which level
// of binary sampling (both 0 outside one).
struct Probe {
    TuningConfig config;
    std::string region;
    std::string step;
    int sweep = 0;
    int level = 0;
};

// The objective, for a batch of points the search could measure in any order (all the samples of
// one line, or a region's grid): the time of the call at each, in ms (infinity when the point is
// unusable, for example its result is wrong), in the order asked. It may be asked again about a
// point it already measured, so it should remember results.
using Measure = std::function<std::vector<double>(const std::vector<Probe>& probes)>;

// Searches every region of `space` for the point with the lowest `measure`, as `method` says
// (mensurflow::SearchMethod). Binary sampling samples each line down to `levels`.
// The caller keeps the best point measure() saw.
void search(const TunableKernel& tunable, const OpArgs& args,
            const std::vector<SearchRegion>& space, SearchMethod method, int max_sweeps, int levels,
            const Measure& measure);

// A tunable call seen while an algorithm ran: the kernel, where it ran, and its arguments (the
// tensors are shared, so they stay alive with the record).
struct RecordedCall {
    KernelKey kernel;
    Device device;
    OpArgs args;
};

// An op called with Execution::Auto, as dispatch() received it (inputs on the data device).
struct AutoCall {
    OpId op;
    Device data_device;
    OpArgs args;
};

// What dispatch() consults, set on this thread by a TuningScope.
struct TuningContext {
    const TuningCache* cache = nullptr;           // stored results to apply, if set
    std::vector<RecordedCall>* record = nullptr;  // every tunable call is appended here, if set
    std::vector<AutoCall>* record_auto = nullptr; // every Execution::Auto call, if set
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

// True while a TuningScope is active on this thread; dispatch() only looks for tunable kernels then.
bool tuning_active();

// Called by dispatch() for an op with Execution::Auto while a scope is active: records the call and
// returns the execution stored for it (CPU or GPU), or Auto when nothing is stored.
Execution auto_execution(OpId op, const OpArgs& args, Device data_device);

// Called by dispatch() just before a tunable kernel runs on `device` with `args`: records the
// call and sets args.tuning from the stored results of the active context. Does nothing without
// an active context.
void apply_tuning(KernelKey kernel, const TunableKernel& tunable, Device device, OpArgs& args);

} // namespace mensurflow::detail
