// Auto-tuning: choosing kernel settings (tile sizes and the like) for the calls a particular
// algorithm makes, measured on this machine and kept in a text file for later runs.
//
//   mf::Algorithm mlp([](const std::vector<mf::Tensor>& in) {
//       mf::Tensor h = mf::relu(mf::matmul(in[0], in[1], mf::Execution::GPU));
//       return std::vector<mf::Tensor>{mf::matmul(h, in[2], mf::Execution::GPU)};
//   });
//   mf::Tuner tuner("mensurflow_tuning.txt");  // reads earlier results, if the file exists
//   tuner.tune(mlp, {x, w1, w2});               // tunes the calls mlp makes on these inputs
//   auto out = mlp.run({x, w1, w2}, tuner);     // runs with the stored settings
//
// Two things are tuned:
//   - the settings of kernels that declare some (so far matmul on Metal, and on CPUs without AVX2);
//   - the device of every op the algorithm calls with Execution::Auto (the default of every op):
//     each available device is timed with the copies there and back, and the fastest is stored.
//     Without a stored result, Auto runs on the CPU.
// Results are stored per machine, so a file copied to another computer is not applied there.
#pragma once

#include "mensurflow/tensor.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mensurflow {

namespace detail {
class TuningCache;
}

class Tuner;

// An algorithm written as a function from input tensors to output tensors, kept ready to run.
// The function may call any mensurflow ops; it is called once per run (and several times while it is
// tuned and timed), so it should not have side effects beyond its outputs.
class Algorithm {
  public:
    using Function = std::function<std::vector<Tensor>(const std::vector<Tensor>& inputs)>;

    explicit Algorithm(Function function);

    // Runs with every kernel's own settings.
    std::vector<Tensor> run(const std::vector<Tensor>& inputs) const;

    // Runs with the settings stored in `tuner` wherever they exist for this machine; calls
    // without a stored result use the kernel's own settings.
    std::vector<Tensor> run(const std::vector<Tensor>& inputs, const Tuner& tuner) const;

  private:
    Function function_;
};

struct TuneOptions {
    bool retune = false;      // also tune calls that already have a stored result
    bool save = true;         // write the file at the end of tune()
    double min_time_s = 0.05; // per configuration: time samples until this much has passed ...
    int min_samples = 3;      // ... and at least this many samples (the median is kept)
    bool verbose = false;     // print each call's result to std::cout as it is found
};

using TuningValues = std::vector<std::pair<std::string, std::int64_t>>; // name = value, in order

// One configuration measured for a call.
struct TunedCandidate {
    TuningValues settings;
    double ms = 0;         // median per call; 0 when rejected
    bool rejected = false; // its result disagreed with the default's, so it was not timed
};

// One tunable call of the algorithm and what tuning found for it.
struct TunedCall {
    std::string op;          // e.g. "matmul"
    std::string device;      // e.g. "metal:0"
    std::string key;         // e.g. "m=64 n=256 k=784"
    std::string best;        // the chosen settings, e.g. "kernel=1 bm=32 ..."
    TuningValues key_values; // the key as name = value (the input sizes the call is tuned for)
    TuningValues default_settings; // what the kernel uses without tuning
    TuningValues best_settings;    // the chosen settings
    double best_ms = 0;
    double default_ms = 0;
    int tried = 0;       // configurations measured
    int rejected = 0;    // configurations whose result disagreed with the default's (not used)
    bool reused = false; // the result was already stored and was not measured again
    std::vector<TunedCandidate> candidates; // every configuration tried, in the order tried
};

// One op called with Execution::Auto and the device tuning chose for it. Times include copying the
// inputs to that device and the result back to where the data lives.
struct PlacedCall {
    std::string op;          // e.g. "matmul"
    std::string data_device; // where the inputs live, e.g. "cpu:0"
    std::string key;         // dtype and input shapes, e.g. "f32 256x1024 1024x4096"
    std::string chosen;      // "cpu" or "gpu"
    double best_ms = 0;
    double default_ms = 0; // what Execution::Auto does without tuning (the CPU)
    std::vector<std::pair<std::string, double>> times; // "cpu"/"gpu" -> ms; absent if rejected
    std::vector<std::string> rejected; // executions whose result disagreed with the CPU's
    bool reused = false;               // stored earlier and not measured again
};

struct TuneReport {
    std::vector<TunedCall> calls;
    std::vector<PlacedCall> placements; // the ops called with Execution::Auto
    // The whole algorithm, median ms per run, with every kernel's own settings and with the
    // tuned ones. When the tuned run was slower, the new results were dropped (`kept` is false).
    double default_ms = 0;
    double tuned_ms = 0;
    bool kept = true;
};

// A kernel whose settings can be tuned.
struct TunableKernelInfo {
    std::string name;   // e.g. "matmul"
    std::string device; // device type: cpu, cuda, metal
    std::vector<std::string>
        key; // what a call is identified by, e.g. m, n, k (plus threads on the CPU)
    std::vector<std::string> params; // the settings tuning chooses
};

// Every tunable kernel this build registered, for the available devices.
std::vector<TunableKernelInfo> tunable_kernels();

class Tuner {
  public:
    // Reads the results already stored in `path`; a missing file is fine (it is created on save).
    // Throws runtime_error if the file exists but is malformed.
    explicit Tuner(std::string path = "mensurflow_tuning.txt");

    // Offline, tested: runs `algorithm` once on `inputs`, finds every call of a tunable kernel,
    // and measures each candidate configuration on exactly that call, keeping the fastest one
    // whose result agrees with the default's. Calls already stored are skipped unless
    // options.retune. Then the whole algorithm is timed with the default and the tuned settings;
    // if tuning made it slower, the new results are dropped. Finally the file is saved.
    TuneReport tune(const Algorithm& algorithm, const std::vector<Tensor>& inputs,
                    const TuneOptions& options = {});

    // Writes the results to the file. Throws runtime_error if it cannot.
    void save() const;

    const std::string& path() const { return path_; }

    // Results stored for this machine: kernel settings and device choices.
    std::size_t size() const;

    // The results stored for this machine, as they appear in the file.
    std::string describe() const;

  private:
    friend class Algorithm;
    std::string path_;
    std::shared_ptr<detail::TuningCache> cache_;
};

} // namespace mensurflow
