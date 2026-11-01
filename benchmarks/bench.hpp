// Shared benchmark harness: every benchmark in this repository measures, checks, and prints
// through these functions, so the numbers and tables look the same whoever wrote the benchmark.
// How to write one: docs/BENCHMARKING.md.
//
//   1. print_header()  machine, compiler, and build, so a result can be traced.
//   2. Table::section() then, for each size and each implementation:
//        check_*()      the result matches the reference (before and outside the timing),
//        measure()      warm up, then time only the call, then read the result back,
//        Table::row()   print median, min, max, rate, speedups, and CPU/wall.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

// ---------------------------------------------------------------- measuring

struct Config {
    int warmups = 1;            // calls before timing: caches, page faults, thread pool, CPU clock
    int repeats = 9;            // timed samples; the median is reported
    double min_sample_s = 2e-3; // a fast call is repeated until one sample lasts at least this long
    double slow_call_s = 0.2;   // a call slower than this is timed one call per sample ...
    int slow_repeats = 3;       // ... with this many samples
};

struct Sample {
    double median_s = 0; // wall-clock seconds per call
    double min_s = 0;
    double max_s = 0;
    // Process CPU time / wall time over all the timed samples together: about 1 for one busy
    // thread, about N for N busy threads, lower when threads wait. 0 if the run was too short for
    // the OS CPU clock, which only advances in coarse ticks, to give a meaningful ratio.
    double cpu_over_wall = 0;
    int calls_per_sample = 1;
    std::vector<double> runs_s; // every sample, per call
};

// Process CPU time in seconds (all threads), for the CPU/wall column. A parallel kernel on 8 busy
// threads shows a ratio near 8; a ratio near 1 means the threads were mostly idle.
double process_cpu_seconds();

// Keep the compiler from deleting work whose result is never read: both are noinline and fold
// their input into a global atomic.
void consume(double value);
double checksum(const float* data, std::size_t n);

// Times run(). observe() returns a double that depends on the result (a checksum, or the returned
// value); it runs after the timer stops and its value is passed to consume(), so the call cannot
// be optimized away and the observation is never timed. If run() modifies its own input (an
// accumulating kernel), reset the input inside run() or the results drift.
template <class Run, class Observe>
Sample measure(const Config& config, Run&& run, Observe&& observe) {
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::time_point from, Clock::time_point to) {
        return std::chrono::duration<double>(to - from).count();
    };

    for (int i = 0; i < config.warmups; ++i) {
        run();
        consume(observe());
    }

    // How many calls make one sample long enough to time reliably.
    int calls = 1;
    int repeats = config.repeats;
    {
        const auto start = Clock::now();
        run();
        const double first = seconds(start, Clock::now());
        consume(observe());
        if (first > config.slow_call_s) {
            repeats = config.slow_repeats;
        } else {
            while (calls < (1 << 20)) {
                const auto t0 = Clock::now();
                for (int c = 0; c < calls; ++c) {
                    run();
                }
                const double elapsed = seconds(t0, Clock::now());
                consume(observe());
                if (elapsed >= config.min_sample_s) {
                    break;
                }
                calls *= 2;
            }
        }
    }

    Sample sample;
    sample.calls_per_sample = calls;
    double wall_total = 0.0;
    double cpu_total = 0.0;
    for (int r = 0; r < repeats; ++r) {
        const double cpu0 = process_cpu_seconds();
        const auto t0 = Clock::now();
        for (int c = 0; c < calls; ++c) {
            run();
        }
        const auto t1 = Clock::now();
        cpu_total += process_cpu_seconds() - cpu0;
        consume(observe()); // outside the timed region
        sample.runs_s.push_back(seconds(t0, t1) / calls);
        wall_total += seconds(t0, t1);
    }
    std::vector<double> sorted = sample.runs_s;
    std::sort(sorted.begin(), sorted.end());
    sample.median_s = sorted[sorted.size() / 2];
    sample.min_s = sorted.front();
    sample.max_s = sorted.back();
    // The CPU clock advances in ticks of about 15 ms on Windows, so a ratio is only meaningful
    // over a long enough total.
    if (wall_total >= 0.1 && cpu_total > 0.0) {
        sample.cpu_over_wall = cpu_total / wall_total;
    }
    return sample;
}

// ----------------------------------------------------------------- checking
// All of these throw std::runtime_error naming `what` (and the first bad index), so a wrong
// kernel stops the benchmark instead of producing a fast, meaningless number. Call them once per
// implementation and size, before measure().
//
//   Elementwise ops, same arithmetic ........ check_bits   (exact)
//   Elementwise ops, exp/sqrt/sigmoid ....... check_close  (rel_tol about 1e-4)
//   Reductions (dot, sum) ................... check_value  (rel_tol about 1e-4 against a double
//                                             accumulation; summation order changes the result)

// Bit-for-bit equal.
void check_bits(const float* expected, const float* actual, std::size_t n, const std::string& what);
// |actual - expected| <= rel_tol * (1 + |expected|), and NaN only where NaN is expected.
void check_close(const float* expected, const float* actual, std::size_t n, double rel_tol,
                 const std::string& what);
// The same test for one number. `scale` widens the allowed error for a sum whose terms cancel:
// pass the sum of the absolute values of the terms (float rounding error is relative to that, not
// to the result), or leave it 0 when the result is the natural scale.
void check_value(double expected, double actual, double rel_tol, const std::string& what,
                 double scale = 0.0);

// ------------------------------------------------------------------- inputs

// Fixed, reproducible inputs shared by all benchmarks, so rates are comparable across them:
// fill_a gives (i % 13) / 8 - 0.75, fill_b gives (i % 17) / 8 - 1.0.
void fill_a(float* data, std::size_t n);
void fill_b(float* data, std::size_t n);

// `count` float arrays of n elements, starting 1 KB apart modulo 4 KB. Equally sized arrays tend to
// start at the same offset within a page, and addresses that agree in their low 12 bits make loads
// and stores falsely depend on each other ("4K aliasing"), which once made one op look 2x slower
// than another. Always allocate benchmark arrays with this.
class Arrays {
  public:
    Arrays(std::size_t n, int count);

    float* operator[](int i) { return storage_.data() + static_cast<std::size_t>(i) * stride_; }

  private:
    std::size_t stride_;
    std::vector<float> storage_;
};

// Thread counts to sweep for a machine with `max_threads` threads: 1, a few powers of two,
// max_threads / 2 (the physical core count on SMT CPUs), and max_threads.
std::vector<int> thread_counts(int max_threads);

// -------------------------------------------------------------- environment

std::string cpu_name();
bool avx2_supported(); // this CPU and OS can run AVX2 code (the check an AVX2 variant needs)
std::string compiler_name();
bool is_release_build(); // NDEBUG is defined

// First lines of every benchmark: the title, the machine and build (so a number can be traced to
// them), a warning if this is not a Release build, the timing settings, and any extra notes.
void print_header(const std::string& title, const Config& config,
                  const std::vector<std::string>& notes = {});

// ------------------------------------------------------------------- table

// One standard table. Columns:
//   n | impl | threads | median ms | min ms | max ms | rate | speedup A | speedup B | cpu/wall
// "rate" is work / median time (work in bytes, flops, or elements per call; `rate_scale` turns it
// into G or M). The two speedup columns divide a baseline's median time by this row's, so a value
// above 1 means faster than that baseline; pass nullptr to print "-". The usual baselines are
// "scalar" (vectorization off) and "seq" (the plain sequential kernel). A baseline whose title is
// empty has no column at all (a benchmark that compares against one thing uses only the first).
class Table {
  public:
    explicit Table(std::string rate_unit, double rate_scale = 1e9,
                   std::string baseline_a = "vs scalar", std::string baseline_b = "vs seq");

    // A blank line, "== title ==", and the column titles.
    void section(const std::string& title);
    void row(const std::string& n, const std::string& impl, const std::string& threads,
             const Sample& sample, double work, const Sample* baseline_a = nullptr,
             const Sample* baseline_b = nullptr);
    void blank();

  private:
    std::string rate_unit_;
    double rate_scale_;
    std::string baseline_a_;
    std::string baseline_b_;
};

} // namespace bench
