// The auto-tuner: the results file, the search, the hook dispatch() calls, and Tuner::tune (see
// include/mensurflow/tuning.hpp and core/tuning.hpp).
#include "core/tuning.hpp"

#include "backends/cpu/threads.hpp"
#include "core/registry.hpp"
#include "mensurflow/tuning.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace mensurflow::detail {

namespace {

thread_local TuningContext* g_context = nullptr;

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string cpu_model() {
    std::string name;
#if defined(__APPLE__)
    char brand[256] = {0};
    std::size_t size = sizeof brand;
    if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) {
        name = brand;
    }
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4];
    __cpuid(info, static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(info[0]) >= 0x80000004u) {
        char brand[49] = {0};
        for (unsigned i = 0; i < 3; ++i) {
            __cpuid(info, static_cast<int>(0x80000002u + i));
            std::memcpy(brand + 16 * i, info, 16);
        }
        name = brand;
    }
#elif defined(__x86_64__) || defined(__i386__)
    unsigned regs[4] = {0, 0, 0, 0};
    if (__get_cpuid(0x80000000u, &regs[0], &regs[1], &regs[2], &regs[3]) &&
        regs[0] >= 0x80000004u) {
        char brand[49] = {0};
        for (unsigned i = 0; i < 3; ++i) {
            __get_cpuid(0x80000002u + i, &regs[0], &regs[1], &regs[2], &regs[3]);
            std::memcpy(brand + 16 * i, regs, 16);
        }
        name = brand;
    }
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Model", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                name = line.substr(colon + 1);
                break;
            }
        }
    }
#endif
    name = trim(name);
    return name.empty() ? "unknown CPU" : name;
}

// "name=value name=value ..." -> pairs. Throws invalid_argument on a malformed pair.
TuningValues parse_values(const std::string& text) {
    TuningValues values;
    std::istringstream in(text);
    for (std::string token; in >> token;) {
        const auto equals = token.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 == token.size()) {
            throw std::invalid_argument("expected name=value, got '" + token + "'");
        }
        std::size_t used = 0;
        const std::string number = token.substr(equals + 1);
        const long long value = std::stoll(number, &used);
        if (used != number.size()) {
            throw std::invalid_argument("not an integer: '" + number + "'");
        }
        values.emplace_back(token.substr(0, equals), static_cast<std::int64_t>(value));
    }
    return values;
}

// The words of `text` separated by single spaces, so a hand-edited key still matches.
std::string normalize(const std::string& text) {
    std::istringstream in(text);
    std::string out;
    for (std::string word; in >> word;) {
        out += (out.empty() ? "" : " ") + word;
    }
    return out;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return "scalar";
    }
    std::string text;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += (i ? "x" : "") + std::to_string(shape[i]);
    }
    return text;
}

bool same_call(const TuningEntry& a, const TuningEntry& b) {
    return a.machine == b.machine && a.op == b.op && a.device == b.device && a.key == b.key;
}

// The value of `name` in `settings`, or `fallback`.
std::int64_t value_of(const TuningValues& settings, const std::string& name,
                      std::int64_t fallback) {
    for (const auto& [n, v] : settings) {
        if (n == name) {
            return v;
        }
    }
    return fallback;
}

} // namespace

// ---------------------------------------------------------------------------- results file

void TuningCache::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return; // nothing stored yet
    }
    std::vector<TuningEntry> loaded;
    std::string machine;
    int number = 0;
    for (std::string line; std::getline(in, line);) {
        ++number;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto fail = [&](const std::string& why) {
            throw std::runtime_error(path + ":" + std::to_string(number) + ": " + why);
        };
        if (line.rfind("machine ", 0) == 0) {
            machine = trim(line.substr(8));
            continue;
        }
        if (machine.empty()) {
            fail("a result before any 'machine' line");
        }
        if (line.rfind("place ", 0) == 0) {
            continue; // a device choice in the old format: dropped, and measured again when tuned
        }
        const auto arrow = line.find(" -> ");
        const auto bar = line.find(" | ", arrow == std::string::npos ? 0 : arrow);
        if (arrow == std::string::npos || bar == std::string::npos) {
            fail("expected '<op> <device> v<version> <key> -> <settings> | <ms> ms (default <ms> "
                 "ms) <mode>'");
        }
        TuningEntry entry;
        entry.machine = machine;
        try {
            std::istringstream head(line.substr(0, arrow));
            std::string version;
            head >> entry.op >> entry.device >> version;
            if (entry.device.empty() || version.size() < 2 || version[0] != 'v') {
                fail("expected '<op> <device> v<version>' at the start");
            }
            entry.version = std::stoi(version.substr(1));
            std::string rest;
            std::getline(head, rest);
            entry.key = normalize(rest);
            entry.config = parse_values(line.substr(arrow + 4, bar - arrow - 4));
            char mode[64] = {0};
            if (std::sscanf(line.c_str() + bar + 3, "%lf ms (default %lf ms) %63s", &entry.best_ms,
                            &entry.default_ms, mode) != 3) {
                fail("expected '<ms> ms (default <ms> ms) <mode>' after '|'");
            }
            entry.mode = mode;
        } catch (const std::invalid_argument& e) {
            fail(e.what());
        } catch (const std::out_of_range& e) {
            fail(std::string("number out of range: ") + e.what());
        }
        loaded.push_back(std::move(entry));
    }
    entries_ = std::move(loaded);
}

void TuningCache::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write tuning results to " + path);
    }
    write(out);
    if (!out) {
        throw std::runtime_error("cannot write tuning results to " + path);
    }
}

void TuningCache::write(std::ostream& out) const {
    out << "# mensurflow tuning results (format: src/core/tuning.hpp)\n"
        << "# <op> <device> v<version> <key> -> <settings> | <best> ms (default <ms> ms) <mode>\n"
        << "# device 'auto': the device for Execution::Auto, gpu=0 or gpu=1 (times include the "
           "copies)\n";
    std::vector<std::string> machines;
    for (const TuningEntry& entry : entries_) {
        if (std::find(machines.begin(), machines.end(), entry.machine) == machines.end()) {
            machines.push_back(entry.machine);
        }
    }
    char times[96];
    for (const std::string& machine : machines) {
        out << "\nmachine " << machine << '\n';
        for (const TuningEntry& entry : entries_) {
            if (entry.machine != machine) {
                continue;
            }
            std::snprintf(times, sizeof times, "%.4f ms (default %.4f ms) ", entry.best_ms,
                          entry.default_ms);
            out << entry.op << ' ' << entry.device << " v" << entry.version << ' ' << entry.key
                << " -> " << mensurflow::to_string(entry.config) << " | " << times << entry.mode
                << '\n';
        }
    }
}

const TuningEntry* TuningCache::find(const std::string& machine, const std::string& op,
                                     const std::string& device, int version,
                                     const std::string& key) const {
    for (const TuningEntry& entry : entries_) {
        if (entry.machine == machine && entry.op == op && entry.device == device &&
            entry.version == version && entry.key == key) {
            return &entry;
        }
    }
    return nullptr;
}

void TuningCache::store(TuningEntry entry) {
    for (TuningEntry& existing : entries_) {
        if (same_call(existing, entry)) {
            existing = std::move(entry);
            return;
        }
    }
    entries_.push_back(std::move(entry));
}

bool TuningCache::erase(const TuningEntry& entry) {
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [&](const TuningEntry& e) { return same_call(e, entry); });
    if (it == entries_.end()) {
        return false;
    }
    entries_.erase(it);
    return true;
}

// ---------------------------------------------------------------------------- keys and configs

const std::string& machine_name() {
    static const std::string name = [] {
        std::string text = cpu_model();
        Registry& registry = Registry::instance();
        for (const Device& device : registry.devices()) {
            if (device.type != DeviceType::CPU) {
                text += " | " + registry.get(device).info(device.index).name;
                break;
            }
        }
        return text;
    }();
    return name;
}

std::string call_key(const TunableKernel& tunable, const OpArgs& args, Device device) {
    const std::vector<std::int64_t> values = tunable.key(args);
    TuningValues key;
    for (std::size_t i = 0; i < values.size() && i < tunable.key_names.size(); ++i) {
        key.emplace_back(tunable.key_names[i], values[i]);
    }
    if (device.type == DeviceType::CPU) {
        key.emplace_back("threads", cpu::max_threads());
    }
    return mensurflow::to_string(key);
}

std::string placement_key(const OpArgs& args, Device data_device) {
    std::string key = std::string("data=") + device_type_name(data_device.type) + ' ' +
                      dtype_name(args.inputs.at(0).dtype());
    for (const Tensor& input : args.inputs) {
        key += ' ' + shape_text(input.shape());
    }
    return key;
}

TuningConfig to_config(const TunableKernel& tunable, const TuningValues& settings) {
    TuningConfig result = tunable.defaults();
    result.size = static_cast<int>(tunable.params.size());
    for (const auto& [name, value] : settings) {
        const auto it = std::find(tunable.params.begin(), tunable.params.end(), name);
        if (it != tunable.params.end()) {
            result.values[static_cast<std::size_t>(it - tunable.params.begin())] = value;
        }
    }
    return result;
}

TuningValues from_config(const TunableKernel& tunable, const TuningConfig& config) {
    TuningValues values;
    for (int i = 0; i < config.size && i < static_cast<int>(tunable.params.size()); ++i) {
        values.emplace_back(tunable.params[static_cast<std::size_t>(i)],
                            config.values[static_cast<std::size_t>(i)]);
    }
    return values;
}

// ---------------------------------------------------------------------------- the search

std::vector<TuningConfig> region_points(const TunableKernel& tunable, const SearchRegion& region,
                                        const OpArgs& args) {
    std::vector<TuningConfig> points;
    TuningConfig point = region.start;
    std::function<void(std::size_t)> walk = [&](std::size_t axis) {
        if (axis == region.axes.size()) {
            if (tunable.allowed(args, point)) {
                points.push_back(point);
            }
            return;
        }
        const SearchAxis& a = region.axes[axis];
        for (const std::int64_t value : a.values) {
            point.values[static_cast<std::size_t>(a.param)] = value;
            walk(axis + 1);
        }
    };
    walk(0);
    return points;
}

namespace {

// Where binary sampling starts: the region's start, each value moved to the nearest one on its
// axis; if that point is not allowed, the allowed point fewest axis steps away from it.
bool start_point(const TunableKernel& tunable, const SearchRegion& region, const OpArgs& args,
                 const std::vector<TuningConfig>& allowed, TuningConfig& start) {
    if (allowed.empty()) {
        return false;
    }
    start = region.start;
    std::vector<std::size_t> index(region.axes.size(), 0);
    for (std::size_t a = 0; a < region.axes.size(); ++a) {
        const auto& values = region.axes[a].values;
        std::int64_t& value = start.values[static_cast<std::size_t>(region.axes[a].param)];
        std::size_t nearest = 0;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (std::llabs(values[i] - value) < std::llabs(values[nearest] - value)) {
                nearest = i;
            }
        }
        index[a] = nearest;
        value = values[nearest];
    }
    if (tunable.allowed(args, start)) {
        return true;
    }
    auto position = [&](const TuningConfig& p, std::size_t a) {
        const auto& values = region.axes[a].values;
        return static_cast<std::size_t>(
            std::find(values.begin(), values.end(),
                      p.values[static_cast<std::size_t>(region.axes[a].param)]) -
            values.begin());
    };
    std::size_t best_distance = std::numeric_limits<std::size_t>::max();
    for (const TuningConfig& p : allowed) {
        std::size_t distance = 0;
        for (std::size_t a = 0; a < region.axes.size(); ++a) {
            const std::size_t at = position(p, a);
            distance += at > index[a] ? at - index[a] : index[a] - at;
        }
        if (distance < best_distance) {
            best_distance = distance;
            start = p;
        }
    }
    return true;
}

} // namespace

void search(const TunableKernel& tunable, const OpArgs& args,
            const std::vector<SearchRegion>& space, SearchMethod method, int max_sweeps, int levels,
            const Measure& measure) {
    for (const SearchRegion& region : space) {
        const std::vector<TuningConfig> allowed = region_points(tunable, region, args);
        if (method == SearchMethod::Exhaustive) {
            if (!allowed.empty()) {
                measure(allowed);
            }
            continue;
        }
        TuningConfig x;
        if (!start_point(tunable, region, args, allowed, x)) {
            continue; // no allowed point in this region for this call
        }
        double fx = measure({x})[0];
        for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
            bool moved = false;
            for (const SearchAxis& axis : region.axes) {
                const auto param = static_cast<std::size_t>(axis.param);
                // The line through x along this axis: the values the other arguments allow.
                std::vector<TuningConfig> line;
                for (const std::int64_t value : axis.values) {
                    TuningConfig p = x;
                    p.values[param] = value;
                    if (tunable.allowed(args, p)) {
                        line.push_back(p);
                    }
                }
                if (line.size() < 2) {
                    continue;
                }
                // Binary sampling: level 0 is the two ends of the line, level 1 its middle, level 2
                // the quarter points, and so on (positions rounded to the nearest value). Every
                // sample is measured and the best is kept; nothing is ruled out by a comparison,
                // so the samples do not depend on each other and are measured as one batch.
                std::vector<bool> sampled(line.size(), false);
                std::vector<std::size_t> picked;
                std::vector<TuningConfig> samples;
                const double last = static_cast<double>(line.size() - 1);
                for (int level = 0; level <= levels; ++level) {
                    std::vector<std::size_t> picks;
                    if (level == 0) {
                        picks = {0, line.size() - 1};
                    } else {
                        const long parts = 1L << level;
                        for (long j = 1; j < parts; j += 2) {
                            picks.push_back(static_cast<std::size_t>(std::lround(
                                last * static_cast<double>(j) / static_cast<double>(parts))));
                        }
                    }
                    for (const std::size_t i : picks) {
                        if (!sampled[i]) {
                            sampled[i] = true;
                            picked.push_back(i);
                            samples.push_back(line[i]);
                        }
                    }
                    if (std::all_of(sampled.begin(), sampled.end(), [](bool v) { return v; })) {
                        break;
                    }
                }
                const std::vector<double> times = measure(samples);
                std::size_t best_at = line.size();
                double best_here = fx;
                for (std::size_t j = 0; j < picked.size(); ++j) {
                    if (times[j] < best_here) {
                        best_here = times[j];
                        best_at = picked[j];
                    }
                }
                if (best_at < line.size() && line[best_at] != x) {
                    x = line[best_at];
                    fx = best_here;
                    moved = true;
                }
            }
            if (!moved) {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------- dispatch side

TuningScope::TuningScope(TuningContext context) : previous_(g_context), context_(context) {
    g_context = &context_;
}

TuningScope::~TuningScope() { g_context = previous_; }

bool tuning_active() { return g_context != nullptr; }

Device apply_tuning(OpId op, Execution execution, Device data_device, OpArgs& args) {
    const TuningContext& context = *g_context;
    const TuningCache* cache = context.cache;
    Execution chosen = execution;
    if (execution == Execution::Auto && cache != nullptr && !args.inputs.empty()) {
        if (const TuningEntry* entry = cache->find(machine_name(), op_name(op), kAutoDevice, 1,
                                                   placement_key(args, data_device))) {
            chosen = value_of(entry->config, "gpu", 0) != 0 ? Execution::GPU : Execution::CPU;
        }
    }
    const Device target = execution_device(chosen, data_device);
    if (context.record != nullptr) {
        context.record->push_back({op, execution, data_device, target, args});
    }
    if (cache != nullptr) {
        const Tensor& source = args.inputs.empty() ? args.outputs[0] : args.inputs[0];
        const TunableKernel* tunable =
            Registry::instance().dispatcher().tunable({op, target.type, source.dtype()});
        if (tunable != nullptr) {
            if (const TuningEntry* entry =
                    cache->find(machine_name(), tunable->name, device_type_name(target.type),
                                tunable->version, call_key(*tunable, args, target))) {
                args.tuning = to_config(*tunable, entry->config);
            }
        }
    }
    return target;
}

} // namespace mensurflow::detail

// ---------------------------------------------------------------------------- public API

namespace mensurflow {

std::string to_string(const TuningValues& values) {
    std::string text;
    for (const auto& [name, value] : values) {
        text += (text.empty() ? "" : " ") + name + "=" + std::to_string(value);
    }
    return text;
}

namespace {

using detail::OpArgs;
using detail::RecordedCall;
using detail::Registry;
using detail::SearchRegion;
using detail::TunableKernel;
using detail::TuningConfig;
using detail::TuningEntry;

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Waits until every output is computed (a GPU may still be working on it).
void finish(const std::vector<Tensor>& tensors) {
    for (const Tensor& t : tensors) {
        if (t.defined() && t.device().type != DeviceType::CPU) {
            Registry::instance().get(t.device()).stream(t.device().index).synchronize();
        }
    }
}

// Median milliseconds per call of `run`, which must wait for its own work to finish. Each sample
// repeats the call until it lasts about a millisecond, so short calls are not lost in clock noise.
// With `give_up_ms` > 0, timing stops once there are min_samples samples and even the fastest is
// above it: the point is already clearly slower, and the median so far is returned (`*stopped` set).
double median_ms(const std::function<void()>& run, const TuneOptions& options,
                 const std::function<double()>& give_up_ms = nullptr, bool* stopped = nullptr) {
    run(); // warm-up: caches, compiled pipelines
    auto first = Clock::now();
    run();
    const double one = std::max(seconds_since(first), 1e-7);
    const int repeats = std::max(1, static_cast<int>(1e-3 / one));
    std::vector<double> samples;
    double fastest = std::numeric_limits<double>::infinity();
    const auto start = Clock::now();
    while (static_cast<int>(samples.size()) < options.min_samples ||
           seconds_since(start) < options.min_time_s) {
        const auto t0 = Clock::now();
        for (int i = 0; i < repeats; ++i) {
            run();
        }
        samples.push_back(seconds_since(t0) / repeats);
        fastest = std::min(fastest, samples.back());
        if (samples.size() >= 1000) {
            break;
        }
        if (give_up_ms && static_cast<int>(samples.size()) >= options.min_samples &&
            fastest * 1e3 > give_up_ms()) {
            if (stopped != nullptr) {
                *stopped = true;
            }
            break;
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2] * 1e3;
}

// The call's arguments on `device` (the inputs copied there if they live elsewhere) with fresh
// outputs, so measuring never overwrites the algorithm's tensors.
OpArgs on_device(const OpArgs& args, Device device) {
    OpArgs copy = args;
    for (Tensor& input : copy.inputs) {
        if (input.device() != device) {
            input = input.to(device);
        }
    }
    for (Tensor& out : copy.outputs) {
        out = Tensor::empty(out.shape(), out.dtype(), device);
    }
    return copy;
}

std::vector<std::vector<float>> read_outputs(const OpArgs& args) {
    std::vector<std::vector<float>> values;
    for (const Tensor& out : args.outputs) {
        values.push_back(out.dtype() == DType::F32 ? out.to_vector<float>() : std::vector<float>{});
    }
    return values;
}

// Within float rounding of the default's result: each element within 1e-3 of the largest
// magnitude in that output (different tilings add in different orders).
bool agrees(const std::vector<std::vector<float>>& expected,
            const std::vector<std::vector<float>>& actual) {
    for (std::size_t o = 0; o < expected.size(); ++o) {
        const auto& e = expected[o];
        const auto& a = actual[o];
        if (e.size() != a.size()) {
            return false;
        }
        double scale = 0;
        for (float v : e) {
            scale = std::max(scale, std::abs(static_cast<double>(v)));
        }
        const double tolerance = 1e-3 * scale + 1e-6;
        for (std::size_t i = 0; i < e.size(); ++i) {
            const double difference = std::abs(static_cast<double>(a[i]) - e[i]);
            if (!(difference <= tolerance) && !(std::isnan(e[i]) && std::isnan(a[i]))) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

const char* search_method_name(SearchMethod method) {
    return method == SearchMethod::Exhaustive ? "exhaustive" : "binary";
}

SearchMethod parse_search_method(const std::string& name) {
    if (name == "binary") {
        return SearchMethod::BinarySampling;
    }
    if (name == "exhaustive") {
        return SearchMethod::Exhaustive;
    }
    throw std::invalid_argument("unknown search method '" + name + "' (binary or exhaustive)");
}

std::vector<TunableKernelInfo> tunable_kernels() {
    Registry& registry = Registry::instance();
    std::vector<TunableKernelInfo> list;
    std::set<std::pair<std::string, DeviceType>> seen;
    for (const auto& [kernel, t] : registry.dispatcher().tunables()) {
        if (registry.find(kernel.device) == nullptr ||
            !seen.insert({t.name, kernel.device}).second) {
            continue; // no such device here, or the same kernel for another dtype
        }
        TunableKernelInfo info;
        info.name = t.name;
        info.device = device_type_name(kernel.device);
        info.key = t.key_names;
        if (kernel.device == DeviceType::CPU) {
            info.key.push_back("threads");
        }
        info.params = t.params;
        list.push_back(info);
    }
    return list;
}

Algorithm::Algorithm(Function function) : function_(std::move(function)) {
    if (!function_) {
        throw std::invalid_argument("Algorithm: empty function");
    }
}

std::vector<Tensor> Algorithm::run(const std::vector<Tensor>& inputs) const {
    return function_(inputs);
}

std::vector<Tensor> Algorithm::run(const std::vector<Tensor>& inputs, const Tuner& tuner) const {
    detail::TuningScope scope({tuner.cache_.get(), nullptr});
    return function_(inputs);
}

Tuner::Tuner(std::string path)
    : path_(std::move(path)), cache_(std::make_shared<detail::TuningCache>()) {
    cache_->load(path_);
}

void Tuner::save() const { cache_->save(path_); }

std::size_t Tuner::size() const {
    const auto& entries = cache_->entries();
    return static_cast<std::size_t>(
        std::count_if(entries.begin(), entries.end(),
                      [](const TuningEntry& e) { return e.machine == detail::machine_name(); }));
}

std::string Tuner::describe() const {
    detail::TuningCache mine;
    for (const TuningEntry& entry : cache_->entries()) {
        if (entry.machine == detail::machine_name()) {
            mine.store(entry);
        }
    }
    std::ostringstream text;
    mine.write(text);
    return text.str();
}

TuneReport Tuner::tune(const Algorithm& algorithm, const std::vector<Tensor>& inputs,
                       const TuneOptions& options) {
    Registry& registry = Registry::instance();
    const std::string& machine = detail::machine_name();
    TuneReport report;

    // What was stored before and what tuning adds, to restore if the tuned algorithm is slower.
    std::vector<TuningEntry> replaced;
    std::vector<TuningEntry> added;
    auto keep = [&](const TunedCall& result, const std::string& op, const std::string& device,
                    int version) {
        TuningEntry entry;
        entry.machine = machine;
        entry.op = op;
        entry.device = device;
        entry.version = version;
        entry.key = result.key;
        entry.config = result.best_settings;
        entry.best_ms = result.best_ms;
        entry.default_ms = result.default_ms;
        entry.mode = "tested";
        if (const TuningEntry* old = cache_->find(machine, op, device, version, entry.key)) {
            replaced.push_back(*old);
        }
        added.push_back(entry);
        cache_->store(entry);
        if (options.verbose) {
            const auto stopped = std::count_if(result.points.begin(), result.points.end(),
                                               [](const TunedPoint& p) { return p.stopped; });
            std::cout << "  " << result.op << " on " << result.device << " " << result.key << ": "
                      << to_string(result.best_settings) << " " << result.best_ms << " ms (default "
                      << result.default_ms << " ms; " << result.points.size() << " of "
                      << result.space_size << " points measured";
            if (stopped > 0) {
                std::cout << ", " << stopped << " stopped early";
            }
            std::cout << ", " << result.search_s << " s)\n";
        }
        report.calls.push_back(result);
    };
    // A result already stored is reported as it is, unless options.retune.
    auto reuse = [&](TunedCall& result, const std::string& op, const std::string& device,
                     int version) {
        const TuningEntry* entry = cache_->find(machine, op, device, version, result.key);
        if (entry == nullptr || options.retune) {
            return false;
        }
        result.reused = true;
        result.best_settings = entry->config;
        result.best_ms = entry->best_ms;
        result.default_ms = entry->default_ms;
        report.calls.push_back(result);
        return true;
    };

    // The GPU raises its clock only after some time under load; keep it busy for a second before
    // measuring on it, or the first configurations are measured at a lower clock.
    std::set<std::pair<int, int>> warmed;
    auto warm = [&](Device device, const std::function<void()>& run) {
        if (device.type != DeviceType::CPU &&
            warmed.insert({static_cast<int>(device.type), device.index}).second) {
            const auto start = Clock::now();
            while (seconds_since(start) < 1.0) {
                run();
            }
        }
    };

    std::set<std::tuple<std::string, std::string, std::string>> seen; // op, device, key

    // Kernel settings: the call's kernel on `device`, if it is tunable, searched over its space.
    auto tune_kernel = [&](const RecordedCall& call, Device device) {
        const Tensor& source =
            call.args.inputs.empty() ? call.args.outputs[0] : call.args.inputs[0];
        const detail::KernelKey kernel_key{call.op, device.type, source.dtype()};
        const TunableKernel* tunable = registry.dispatcher().tunable(kernel_key);
        if (tunable == nullptr) {
            return;
        }
        const std::string device_type = device_type_name(device.type);
        TunedCall result;
        result.op = tunable->name;
        result.device = to_string(device);
        result.key = detail::call_key(*tunable, call.args, device);
        if (!seen.insert({result.op, device_type, result.key}).second) {
            return; // the same call again
        }
        const TuningConfig defaults = tunable->defaults();
        result.default_settings = detail::from_config(*tunable, defaults);
        if (reuse(result, result.op, device_type, tunable->version)) {
            return;
        }
        const std::vector<SearchRegion> space = tunable->space(call.args);
        if (space.empty()) {
            return; // nothing to tune for this call
        }
        for (const SearchRegion& region : space) {
            result.space_size += detail::region_points(*tunable, region, call.args).size();
        }

        const detail::KernelFn kernel = registry.dispatcher().lookup(kernel_key);
        detail::Stream& stream = registry.get(device).stream(device.index);
        OpArgs args = on_device(call.args, device);
        auto run_with = [&](const TuningConfig& config) {
            args.tuning = config;
            kernel(args, stream);
            stream.synchronize();
        };
        warm(device, [&] { run_with(TuningConfig{}); });
        run_with(TuningConfig{});
        const auto expected = read_outputs(args);

        // Early stop: a point already slower than give_up x the best median so far.
        std::function<double()> give_up_ms;
        if (options.give_up > 0) {
            give_up_ms = [&] { return result.best_ms * options.give_up; };
        }

        // The objective: each point is checked against the default's result (infinite when it
        // disagrees), then timed. It is measured once; the search may ask again, for free.
        std::map<TuningConfig, double> measured;
        std::vector<std::pair<double, TuningConfig>> timed; // the points that agreed
        TuningConfig best = defaults;
        result.best_ms = std::numeric_limits<double>::infinity();
        const auto search_start = Clock::now();
        auto measure = [&](const std::vector<TuningConfig>& points) {
            std::vector<double> times;
            for (const TuningConfig& config : points) {
                if (const auto it = measured.find(config); it != measured.end()) {
                    times.push_back(it->second);
                    continue;
                }
                TunedPoint point;
                point.settings = detail::from_config(*tunable, config);
                double ms = std::numeric_limits<double>::infinity();
                run_with(config);
                if (!agrees(expected, read_outputs(args))) {
                    point.rejected = true;
                } else {
                    ms = median_ms([&] { run_with(config); }, options, give_up_ms, &point.stopped);
                    point.ms = ms;
                    timed.emplace_back(ms, config);
                    if (ms < result.best_ms) {
                        result.best_ms = ms;
                        best = config;
                    }
                }
                measured[config] = ms;
                result.points.push_back(point);
                times.push_back(ms);
            }
            return times;
        };
        result.default_ms = measure({defaults})[0];
        detail::search(*tunable, call.args, space, options.search, options.max_sweeps,
                       options.levels, measure);
        result.search_s = seconds_since(search_start);
        if (!std::isfinite(result.best_ms)) { // even the default disagreed with itself
            result.best_ms = result.default_ms;
            best = defaults;
        }

        // Confirmation: the points were measured seconds apart, and the clock (a GPU's above all)
        // or the load can change in between. The default and the fastest few are timed again,
        // taking turns, and the fastest of those times is kept.
        if (options.confirm > 0) {
            std::sort(timed.begin(), timed.end());
            std::vector<TuningConfig> finalists = {defaults};
            for (std::size_t i = 0;
                 i < timed.size() && finalists.size() <= static_cast<std::size_t>(options.confirm);
                 ++i) {
                if (timed[i].second != defaults) {
                    finalists.push_back(timed[i].second);
                }
            }
            std::vector<std::vector<double>> times(finalists.size());
            for (int round = 0; round < 3; ++round) {
                for (std::size_t i = 0; i < finalists.size(); ++i) {
                    // Rotate the order, so no point is always timed first.
                    const std::size_t at = (i + static_cast<std::size_t>(round)) % finalists.size();
                    times[at].push_back(median_ms([&] { run_with(finalists[at]); }, options));
                }
            }
            std::size_t winner = 0;
            std::vector<double> confirmed;
            for (std::size_t i = 0; i < finalists.size(); ++i) {
                std::sort(times[i].begin(), times[i].end());
                confirmed.push_back(times[i][times[i].size() / 2]);
                if (confirmed[i] < confirmed[winner]) {
                    winner = i;
                }
            }
            result.default_ms = confirmed[0];
            result.best_ms = confirmed[winner];
            best = finalists[winner];
        }
        result.best_settings = detail::from_config(*tunable, best);
        keep(result, tunable->name, device_type, tunable->version);
    };

    // Device choice for an Execution::Auto call: a setting with two values, gpu=0 (the CPU, what
    // Auto does without a result) and gpu=1. Each device is timed through dispatch() with the
    // copies there and back, after its own kernel settings were tuned, and must give the CPU's
    // result.
    auto choose_device = [&](const RecordedCall& call) {
        if (call.args.inputs.empty()) {
            return; // random factories have no inputs to key on
        }
        TunedCall result;
        result.op = detail::op_name(call.op);
        result.device = detail::kAutoDevice;
        result.key = detail::placement_key(call.args, call.data_device);
        if (!seen.insert({result.op, result.device, result.key}).second) {
            return;
        }
        result.default_settings = {{"gpu", 0}};
        if (reuse(result, result.op, result.device, 1)) {
            return;
        }
        const DType dtype = call.args.inputs[0].dtype();
        std::vector<std::pair<Execution, Device>> choices;
        for (const Execution execution : {Execution::CPU, Execution::GPU}) {
            try {
                const Device target = detail::execution_device(execution, call.data_device);
                if (registry.dispatcher().find({call.op, target.type, dtype}) != nullptr) {
                    choices.push_back({execution, target});
                }
            } catch (const std::runtime_error&) {
                // no GPU on this machine
            }
        }
        if (choices.size() < 2 || choices[0].first != Execution::CPU) {
            return; // nothing to choose from
        }
        result.space_size = choices.size();

        std::vector<std::vector<float>> expected;
        const auto search_start = Clock::now();
        for (const auto& [execution, target] : choices) {
            tune_kernel(call, target); // so each device is timed at its best
            OpArgs args = on_device(call.args, call.data_device);
            auto run = [&, execution = execution] {
                detail::TuningScope scope({cache_.get(), nullptr});
                detail::dispatch(call.op, args, execution);
                finish(args.outputs);
            };
            warm(target, run);
            run();
            TunedPoint point;
            point.settings = {{"gpu", execution == Execution::GPU ? 1 : 0}};
            if (execution == Execution::CPU) {
                expected = read_outputs(args);
            } else if (!agrees(expected, read_outputs(args))) {
                point.rejected = true;
                result.points.push_back(point);
                continue;
            }
            point.ms = median_ms(run, options);
            result.points.push_back(point);
            if (execution == Execution::CPU) {
                result.default_ms = point.ms;
            }
            if (result.best_settings.empty() || point.ms < result.best_ms) {
                result.best_ms = point.ms;
                result.best_settings = point.settings;
            }
        }
        result.search_s = seconds_since(search_start);
        keep(result, result.op, result.device, 1);
    };

    // 1. Run once, recording every op call.
    std::vector<RecordedCall> recorded;
    {
        detail::TuningScope scope({nullptr, &recorded});
        finish(algorithm.run(inputs));
    }

    // 2. Kernel settings of each call, on the device it ran on.
    for (const RecordedCall& call : recorded) {
        tune_kernel(call, call.device);
    }

    // 3. The device of each Execution::Auto call.
    for (const RecordedCall& call : recorded) {
        if (call.execution == Execution::Auto) {
            choose_device(call);
        }
    }

    // 4. Tested: the whole algorithm with the default and the tuned settings.
    TuneOptions whole = options;
    whole.min_time_s = std::max(options.min_time_s, 0.2);
    report.default_ms = median_ms([&] { finish(algorithm.run(inputs)); }, whole);
    report.tuned_ms = median_ms([&] { finish(algorithm.run(inputs, *this)); }, whole);
    if (!added.empty() && report.tuned_ms > report.default_ms * 1.02) {
        report.kept = false;
        for (const TuningEntry& entry : added) {
            cache_->erase(entry);
        }
        for (const TuningEntry& entry : replaced) {
            cache_->store(entry);
        }
    }
    if (options.save) {
        save();
    }
    return report;
}

} // namespace mensurflow
