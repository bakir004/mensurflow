// The auto-tuner: the results file, applying stored results in dispatch(), and Tuner::tune (see
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

std::string join(const NamedValues& values) {
    std::string text;
    for (const auto& [name, value] : values) {
        text += (text.empty() ? "" : " ") + name + "=" + std::to_string(value);
    }
    return text;
}

// "name=value name=value ..." -> pairs. Throws invalid_argument on a malformed pair.
NamedValues parse_values(const std::string& text) {
    NamedValues values;
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

std::vector<std::int64_t> parse_shape(const std::string& text) {
    std::vector<std::int64_t> shape;
    if (text == "scalar") {
        return shape;
    }
    std::stringstream in(text);
    for (std::string dim; std::getline(in, dim, 'x');) {
        std::size_t used = 0;
        shape.push_back(static_cast<std::int64_t>(std::stoll(dim, &used)));
        if (used != dim.size()) {
            throw std::invalid_argument("not a shape: '" + text + "'");
        }
    }
    return shape;
}

// "place <op> <dtype> <data device> <shape> ... -> <cpu|gpu> | <ms> ms (<exec> <ms> ms, ...) <mode>"
PlacementEntry parse_placement(const std::string& line, const std::string& machine) {
    const auto arrow = line.find(" -> ");
    const auto bar = line.find(" | ", arrow == std::string::npos ? 0 : arrow);
    if (arrow == std::string::npos || bar == std::string::npos) {
        throw std::invalid_argument("expected 'place <op> <dtype> <device> <shapes> -> <cpu|gpu> | "
                                    "<ms> ms (<cpu|gpu> <ms> ms, ...) <mode>'");
    }
    PlacementEntry entry;
    entry.machine = machine;
    std::istringstream head(line.substr(0, arrow));
    std::string word;
    head >> word >> entry.op >> entry.dtype >> entry.data_device;
    if (entry.data_device.empty()) {
        throw std::invalid_argument("expected 'place <op> <dtype> <device> <shapes>'");
    }
    for (std::string shape; head >> shape;) {
        entry.shapes.push_back(parse_shape(shape));
    }
    entry.chosen = trim(line.substr(arrow + 4, bar - arrow - 4));
    if (entry.chosen != "cpu" && entry.chosen != "gpu") {
        throw std::invalid_argument("the chosen execution must be cpu or gpu, got '" +
                                    entry.chosen + "'");
    }
    const std::string tail = line.substr(bar + 3);
    const auto open = tail.find(" ms (");
    const auto close = tail.find(") ", open == std::string::npos ? 0 : open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::invalid_argument("expected '<ms> ms (<cpu|gpu> <ms> ms, ...) <mode>' after '|'");
    }
    entry.best_ms = std::stod(tail.substr(0, open));
    std::stringstream times(tail.substr(open + 5, close - open - 5));
    for (std::string item; std::getline(times, item, ',');) {
        std::istringstream one(item);
        std::string execution, unit;
        double ms = 0;
        if (!(one >> execution >> ms >> unit) || unit != "ms") {
            throw std::invalid_argument("expected '<cpu|gpu> <ms> ms', got '" + trim(item) + "'");
        }
        entry.times.emplace_back(execution, ms);
    }
    entry.mode = trim(tail.substr(close + 2));
    return entry;
}

bool same_call(const TuningEntry& a, const TuningEntry& b) {
    return a.machine == b.machine && a.op == b.op && a.device == b.device && a.key == b.key;
}

} // namespace

// ---------------------------------------------------------------------------- results file

void TuningCache::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return; // nothing stored yet
    }
    std::vector<TuningEntry> loaded;
    std::vector<PlacementEntry> placements;
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
            try {
                placements.push_back(parse_placement(line, machine));
            } catch (const std::invalid_argument& e) {
                fail(e.what());
            } catch (const std::out_of_range& e) {
                fail(std::string("number out of range: ") + e.what());
            }
            continue;
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
            entry.key = parse_values(rest);
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
    placements_ = std::move(placements);
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
    out << "# mensurflow tuning results (format: src/core/tuning.hpp). Kernel settings:\n"
        << "# <op> <device> v<version> <key> -> <settings> | <best> ms (default <ms> ms) <mode>\n"
        << "# Device choices for Execution::Auto (times include the copies there and back):\n"
        << "# place <op> <dtype> <data device> <shapes> -> <cpu|gpu> | <best> ms (<exec> <ms> ms, "
           "...) <mode>\n";
    std::vector<std::string> machines;
    auto add_machine = [&](const std::string& machine) {
        if (std::find(machines.begin(), machines.end(), machine) == machines.end()) {
            machines.push_back(machine);
        }
    };
    for (const TuningEntry& entry : entries_) {
        add_machine(entry.machine);
    }
    for (const PlacementEntry& entry : placements_) {
        add_machine(entry.machine);
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
            out << entry.op << ' ' << entry.device << " v" << entry.version << ' '
                << join(entry.key) << " -> " << join(entry.config) << " | " << times << entry.mode
                << '\n';
        }
        for (const PlacementEntry& entry : placements_) {
            if (entry.machine != machine) {
                continue;
            }
            out << "place " << entry.op << ' ' << entry.dtype << ' ' << entry.data_device;
            for (const auto& shape : entry.shapes) {
                out << ' ' << shape_text(shape);
            }
            std::snprintf(times, sizeof times, "%.4f ms (", entry.best_ms);
            out << " -> " << entry.chosen << " | " << times;
            for (std::size_t i = 0; i < entry.times.size(); ++i) {
                std::snprintf(times, sizeof times, "%s%s %.4f ms", i ? ", " : "",
                              entry.times[i].first.c_str(), entry.times[i].second);
                out << times;
            }
            out << ") " << entry.mode << '\n';
        }
    }
}

const TuningEntry* TuningCache::find(const std::string& machine, const std::string& op,
                                     const std::string& device, int version,
                                     const NamedValues& key) const {
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

bool same_placement(const PlacementEntry& a, const PlacementEntry& b) {
    return a.machine == b.machine && a.op == b.op && a.dtype == b.dtype &&
           a.data_device == b.data_device && a.shapes == b.shapes;
}

const PlacementEntry* TuningCache::find_placement(const PlacementEntry& key) const {
    for (const PlacementEntry& entry : placements_) {
        if (same_placement(entry, key)) {
            return &entry;
        }
    }
    return nullptr;
}

void TuningCache::store_placement(PlacementEntry entry) {
    for (PlacementEntry& existing : placements_) {
        if (same_placement(existing, entry)) {
            existing = std::move(entry);
            return;
        }
    }
    placements_.push_back(std::move(entry));
}

bool TuningCache::erase_placement(const PlacementEntry& entry) {
    const auto it = std::find_if(placements_.begin(), placements_.end(),
                                 [&](const PlacementEntry& e) { return same_placement(e, entry); });
    if (it == placements_.end()) {
        return false;
    }
    placements_.erase(it);
    return true;
}

PlacementEntry placement_key(OpId op, const OpArgs& args, Device data_device) {
    PlacementEntry key;
    key.machine = machine_name();
    key.op = op_name(op);
    key.dtype = args.inputs.empty() ? "" : dtype_name(args.inputs[0].dtype());
    key.data_device = device_type_name(data_device.type);
    for (const Tensor& input : args.inputs) {
        key.shapes.push_back(input.shape());
    }
    return key;
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

NamedValues call_key(const TunableKernel& tunable, const OpArgs& args, Device device) {
    const std::vector<std::int64_t> values = tunable.key(args);
    NamedValues key;
    for (std::size_t i = 0; i < values.size() && i < tunable.key_names.size(); ++i) {
        key.emplace_back(tunable.key_names[i], values[i]);
    }
    if (device.type == DeviceType::CPU) {
        key.emplace_back("threads", cpu::max_threads());
    }
    return key;
}

TuningConfig to_config(const TunableKernel& tunable, const NamedValues& config) {
    TuningConfig result = tunable.defaults();
    result.size = static_cast<int>(tunable.params.size());
    for (const auto& [name, value] : config) {
        const auto it = std::find(tunable.params.begin(), tunable.params.end(), name);
        if (it != tunable.params.end()) {
            result.values[static_cast<std::size_t>(it - tunable.params.begin())] = value;
        }
    }
    return result;
}

NamedValues from_config(const TunableKernel& tunable, const TuningConfig& config) {
    NamedValues values;
    for (int i = 0; i < config.size && i < static_cast<int>(tunable.params.size()); ++i) {
        values.emplace_back(tunable.params[static_cast<std::size_t>(i)],
                            config.values[static_cast<std::size_t>(i)]);
    }
    return values;
}

// ---------------------------------------------------------------------------- the search space

TuningConfig with_values(const TuningConfig& base, const TuningPoint& values) {
    TuningConfig config = base;
    config.values = values;
    return config;
}

// Every allowed point of a region, in order (the last axis changes fastest).
std::vector<TuningPoint> region_points(const TunableKernel& tunable, const SearchRegion& region,
                                       const OpArgs& args) {
    std::vector<TuningPoint> points;
    TuningPoint point = region.start.values;
    std::function<void(std::size_t)> walk = [&](std::size_t axis) {
        if (axis == region.axes.size()) {
            if (tunable.allowed(args, with_values(region.start, point))) {
                points.push_back(point);
            }
            return;
        }
        const SearchAxis& a = region.axes[axis];
        for (const std::int64_t value : a.values) {
            point[static_cast<std::size_t>(a.param)] = value;
            walk(axis + 1);
        }
    };
    walk(0);
    return points;
}

// Where binary sampling starts: the region's start, each value moved to the nearest one on its
// axis; if that point is not allowed, the allowed point fewest axis steps away from it.
namespace {

bool start_point(const TunableKernel& tunable, const SearchRegion& region, const OpArgs& args,
                 const std::vector<TuningPoint>& allowed, TuningPoint& start) {
    if (allowed.empty()) {
        return false;
    }
    start = region.start.values;
    std::vector<std::size_t> index(region.axes.size(), 0);
    for (std::size_t a = 0; a < region.axes.size(); ++a) {
        const auto& values = region.axes[a].values;
        std::int64_t& value = start[static_cast<std::size_t>(region.axes[a].param)];
        std::size_t nearest = 0;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (std::llabs(values[i] - value) < std::llabs(values[nearest] - value)) {
                nearest = i;
            }
        }
        index[a] = nearest;
        value = values[nearest];
    }
    if (tunable.allowed(args, with_values(region.start, start))) {
        return true;
    }
    auto position = [&](const TuningPoint& p, std::size_t a) {
        const auto& values = region.axes[a].values;
        return static_cast<std::size_t>(
            std::find(values.begin(), values.end(),
                      p[static_cast<std::size_t>(region.axes[a].param)]) -
            values.begin());
    };
    std::size_t best_distance = std::numeric_limits<std::size_t>::max();
    for (const TuningPoint& p : allowed) {
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
        const std::vector<TuningPoint> allowed = region_points(tunable, region, args);
        if (method == SearchMethod::Exhaustive) {
            std::vector<Probe> grid;
            for (const TuningPoint& p : allowed) {
                grid.push_back({with_values(region.start, p), region.name, "grid", 0, 0});
            }
            if (!grid.empty()) {
                measure(grid);
            }
            continue;
        }
        TuningPoint x;
        if (!start_point(tunable, region, args, allowed, x)) {
            continue; // no allowed point in this region for this call
        }
        double fx = measure({{with_values(region.start, x), region.name, "start", 0, 0}})[0];
        for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
            bool moved = false;
            for (const SearchAxis& axis : region.axes) {
                const auto param = static_cast<std::size_t>(axis.param);
                const std::string& name = tunable.params[param];
                // The line through x along this axis: the values the other arguments allow.
                std::vector<TuningPoint> line;
                for (const std::int64_t value : axis.values) {
                    TuningPoint p = x;
                    p[param] = value;
                    if (tunable.allowed(args, with_values(region.start, p))) {
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
                std::vector<Probe> probes;
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
                            probes.push_back({with_values(region.start, line[i]), region.name, name,
                                              sweep, level});
                        }
                    }
                    if (std::all_of(sampled.begin(), sampled.end(), [](bool v) { return v; })) {
                        break;
                    }
                }
                const std::vector<double> times = measure(probes);
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

Execution auto_execution(OpId op, const OpArgs& args, Device data_device) {
    TuningContext* context = g_context;
    if (context == nullptr || args.inputs.empty()) {
        return Execution::Auto; // random factories have no inputs to key on
    }
    if (context->record_auto != nullptr) {
        context->record_auto->push_back({op, data_device, args});
    }
    if (context->cache != nullptr) {
        if (const PlacementEntry* entry =
                context->cache->find_placement(placement_key(op, args, data_device))) {
            return entry->chosen == "gpu" ? Execution::GPU : Execution::CPU;
        }
    }
    return Execution::Auto;
}

void apply_tuning(KernelKey kernel, const TunableKernel& tunable, Device device, OpArgs& args) {
    TuningContext* context = g_context;
    if (context == nullptr) {
        return;
    }
    if (context->record != nullptr) {
        context->record->push_back({kernel, device, args});
    }
    if (context->cache != nullptr) {
        const TuningEntry* entry =
            context->cache->find(machine_name(), tunable.name, device_type_name(device.type),
                                 tunable.version, call_key(tunable, args, device));
        if (entry != nullptr) {
            args.tuning = to_config(tunable, entry->config);
        }
    }
}

} // namespace mensurflow::detail

// ---------------------------------------------------------------------------- public API

namespace mensurflow {

namespace {

using detail::NamedValues;
using detail::OpArgs;
using detail::RecordedCall;
using detail::Registry;
using detail::TunableKernel;
using detail::TuningConfig;
using detail::TuningEntry;

using Clock = std::chrono::steady_clock;

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

std::string text_of(const NamedValues& values) {
    std::string text;
    for (const auto& [name, value] : values) {
        text += (text.empty() ? "" : " ") + name + "=" + std::to_string(value);
    }
    return text;
}

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

// Fresh outputs on the call's device, so measuring never overwrites the algorithm's tensors.
OpArgs with_fresh_outputs(const OpArgs& args) {
    OpArgs copy = args;
    for (Tensor& out : copy.outputs) {
        out = Tensor::empty(out.shape(), out.dtype(), out.device());
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
    const std::string& machine = detail::machine_name();
    const auto& entries = cache_->entries();
    const auto& places = cache_->placements();
    return static_cast<std::size_t>(
        std::count_if(entries.begin(), entries.end(),
                      [&](const TuningEntry& e) { return e.machine == machine; }) +
        std::count_if(places.begin(), places.end(),
                      [&](const detail::PlacementEntry& e) { return e.machine == machine; }));
}

std::string Tuner::describe() const {
    detail::TuningCache mine;
    for (const TuningEntry& entry : cache_->entries()) {
        if (entry.machine == detail::machine_name()) {
            mine.store(entry);
        }
    }
    for (const detail::PlacementEntry& entry : cache_->placements()) {
        if (entry.machine == detail::machine_name()) {
            mine.store_placement(entry);
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
    std::vector<detail::PlacementEntry> replaced_places;
    std::vector<detail::PlacementEntry> added_places;

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

    // Kernel settings: every candidate of each distinct tunable call, timed on that call.
    std::set<std::tuple<std::string, std::string, NamedValues>> seen;
    auto tune_kernels = [&](const std::vector<RecordedCall>& recorded) {
        for (const RecordedCall& call : recorded) {
            const TunableKernel* tunable = registry.dispatcher().tunable(call.kernel);
            if (tunable == nullptr) {
                continue;
            }
            const std::string device = device_type_name(call.device.type);
            const NamedValues key = detail::call_key(*tunable, call.args, call.device);
            if (!seen.insert({tunable->name, device, key}).second) {
                continue; // the same call again
            }
            TunedCall result;
            result.op = tunable->name;
            result.device = to_string(call.device);
            result.key = text_of(key);
            result.key_values = key;
            result.default_settings = detail::from_config(*tunable, tunable->defaults());
            if (const TuningEntry* stored =
                    cache_->find(machine, tunable->name, device, tunable->version, key);
                stored != nullptr && !options.retune) {
                result.reused = true;
                result.best_ms = stored->best_ms;
                result.default_ms = stored->default_ms;
                result.best = text_of(stored->config);
                result.best_settings = stored->config;
                report.calls.push_back(result);
                continue;
            }
            const std::vector<detail::SearchRegion> space = tunable->space(call.args);
            if (space.empty()) {
                continue; // nothing to tune for this call
            }
            result.method = search_method_name(options.search);
            for (const detail::SearchRegion& region : space) {
                TuningRegion info;
                info.name = region.name;
                info.start = detail::from_config(*tunable, region.start);
                for (const detail::SearchAxis& axis : region.axes) {
                    info.arguments.push_back(
                        {tunable->params[static_cast<std::size_t>(axis.param)], axis.values});
                }
                info.points = detail::region_points(*tunable, region, call.args).size();
                result.space_size += info.points;
                result.regions.push_back(info);
            }

            const detail::KernelFn kernel = registry.dispatcher().lookup(call.kernel);
            detail::Stream& stream = registry.get(call.device).stream(call.device.index);
            OpArgs args = with_fresh_outputs(call.args);
            auto run_with = [&](const TuningConfig& config) {
                args.tuning = config;
                kernel(args, stream);
                stream.synchronize();
            };
            warm(call.device, [&] { run_with(TuningConfig{}); });

            run_with(TuningConfig{});
            const auto expected = read_outputs(args);
            const TuningConfig defaults = tunable->defaults();

            // Early stop: a point already slower than give_up x the best median so far.
            std::function<double()> give_up_ms;
            if (options.give_up > 0) {
                give_up_ms = [&] { return result.best_ms * options.give_up; };
            }

            // The objective: each point is checked against the default's result (infinite when
            // it disagrees), then timed. It is measured once; the search may ask again, for free.
            std::map<detail::TuningPoint, double> measured;
            TuningConfig best = defaults;
            result.best_ms = std::numeric_limits<double>::infinity();
            const auto search_start = Clock::now();
            auto measure = [&](const std::vector<detail::Probe>& probes) {
                std::vector<double> times;
                for (const detail::Probe& probe : probes) {
                    if (const auto it = measured.find(probe.config.values); it != measured.end()) {
                        times.push_back(it->second);
                        continue;
                    }
                    TunedCandidate c;
                    c.settings = detail::from_config(*tunable, probe.config);
                    c.region = probe.region;
                    c.step = probe.step;
                    c.sweep = probe.sweep;
                    c.level = probe.level;
                    double ms = std::numeric_limits<double>::infinity();
                    run_with(probe.config);
                    if (!agrees(expected, read_outputs(args))) {
                        ++result.rejected;
                        c.rejected = true;
                    } else {
                        ++result.tried;
                        ms = median_ms([&] { run_with(probe.config); }, options, give_up_ms,
                                       &c.stopped);
                        c.ms = ms;
                        result.stopped += c.stopped ? 1 : 0;
                        if (ms < result.best_ms) {
                            result.best_ms = ms;
                            best = probe.config;
                        }
                    }
                    measured[probe.config.values] = ms;
                    result.candidates.push_back(c);
                    times.push_back(ms);
                }
                return times;
            };
            result.default_ms = measure({{defaults, "", "default", 0, 0}})[0];

            detail::search(*tunable, call.args, space, options.search, options.max_sweeps,
                           options.levels, measure);
            result.search_s = seconds_since(search_start);
            if (!std::isfinite(result.best_ms)) { // even the default disagreed with itself
                result.best_ms = result.default_ms;
                best = defaults;
            }

            // Confirmation: the points were measured seconds apart, and the clock (a GPU's above
            // all) or the load can change in between. The default and the fastest few are timed
            // again, taking turns, and the fastest of those times is kept.
            if (options.confirm > 0) {
                std::vector<std::pair<double, std::size_t>> ranked;
                for (std::size_t i = 0; i < result.candidates.size(); ++i) {
                    const TunedCandidate& c = result.candidates[i];
                    if (!c.rejected && c.step != "default") {
                        ranked.emplace_back(c.ms, i);
                    }
                }
                std::sort(ranked.begin(), ranked.end());
                std::vector<TuningConfig> finalists = {defaults};
                for (std::size_t i = 0;
                     i < ranked.size() &&
                     finalists.size() <= static_cast<std::size_t>(options.confirm);
                     ++i) {
                    const TuningConfig config =
                        detail::to_config(*tunable, result.candidates[ranked[i].second].settings);
                    if (config.values != defaults.values) {
                        finalists.push_back(config);
                    }
                }
                std::vector<std::vector<double>> times(finalists.size());
                for (int round = 0; round < 3; ++round) {
                    for (std::size_t i = 0; i < finalists.size(); ++i) {
                        // Rotate the order, so no point is always timed first.
                        const std::size_t at =
                            (i + static_cast<std::size_t>(round)) % finalists.size();
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
                result.confirmed = finalists.size();
            }

            TuningEntry entry;
            entry.machine = machine;
            entry.op = tunable->name;
            entry.device = device;
            entry.version = tunable->version;
            entry.key = key;
            entry.config = detail::from_config(*tunable, best);
            entry.best_ms = result.best_ms;
            entry.default_ms = result.default_ms;
            entry.mode = "tested";
            result.best = text_of(entry.config);
            result.best_settings = entry.config;
            if (const TuningEntry* old =
                    cache_->find(machine, entry.op, entry.device, entry.version, key)) {
                replaced.push_back(*old);
            }
            added.push_back(entry);
            cache_->store(entry);
            if (options.verbose) {
                std::cout << "  " << result.op << " on " << result.device << " " << result.key
                          << ": " << result.best << " " << result.best_ms << " ms (default "
                          << result.default_ms << " ms; " << result.candidates.size() << " of "
                          << result.space_size << " points measured";
                if (result.stopped > 0) {
                    std::cout << ", " << result.stopped << " stopped early";
                }
                std::cout << ", " << result.method << " search, " << result.search_s << " s)\n";
            }
            report.calls.push_back(result);
        }
    };

    // 1. Run once, recording every tunable kernel call and every op called with Execution::Auto.
    std::vector<RecordedCall> recorded;
    std::vector<detail::AutoCall> autos;
    {
        detail::TuningScope scope({nullptr, &recorded, &autos});
        finish(algorithm.run(inputs));
    }

    // 2. Kernel settings of the calls as they ran.
    tune_kernels(recorded);

    // 3. Device choice for the Execution::Auto calls: each device that has a kernel for the op is
    // timed through dispatch() with the copies there and back, after its own kernel settings were
    // tuned, and must give the CPU's result.
    std::set<std::string> seen_places;
    for (const detail::AutoCall& call : autos) {
        detail::PlacementEntry key = detail::placement_key(call.op, call.args, call.data_device);
        std::string key_text = key.dtype;
        for (const auto& shape : key.shapes) {
            key_text += " " + shape_text(shape);
        }
        if (!seen_places.insert(key.op + " " + key.data_device + " " + key_text).second) {
            continue;
        }
        PlacedCall result;
        result.op = key.op;
        result.data_device = to_string(call.data_device);
        result.key = key_text;
        if (const detail::PlacementEntry* stored = cache_->find_placement(key);
            stored != nullptr && !options.retune) {
            result.reused = true;
            result.chosen = stored->chosen;
            result.best_ms = stored->best_ms;
            result.times = stored->times;
            for (const auto& [execution, ms] : stored->times) {
                if (execution == "cpu") {
                    result.default_ms = ms;
                }
            }
            report.placements.push_back(result);
            continue;
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
            continue; // nothing to choose from
        }

        std::vector<std::vector<float>> expected;
        double best_ms = 0;
        for (const auto& [execution, target] : choices) {
            const std::string name = execution_name(execution);
            {
                // This device's kernel settings first, so each device is timed at its best.
                std::vector<RecordedCall> on_device;
                OpArgs probe = with_fresh_outputs(call.args);
                {
                    detail::TuningScope scope({cache_.get(), &on_device, nullptr});
                    detail::dispatch(call.op, probe, execution);
                    finish(probe.outputs);
                }
                tune_kernels(on_device);
            }
            OpArgs args = with_fresh_outputs(call.args);
            auto run = [&, execution = execution] {
                detail::TuningScope scope({cache_.get(), nullptr, nullptr});
                detail::dispatch(call.op, args, execution);
                finish(args.outputs);
            };
            warm(target, run);
            run();
            if (execution == Execution::CPU) {
                expected = read_outputs(args);
            } else if (!agrees(expected, read_outputs(args))) {
                result.rejected.push_back(name);
                continue;
            }
            const double ms = median_ms(run, options);
            result.times.emplace_back(name, ms);
            if (execution == Execution::CPU) {
                result.default_ms = ms;
            }
            if (result.chosen.empty() || ms < best_ms) {
                best_ms = ms;
                result.chosen = name;
            }
        }
        result.best_ms = best_ms;

        detail::PlacementEntry entry = key;
        entry.chosen = result.chosen;
        entry.best_ms = result.best_ms;
        entry.times = result.times;
        entry.mode = "tested";
        if (const detail::PlacementEntry* old = cache_->find_placement(entry)) {
            replaced_places.push_back(*old);
        }
        added_places.push_back(entry);
        cache_->store_placement(entry);
        if (options.verbose) {
            std::cout << "  " << result.op << " (data on " << result.data_device << ") " << key_text
                      << ": " << result.chosen << " " << result.best_ms << " ms (cpu "
                      << result.default_ms << " ms)\n";
        }
        report.placements.push_back(result);
    }

    // 4. Tested: the whole algorithm with the default and the tuned settings.
    TuneOptions whole = options;
    whole.min_time_s = std::max(options.min_time_s, 0.2);
    report.default_ms = median_ms([&] { finish(algorithm.run(inputs)); }, whole);
    report.tuned_ms = median_ms([&] { finish(algorithm.run(inputs, *this)); }, whole);
    if ((!added.empty() || !added_places.empty()) && report.tuned_ms > report.default_ms * 1.02) {
        report.kept = false;
        for (const TuningEntry& entry : added) {
            cache_->erase(entry);
        }
        for (const TuningEntry& entry : replaced) {
            cache_->store(entry);
        }
        for (const detail::PlacementEntry& entry : added_places) {
            cache_->erase_placement(entry);
        }
        for (const detail::PlacementEntry& entry : replaced_places) {
            cache_->store_placement(entry);
        }
    }
    if (options.save) {
        save();
    }
    return report;
}

} // namespace mensurflow
