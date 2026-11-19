// Tests the auto-tuner: the results file (round trip, several machines, errors with line numbers),
// that stored settings reach the kernel, and Tuner::tune on a small algorithm on every device that
// has a tunable matmul: results found, reused on the second run, the same values as the defaults,
// and kept across a reload of the file.
#include "core/registry.hpp"
#include "core/tuning.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace detail = mensurflow::detail;

namespace {

const char* const path = "tuning_test_results.txt";

std::vector<float> values(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 16;
    }
    return v;
}

void write_file(const std::string& text) {
    std::ofstream out(path);
    out << text;
}

void test_file_format() {
    detail::TuningCache cache;
    detail::TuningEntry a;
    a.machine = "Machine A | GPU A";
    a.op = "matmul";
    a.device = "metal";
    a.version = 1;
    a.key = "m=64 n=256 k=784";
    a.config = {{"kernel", 1}, {"bm", 32}, {"bn", 64}};
    a.best_ms = 0.041;
    a.default_ms = 0.047;
    a.mode = "tested";
    detail::TuningEntry b = a;
    b.machine = "Machine B";
    b.device = "cpu";
    b.key += " threads=8";
    b.config = {{"rows", 4}, {"depth", 64}, {"cols", 1024}};
    cache.store(a);
    cache.store(b);
    detail::TuningEntry a2 = a; // same call: replaces a
    a2.config = {{"kernel", 2}};
    cache.store(a2);
    test::require(cache.entries().size() == 2, "storing the same call replaces it");
    cache.save(path);

    detail::TuningCache loaded;
    loaded.load(path);
    test::require(loaded.entries().size() == 2, "both machines read back");
    const detail::TuningEntry* found = loaded.find(a.machine, "matmul", "metal", 1, a.key);
    test::require(found != nullptr && found->config == a2.config && found->mode == "tested" &&
                      std::abs(found->best_ms - 0.041) < 1e-9 &&
                      std::abs(found->default_ms - 0.047) < 1e-9,
                  "entry round trip");
    test::require(loaded.find(a.machine, "matmul", "metal", 2, a.key) == nullptr,
                  "another kernel version is not found");
    test::require(loaded.find(b.machine, "matmul", "cpu", 1, b.key) != nullptr, "second machine");

    // Comments and blank lines are ignored, extra spaces in a key too; malformed lines name their
    // line.
    write_file("# comment\n\nmachine M\nmatmul cpu v1 m=1  n=2 k=3 -> rows=0 | 1.0 ms (default "
               "2.0 ms) tested\n");
    detail::TuningCache hand;
    hand.load(path);
    test::require(hand.entries().size() == 1 && hand.entries()[0].key == "m=1 n=2 k=3",
                  "hand-written");
    const std::pair<const char*, const char*> bad[] = {
        {"matmul cpu v1 m=1 -> rows=0 | 1 ms (default 2 ms) tested\n", ":1:"},         // no machine
        {"machine M\nmatmul cpu v1 m=1 rows=0 | 1 ms (default 2 ms) tested\n", ":2:"}, // no ->
        {"machine M\nmatmul cpu v1 m=1 -> rows=x | 1 ms (default 2 ms) tested\n", ":2:"},
        {"machine M\nmatmul cpu 1 m=1 -> rows=0 | 1 ms (default 2 ms) tested\n", ":2:"},
        {"machine M\n\nmatmul cpu v1 m=1 -> rows=0 | fast\n", ":3:"},
    };
    for (const auto& [text, where] : bad) {
        write_file(text);
        detail::TuningCache broken;
        bool threw = false;
        try {
            broken.load(path);
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find(where) != std::string::npos;
        }
        test::require(threw, std::string("malformed file names line ") + where + ": " + text);
    }
    std::remove(path);
    detail::TuningCache missing;
    missing.load(path);
    test::require(missing.entries().empty(), "a missing file is an empty cache");
}

bool has_tunable_matmul(Device device) {
    const detail::TunableKernel* t = detail::Registry::instance().dispatcher().tunable(
        {detail::OpId::Matmul, device.type, DType::F32});
    if (t == nullptr) {
        return false;
    }
    // A tunable kernel may still offer nothing for a call (the CPU's AVX2 kernel).
    const Tensor probe = Tensor::from_vector(values(64 * 64, 1), {64, 64}, device);
    detail::OpArgs args;
    args.inputs = {probe, probe};
    return !t->space(args).empty();
}

// Two layers: [48, 96] x [96, 80], plus a [48, 80] tensor (add runs on every device; relu has no
// Metal kernel yet), then x [80, 24].
void test_tune(Device device) {
    const Execution on = test::execution_on(device);
    const Algorithm mlp([on](const std::vector<Tensor>& in) {
        const Tensor h = add(matmul(in[0], in[1], on), in[3], on);
        return std::vector<Tensor>{matmul(h, in[2], on)};
    });
    const std::vector<Tensor> inputs = {Tensor::from_vector(values(48 * 96, 1), {48, 96}, device),
                                        Tensor::from_vector(values(96 * 80, 2), {96, 80}, device),
                                        Tensor::from_vector(values(80 * 24, 3), {80, 24}, device),
                                        Tensor::from_vector(values(48 * 80, 5), {48, 80}, device)};
    std::remove(path);
    TuneOptions fast;
    fast.min_time_s = 0.002;
    fast.min_samples = 2;

    Tuner tuner(path);
    test::require(tuner.size() == 0, "a new file has no results");
    const TuneReport first = tuner.tune(mlp, inputs, fast);
    test::require(first.calls.size() == 2,
                  "two distinct matmul calls, got " + std::to_string(first.calls.size()));
    for (const TunedCall& call : first.calls) {
        const auto rejected = std::count_if(call.points.begin(), call.points.end(),
                                            [](const TunedPoint& p) { return p.rejected; });
        test::require(!call.reused && call.points.size() > 1 && rejected == 0 && call.best_ms > 0 &&
                          call.best_ms <= call.default_ms && !call.best_settings.empty() &&
                          call.device != "auto",
                      "measured " + call.key + ": " + std::to_string(call.points.size()) +
                          " points, " + std::to_string(rejected) + " rejected");
    }
    test::require(first.default_ms > 0 && first.tuned_ms > 0, "the whole algorithm was timed");
    if (first.kept) {
        test::require(tuner.size() == 2, "both results stored");
    }

    // The stored settings give the default's values (exact inputs: multiples of 1/16, small sums).
    const auto expected = mlp.run(inputs)[0].to_vector<float>();
    const auto tuned = mlp.run(inputs, tuner)[0].to_vector<float>();
    test::require(test::matches_f32(tuned, expected), "tuned run gives the default's values");

    // A second tune reuses them; a reload of the file sees them.
    if (first.kept) {
        const TuneReport second = tuner.tune(mlp, inputs, fast);
        test::require(second.calls.size() == 2 && second.calls[0].reused && second.calls[1].reused,
                      "second tune reuses the stored results");
        Tuner reloaded(path);
        test::require(reloaded.size() == 2, "results survive a reload");
        test::require(reloaded.describe() == tuner.describe(), "same text after a reload");
    }

    std::remove(path);
}

// A stored result reaches the kernel: an invalid Metal kernel number in the file makes the call
// fail, while the same algorithm without the tuner runs.
void test_applied(Device device) {
    const Tensor a = Tensor::from_vector(values(8 * 8, 4), {8, 8}, device);
    const Algorithm one([](const std::vector<Tensor>& in) {
        return std::vector<Tensor>{matmul(in[0], in[0], Execution::GPU)};
    });
    write_file("machine " + detail::machine_name() +
               "\nmatmul metal v1 m=8 n=8 k=8 -> kernel=7 | 1 ms (default 1 ms) tested\n");
    Tuner tuner(path);
    test::require(tuner.size() == 1, "hand-written result for this machine");
    one.run({a});
    test::require_throws<std::invalid_argument>([&] { one.run({a}, tuner); },
                                                "the stored kernel=7 reached the Metal kernel");
    std::remove(path);
}

// Device choice lines are ordinary lines with device "auto" and the setting gpu.
void test_placement_format() {
    write_file("machine M\n"
               "matmul auto v1 data=cpu f32 256x1024 1024x4096 -> gpu=1 | 0.8000 ms (default "
               "8.4000 ms) tested\n");
    detail::TuningCache cache;
    cache.load(path);
    const detail::TuningEntry* found =
        cache.find("M", "matmul", "auto", 1, "data=cpu f32 256x1024 1024x4096");
    test::require(found != nullptr && found->config == TuningValues{{"gpu", 1}} &&
                      std::abs(found->default_ms - 8.4) < 1e-9,
                  "device choice read");
    detail::OpArgs args;
    args.inputs = {Tensor::zeros({256, 1024}), Tensor::zeros({1024, 4096})};
    test::require(detail::placement_key(args, Device{DeviceType::CPU, 0}) == found->key,
                  "placement_key matches the line");
    std::remove(path);
}

// A stored device choice moves an Execution::Auto call: with "-> gpu" the matmul of CPU data runs on
// Metal, where the stored kernel=7 makes it fail; without the line it runs on the CPU.
void test_placement_applied() {
    const Tensor a = Tensor::from_vector(values(8 * 8, 6), {8, 8});
    const Algorithm one([](const std::vector<Tensor>& in) {
        return std::vector<Tensor>{matmul(in[0], in[0])}; // Execution::Auto
    });
    const std::string results =
        "machine " + detail::machine_name() +
        "\nmatmul metal v1 m=8 n=8 k=8 -> kernel=7 | 1 ms (default 1 ms) tested\n";
    write_file(results);
    one.run({a}, Tuner(path)); // no device choice: Auto stays on the CPU
    write_file(results +
               "matmul auto v1 data=cpu f32 8x8 8x8 -> gpu=1 | 1 ms (default 2 ms) tested\n");
    const Tuner tuner(path);
    test::require_throws<std::invalid_argument>([&] { one.run({a}, tuner); },
                                                "the stored placement sent the call to Metal");
    std::remove(path);
}

// Tuning an algorithm whose ops use Execution::Auto: a large matmul and a small add on CPU data.
void test_tune_placement() {
    const Algorithm net([](const std::vector<Tensor>& in) {
        return std::vector<Tensor>{add(matmul(in[0], in[1]), in[2])};
    });
    const std::vector<Tensor> inputs = {Tensor::from_vector(values(512 * 512, 7), {512, 512}),
                                        Tensor::from_vector(values(512 * 512, 8), {512, 512}),
                                        Tensor::from_vector(values(512 * 512, 9), {512, 512})};
    std::remove(path);
    TuneOptions fast;
    fast.min_time_s = 0.002;
    fast.min_samples = 2;
    Tuner tuner(path);
    const TuneReport report = tuner.tune(net, inputs, fast);
    auto placements = [](const TuneReport& r) {
        std::vector<TunedCall> out;
        for (const TunedCall& c : r.calls) {
            if (c.device == "auto") {
                out.push_back(c);
            }
        }
        return out;
    };
    const std::vector<TunedCall> placed = placements(report);
    test::require(placed.size() == 2,
                  "matmul and add placed, got " + std::to_string(placed.size()));
    for (const TunedCall& p : placed) {
        test::require(p.points.size() == 2 && !p.points[0].rejected && !p.points[1].rejected &&
                          p.best_ms > 0 && p.best_ms <= p.default_ms &&
                          p.key.rfind("data=cpu f32 ", 0) == 0 && p.best_settings.size() == 1 &&
                          p.best_settings[0].first == "gpu",
                      "device choice of " + p.op + ": " + to_string(p.best_settings));
        std::cout << "    " << p.op << " " << p.key << ": " << to_string(p.best_settings)
                  << " (cpu " << p.default_ms << " ms, best " << p.best_ms << " ms)\n";
    }
    const auto expected = net.run(inputs)[0].to_vector<float>();
    const auto tuned = net.run(inputs, tuner)[0].to_vector<float>();
    test::require(test::matches_f32(tuned, expected), "placed run gives the default's values");
    if (report.kept) {
        const TuneReport second = tuner.tune(net, inputs, fast);
        const std::vector<TunedCall> again = placements(second);
        test::require(again.size() == 2 && again[0].reused && again[1].reused,
                      "second tune reuses the device choices");
        test::require(Tuner(path).describe() == tuner.describe(), "placements survive a reload");
    }
    std::remove(path);
}

} // namespace

namespace {

// A batch objective from a per-point one, for detail::search.
template <typename F> detail::Measure each_point(F f) {
    return [f](const std::vector<detail::TuningConfig>& points) mutable {
        std::vector<double> times;
        for (const detail::TuningConfig& p : points) {
            times.push_back(f(p));
        }
        return times;
    };
}

// The search on a synthetic function of two arguments a, b in 1..16, lowest at a = 11, b = 5,
// with points where a + b > 24 not allowed.
void test_search() {
    detail::TunableKernel t;
    t.name = "synthetic";
    t.params = {"a", "b", "variant"};
    t.allowed = [](const detail::OpArgs&, const detail::TuningConfig& c) {
        return c.values[0] + c.values[1] <= 24;
    };
    std::vector<std::int64_t> axis;
    for (std::int64_t v = 1; v <= 16; ++v) {
        axis.push_back(v);
    }
    detail::TuningConfig start;
    start.values = {16, 16, 0}; // not allowed: the search starts at the nearest allowed point
    start.size = 3;
    const std::vector<detail::SearchRegion> space = {{"bowl", start, {{0, axis}, {1, axis}}}};
    const detail::OpArgs args;
    test::require(detail::region_points(t, space[0], args).size() == 256 - 36,
                  "allowed points of the region");

    for (const SearchMethod method : {SearchMethod::BinarySampling, SearchMethod::Exhaustive}) {
        std::map<detail::TuningConfig, double> seen;
        detail::TuningConfig best;
        double best_value = 1e300;
        bool all_allowed = true;
        detail::search(t, args, space, method, 4, 4, // 4 levels: every value of 16
                       each_point([&](const detail::TuningConfig& c) {
                           all_allowed = all_allowed && t.allowed(args, c);
                           const double a = static_cast<double>(c.values[0]) - 11;
                           const double b = static_cast<double>(c.values[1]) - 5;
                           const double value = a * a + 2 * b * b + 0.5 * a * b + 1;
                           seen[c] = value;
                           if (value < best_value) {
                               best_value = value;
                               best = c;
                           }
                           return value;
                       }));
        const std::string name = search_method_name(method);
        test::require(all_allowed, name + ": every measured point is allowed");
        test::require(best.values[0] == 11 && best.values[1] == 5,
                      name + ": found the minimum, got a=" + std::to_string(best.values[0]) +
                          " b=" + std::to_string(best.values[1]));
        if (method == SearchMethod::BinarySampling) {
            // Whole lines through the current point, a few passes: a fraction of the grid.
            test::require(seen.size() < 80, "binary sampling: measured " +
                                                std::to_string(seen.size()) + " of 220 points");
        } else {
            test::require(seen.size() == 220, "exhaustive: every allowed point");
        }
    }
    // A plateau with noise on it, measured on an M3 Pro (CPU matmul 512^3, depth 32 .. 512): a
    // binary search compared 128 with 256, within noise, and went right; sampling keeps the best.
    {
        detail::TunableKernel line_kernel;
        line_kernel.params = {"depth"};
        line_kernel.allowed = [](const detail::OpArgs&, const detail::TuningConfig&) {
            return true;
        };
        detail::TuningConfig from;
        from.values = {128};
        from.size = 1;
        const std::vector<detail::SearchRegion> line = {
            {"tiles", from, {{0, {32, 64, 128, 256, 512}}}}};
        const std::map<std::int64_t, double> ms = {
            {32, 1.424}, {64, 1.527}, {128, 1.679}, {256, 1.665}, {512, 1.682}};
        std::int64_t found = 0;
        double lowest = 1e300;
        detail::search(line_kernel, args, line, SearchMethod::BinarySampling, 4, 2,
                       each_point([&](const detail::TuningConfig& c) {
                           const double value = ms.at(c.values[0]);
                           if (value < lowest) {
                               lowest = value;
                               found = c.values[0];
                           }
                           return value;
                       }));
        test::require(found == 32, "plateau: found depth " + std::to_string(found));
    }
    test::require(parse_search_method("binary") == SearchMethod::BinarySampling &&
                      parse_search_method("exhaustive") == SearchMethod::Exhaustive,
                  "method names");
    test::require_throws<std::invalid_argument>([] { parse_search_method("random"); },
                                                "unknown method name");
}

} // namespace

int main() {
    return test::run("tuning_test", [] {
        test_file_format();
        test_placement_format();
        test_search();
        for (Device device : list_devices()) {
            if (!has_tunable_matmul(device)) {
                std::cout << "  " << to_string(device) << ": no tunable matmul, skipped\n";
                continue;
            }
            std::cout << "  tune on " << to_string(device) << '\n';
            test_tune(device);
            if (device.type == DeviceType::Metal) {
                test_applied(device);
                test_placement_applied();
                std::cout << "  device choice between cpu:0 and " << to_string(device) << '\n';
                test_tune_placement();
            }
        }
    });
}
