// Tests the auto-tuner: the results file (round trip, several machines, errors with line numbers),
// that stored settings reach the kernel, and Tuner::tune on a small algorithm on every device that
// has a tunable matmul: results found, reused on the second run, the same values as the defaults,
// and kept across a reload of the file.
#include "core/registry.hpp"
#include "core/tuning.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
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
    a.key = {{"m", 64}, {"n", 256}, {"k", 784}};
    a.config = {{"kernel", 1}, {"bm", 32}, {"bn", 64}};
    a.best_ms = 0.041;
    a.default_ms = 0.047;
    a.mode = "tested";
    detail::TuningEntry b = a;
    b.machine = "Machine B";
    b.device = "cpu";
    b.key.push_back({"threads", 8});
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

    // Comments and blank lines are ignored; malformed lines name their line.
    write_file("# comment\n\nmachine M\nmatmul cpu v1 m=1 n=2 k=3 -> rows=0 | 1.0 ms (default "
               "2.0 ms) tested\n");
    detail::TuningCache hand;
    hand.load(path);
    test::require(hand.entries().size() == 1 && hand.entries()[0].key.size() == 3, "hand-written");
    const std::pair<const char*, const char*> bad[] = {
        {"matmul cpu v1 m=1 -> rows=0 | 1 ms (default 2 ms) tested\n", ":1:"},         // no machine
        {"machine M\nmatmul cpu v1 m=1 rows=0 | 1 ms (default 2 ms) tested\n", ":2:"}, // no ->
        {"machine M\nmatmul cpu v1 m=x -> rows=0 | 1 ms (default 2 ms) tested\n", ":2:"},
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
    return !t->candidates(args).empty();
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
        test::require(!call.reused && call.tried > 0 && call.rejected == 0 && call.best_ms > 0 &&
                          call.best_ms <= call.default_ms && !call.best.empty(),
                      "measured " + call.key + ": tried " + std::to_string(call.tried) +
                          ", rejected " + std::to_string(call.rejected));
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

// Device choice lines: round trip, and malformed ones name their line.
void test_placement_format() {
    write_file("machine M\n"
               "place matmul f32 cpu 256x1024 1024x4096 -> gpu | 0.8000 ms (cpu 8.4000 ms, gpu "
               "0.8000 ms) tested\n"
               "place sum f32 metal scalar -> cpu | 0.0100 ms (cpu 0.0100 ms) tested\n");
    detail::TuningCache cache;
    cache.load(path);
    test::require(cache.placements().size() == 2, "two placements read");
    const detail::PlacementEntry& first = cache.placements()[0];
    test::require(
        first.op == "matmul" && first.dtype == "f32" && first.data_device == "cpu" &&
            first.shapes == std::vector<std::vector<std::int64_t>>{{256, 1024}, {1024, 4096}} &&
            first.chosen == "gpu" && first.times.size() == 2 && first.times[0].first == "cpu" &&
            std::abs(first.times[0].second - 8.4) < 1e-9 && first.mode == "tested",
        "placement fields");
    test::require(cache.placements()[1].shapes == std::vector<std::vector<std::int64_t>>{{}},
                  "a 0-d input is written 'scalar'");
    cache.save(path);
    detail::TuningCache again;
    again.load(path);
    test::require(again.placements().size() == 2 && again.find_placement(first) != nullptr &&
                      again.find_placement(first)->times == first.times,
                  "placement round trip");
    const char* const bad[] = {
        "machine M\nplace matmul f32 cpu 8x8 -> npu | 1 ms (cpu 1 ms) tested\n",
        "machine M\nplace matmul f32 cpu 8x8 8xq -> gpu | 1 ms (cpu 1 ms) tested\n",
        "machine M\nplace matmul f32 cpu 8x8 -> gpu | 1 ms (cpu 1) tested\n",
        "machine M\nplace matmul f32 cpu 8x8 -> gpu | fast\n",
    };
    for (const char* text : bad) {
        write_file(text);
        detail::TuningCache broken;
        bool threw = false;
        try {
            broken.load(path);
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find(":2:") != std::string::npos;
        }
        test::require(threw, std::string("malformed placement names line 2: ") + text);
    }
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
    one.run({a}, Tuner(path)); // no placement: Auto stays on the CPU
    write_file(results +
               "place matmul f32 cpu 8x8 8x8 -> gpu | 1 ms (cpu 2 ms, gpu 1 ms) tested\n");
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
    test::require(report.placements.size() == 2,
                  "matmul and add placed, got " + std::to_string(report.placements.size()));
    for (const PlacedCall& p : report.placements) {
        test::require((p.chosen == "cpu" || p.chosen == "gpu") && p.times.size() == 2 &&
                          p.rejected.empty() && p.best_ms > 0 && p.best_ms <= p.default_ms &&
                          p.data_device == "cpu:0",
                      "placement of " + p.op + ": chose " + p.chosen);
        std::cout << "    " << p.op << " " << p.key << ": " << p.chosen << " (cpu " << p.default_ms
                  << " ms, best " << p.best_ms << " ms)\n";
    }
    const auto expected = net.run(inputs)[0].to_vector<float>();
    const auto tuned = net.run(inputs, tuner)[0].to_vector<float>();
    test::require(test::matches_f32(tuned, expected), "placed run gives the default's values");
    if (report.kept) {
        const TuneReport second = tuner.tune(net, inputs, fast);
        test::require(second.placements.size() == 2 && second.placements[0].reused &&
                          second.placements[1].reused,
                      "second tune reuses the placements");
        test::require(Tuner(path).describe() == tuner.describe(), "placements survive a reload");
    }
    std::remove(path);
}

} // namespace

int main() {
    return test::run("tuning_test", [] {
        test_file_format();
        test_placement_format();
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
