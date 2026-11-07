// Comparisons/where: direct CPU kernels, optionally against native preallocated ATen out calls.
// Inputs, output addresses, thread settings, observer and timing harness are shared by both sides.
// Allocation, Tensor wrapping, correctness checks, checksums and Python are outside the timer.
#include "bench.hpp"

#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/threads.hpp"

#ifdef MENSURFLOW_TORCH_REFERENCE
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/Version.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

enum class Kind { Eq, Ne, Lt, Le, Gt, Ge, Where };

using ComparisonKernel = void (*)(const float*, const float*, std::uint8_t*, std::int64_t);

struct Operation {
    const char* name;
    Kind kind;
    ComparisonKernel kernel;
    bool (*reference)(float, float);
};

const Operation operations[] = {
    {"eq", Kind::Eq, cpu::eq_f32, [](float a, float b) { return a == b; }},
    {"ne", Kind::Ne, cpu::ne_f32, [](float a, float b) { return a != b; }},
    {"lt", Kind::Lt, cpu::lt_f32, [](float a, float b) { return a < b; }},
    {"le", Kind::Le, cpu::le_f32, [](float a, float b) { return a <= b; }},
    {"gt", Kind::Gt, cpu::gt_f32, [](float a, float b) { return a > b; }},
    {"ge", Kind::Ge, cpu::ge_f32, [](float a, float b) { return a >= b; }},
    {"where", Kind::Where, nullptr, nullptr}};

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t parsed = 0;
        const auto value = std::stoll(token, &parsed);
        if (value <= 0 || parsed != token.size()) {
            throw std::invalid_argument("list values must be positive integers");
        }
        if (std::find(values.begin(), values.end(), value) != values.end()) {
            throw std::invalid_argument("list values must not repeat");
        }
        values.push_back(value);
    }
    if (values.empty() || text.back() == ',') {
        throw std::invalid_argument("expected a nonempty comma-separated list");
    }
    return values;
}

void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

void check_flags(const std::vector<std::uint8_t>& expected, const std::vector<std::uint8_t>& actual,
                 const std::string& what) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            throw std::runtime_error(what + ": Bool output differs at index " + std::to_string(i));
        }
    }
}

#ifdef MENSURFLOW_TORCH_REFERENCE
void run_torch(Kind kind, const at::Tensor& x, const at::Tensor& y, const at::Tensor& mask,
               at::Tensor& output) {
    switch (kind) {
    case Kind::Eq:
        at::eq_out(output, x, y);
        break;
    case Kind::Ne:
        at::ne_out(output, x, y);
        break;
    case Kind::Lt:
        at::lt_out(output, x, y);
        break;
    case Kind::Le:
        at::le_out(output, x, y);
        break;
    case Kind::Gt:
        at::gt_out(output, x, y);
        break;
    case Kind::Ge:
        at::ge_out(output, x, y);
        break;
    case Kind::Where:
        at::where_out(output, mask, x, y);
        break;
    }
}
#endif

int run(int argc, char** argv) {
    const int maximum_threads = cpu::max_threads();
    std::vector<std::int64_t> sizes{1000, 32769, 1000000, 10000000};
    std::vector<int> threads = bench::thread_counts(maximum_threads);
    std::set<std::string> selected;
    bool reverse = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "selection_benchmark [operation ...] [--sizes=1000,32769,1000000,10000000] "
                         "[--threads=1,2,6,12] [--reverse]\n"
                         "Operations: eq ne lt le gt ge where\n"
                         "With LibTorch, --reverse times ATen first; otherwise it has no effect.\n"
                         "CPU-only mode always includes a one-thread baseline.\n";
            return 0;
        }
        if (arg == "--reverse") {
            reverse = true;
        } else if (arg.rfind("--sizes=", 0) == 0) {
            sizes = parse_list(arg.substr(8));
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads.clear();
            for (auto count : parse_list(arg.substr(10))) {
                if (count > maximum_threads) {
                    throw std::invalid_argument("requested threads exceed initial OpenMP maximum");
                }
                threads.push_back(static_cast<int>(count));
            }
        } else {
            bool known = false;
            for (const auto& op : operations) {
                known = known || arg == op.name;
            }
            if (!known) {
                throw std::invalid_argument("unknown operation or option: " + arg);
            }
            selected.insert(arg);
        }
    }
    // Four padded F32 arrays need at most 4*(n+1279) elements. Validate before narrowing to
    // size_t so neither the padded stride nor its multiplication can wrap on any host width.
    const auto max_stride = std::vector<float>().max_size() / 4;
    for (auto n : sizes) {
        const auto count = static_cast<std::uint64_t>(n);
        if (max_stride <= 1279 || count > max_stride - 1279 ||
            count > std::vector<std::uint8_t>().max_size()) {
            throw std::invalid_argument("size exceeds the padded benchmark allocation limit");
        }
    }

    bench::Config config;
    config.slow_repeats = config.repeats; // Nine samples even when an individual call is slow.
    std::vector<std::string> notes{
        "F32 comparisons write Bool bytes 0/1; where mask repeats true,true,false,false,false",
        "comparisons: 9 logical bytes/element (two F32 reads, one Bool write)",
        "where: 13 logical bytes/element upper bound (mask, both F32 branches, F32 write)",
        "where may physically read only the selected branch; GB/s is not measured DRAM traffic",
        "preallocated buffers; allocation, Tensor wrapping, checksums and Python are not timed",
        "500 ms worker-settling pause before each measurement; repeat after cooling"};
#ifdef MENSURFLOW_TORCH_REFERENCE
    c10::InferenceMode inference_guard;
    at::set_num_interop_threads(1);
    notes.push_back(std::string("LibTorch version: ") + TORCH_VERSION);
    notes.push_back("scope: direct MensurFlow kernels vs native C++ ATen out operators; ATen dispatch included");
    notes.push_back("both sides share input/output addresses and requested intra-op threads; inter-op=1");
    notes.push_back("vs torch = ATen median / MensurFlow median; above 1 favors MensurFlow");
    notes.push_back(reverse ? "measurement order: ATen then MensurFlow"
                            : "measurement order: MensurFlow then ATen");
#else
    notes.push_back("LibTorch reference disabled; sequential baseline is this kernel on one thread");
    notes.push_back("vs seq = one-thread median / this row's median");
    if (reverse) {
        notes.push_back("--reverse has no effect without the optional LibTorch reference");
    }
#endif
    bench::print_header("CPU comparisons and where", config, notes);
#ifdef MENSURFLOW_TORCH_REFERENCE
    std::cout << "\nLibTorch build configuration:\n" << at::show_config()
              << "\nInitial LibTorch parallel configuration:\n" << at::get_parallel_info() << '\n';
    bench::Table table("GB/s", 1e9, "vs torch", "");
#else
    bench::Table table("GB/s", 1e9, "vs seq", "");
#endif

    for (const auto& op : operations) {
        if (!selected.empty() && selected.count(op.name) == 0) {
            continue;
        }
        const bool is_where = op.kind == Kind::Where;
        table.section(op.name);
        for (auto n : sizes) {
            const auto len = static_cast<std::size_t>(n);
            bench::Arrays arrays(len, 4);
            float* x = arrays[0];
            float* y = arrays[1];
            float* output = arrays[2];
            float* expected = arrays[3];
            std::vector<std::uint8_t> mask(len), flags(len), expected_flags(len);
            bench::fill_a(x, len);
            bench::fill_b(y, len);
            for (std::size_t i = 0; i < len; ++i) {
                mask[i] = static_cast<std::uint8_t>(i % 5 < 2);
                if (is_where) {
                    expected[i] = mask[i] != 0 ? x[i] : y[i];
                } else {
                    expected_flags[i] = static_cast<std::uint8_t>(op.reference(x[i], y[i]));
                }
            }
            auto ours = [&] {
                if (is_where) {
                    cpu::where_f32(mask.data(), x, y, output, n);
                } else {
                    op.kernel(x, y, flags.data(), n);
                }
            };
            auto observe = [&] {
                return is_where ? bench::checksum(output, len)
                                : static_cast<double>(std::accumulate(flags.begin(), flags.end(),
                                                                       std::uint64_t{0}));
            };
            auto reset_output = [&] {
                if (is_where) {
                    std::fill(output, output + len, 6789.5f);
                } else {
                    std::fill(flags.begin(), flags.end(), std::uint8_t{0xa5});
                }
            };
            auto check = [&](const std::string& label) {
                if (is_where) {
                    bench::check_bits(expected, output, len, label);
                } else {
                    check_flags(expected_flags, flags, label);
                }
            };
            const double bytes = (is_where ? 13.0 : 9.0) * static_cast<double>(n);
            const std::string size_label = std::to_string(n);
#ifdef MENSURFLOW_TORCH_REFERENCE
            const auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            const auto bool_options = at::TensorOptions().dtype(at::kBool).device(at::kCPU);
            // from_blob borrows memory; arrays/vectors outlive all wrappers and timed calls.
            const at::Tensor tx = at::from_blob(x, {n}, float_options);
            const at::Tensor ty = at::from_blob(y, {n}, float_options);
            const at::Tensor condition = at::from_blob(mask.data(), {n}, bool_options);
            void* output_address = is_where ? static_cast<void*>(output)
                                            : static_cast<void*>(flags.data());
            at::Tensor tout = at::from_blob(output_address, {n},
                                            is_where ? float_options : bool_options);
            auto torch = [&] { run_torch(op.kind, tx, ty, condition, tout); };
            for (int count : threads) {
                at::set_num_threads(count);
                cpu::set_max_threads(count);
                if (cpu::max_threads() != count || at::get_num_threads() != count) {
                    throw std::runtime_error("libraries did not accept the same thread count");
                }
                reset_output();
                ours();
                check(std::string(op.name) + " MensurFlow vs independent reference");
                reset_output();
                torch();
                check(std::string(op.name) + " ATen vs independent reference");
                if (tout.data_ptr() != output_address) {
                    throw std::runtime_error("ATen replaced the preallocated output buffer");
                }
                bench::Sample own_sample, torch_sample;
                auto measure_ours = [&] {
                    settle();
                    own_sample = bench::measure(config, ours, observe);
                };
                auto measure_torch = [&] {
                    settle();
                    torch_sample = bench::measure(config, torch, observe);
                };
                if (reverse) {
                    measure_torch();
                    measure_ours();
                } else {
                    measure_ours();
                    measure_torch();
                }
                table.row(size_label, "mensurflow", std::to_string(count), own_sample, bytes,
                          &torch_sample);
                table.row(size_label, "pytorch", std::to_string(count), torch_sample, bytes,
                          &torch_sample);
                std::cout.flush();
            }
#else
            cpu::set_max_threads(1);
            reset_output();
            ours();
            check(std::string(op.name) + " one thread vs independent reference");
            settle();
            const bench::Sample baseline = bench::measure(config, ours, observe);
            table.row(size_label, "mensurflow", "1", baseline, bytes, &baseline);
            for (int count : threads) {
                if (count == 1) {
                    continue;
                }
                cpu::set_max_threads(count);
                reset_output();
                ours();
                check(std::string(op.name) + " threads=" + std::to_string(count));
                settle();
                const bench::Sample sample = bench::measure(config, ours, observe);
                table.row(size_label, "mensurflow", std::to_string(count), sample, bytes, &baseline);
                std::cout.flush();
            }
#endif
            table.blank();
        }
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "selection_benchmark: " << error.what() << '\n';
        return 1;
    }
}
