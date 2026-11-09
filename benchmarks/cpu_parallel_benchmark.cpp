// One thread vs many on the CPU, for the elementwise ops and dot. Each kernel in
// src/backends/cpu/ops is a single loop that OpenMP splits across threads (MENSURFLOW_PARALLEL_FOR).
//   sequential: the kernel on one thread.
//   parallel:   the same kernel on 2, 4, ... threads.
// "vs seq" divides the sequential median by this row's median: what the threads add.
// Kernels are called directly into preallocated buffers: no dispatcher, no allocation. GB/s counts
// the bytes each op must touch (binary: 12 per element, unary and dot: 8 per element, bias_add:
// 8*n + 4*features).
// Optional command-line names select operations, e.g. cpu_parallel_benchmark axpy bias_add.
// Built on the shared harness: see benchmarks/bench.hpp and docs/BENCHMARKING.md.
#include "bench.hpp"
#include "elementwise_cutoffs.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/threads.hpp"
#include "reference/elementwise.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

using Kernel = void (*)(const float*, const float*, float*, std::int64_t);

// Unary operations use the same timing loop, with the unused second input ignored.
template <void (*Unary)(const float*, float*, std::int64_t)>
void unary_kernel(const float* x, const float*, float* out, std::int64_t n) {
    Unary(x, out, n);
}

void leaky_relu_kernel(const float* x, const float*, float* out, std::int64_t n) {
    cpu::leaky_relu_f32(x, out, n, 0.01f);
}

void gelu_kernel(const float* x, const float*, float* out, std::int64_t n) {
    cpu::gelu_f32(x, out, n);
}

void gelu_tanh_kernel(const float* x, const float*, float* out, std::int64_t n) {
    cpu::gelu_f32(x, out, n, true);
}

void elu_kernel(const float* x, const float*, float* out, std::int64_t n) {
    cpu::elu_f32(x, out, n, 1.0f);
}

void softplus_kernel(const float* x, const float*, float* out, std::int64_t n) {
    cpu::softplus_f32(x, out, n, 1.0f, 20.0f);
}

void clamp_kernel(const float* x, const float*, float* out, std::int64_t n) {
    cpu::clamp_f32(x, out, n, -1.0f, 1.0f);
}

constexpr float axpy_alpha = 1.25f;

template <void (*Axpy)(float, const float*, const float*, float*, std::int64_t)>
void axpy_kernel(const float* x, const float* y, float* out, std::int64_t n) {
    Axpy(axpy_alpha, x, y, out, n);
}

// A power-of-two feature count up to 256 that divides every benchmark size exactly.
std::int64_t bias_features(std::int64_t n) {
    std::int64_t features = 1;
    while (features < 256 && (n & (2 * features - 1)) == 0) {
        features *= 2;
    }
    return features;
}

template <void (*BiasAdd)(const float*, const float*, float*, std::int64_t, std::int64_t)>
void bias_add_kernel(const float* x, const float* bias, float* out, std::int64_t n) {
    BiasAdd(x, bias, out, n, bias_features(n));
}

struct Op {
    const char* name;
    double bytes_per_element;
    Kernel kernel; // nullptr for dot, which returns a value instead of filling a buffer
    Kernel reference = nullptr; // Optional independent float64 check before timing.
};

std::vector<std::int64_t> parse_sizes(const std::string& text) {
    std::vector<std::int64_t> sizes;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t parsed = 0;
        const auto n = std::stoll(token, &parsed);
        if (n <= 0 || parsed != token.size()) {
            throw std::invalid_argument("sizes must be positive integers");
        }
        // Arrays(len, 4) rounds each stride up by at most 1279 floats. Validate before
        // converting to size_t so both the padding and four-array allocation cannot wrap.
        const auto max_stride = std::vector<float>().max_size() / 4;
        if (max_stride <= 1279 || static_cast<std::uint64_t>(n) > max_stride - 1279) {
            throw std::invalid_argument("size exceeds the four-array allocation limit");
        }
        sizes.push_back(n);
    }
    if (sizes.empty() || text.back() == ',') {
        throw std::invalid_argument("sizes must be a nonempty comma-separated list");
    }
    return sizes;
}

} // namespace

int main(int argc, char* argv[]) {
    const int max = cpu::max_threads();
    const std::vector<int> counts = bench::thread_counts(max);

    const Op ops[] = {{"add", 12.0, cpu::add_f32},
                      {"sub", 12.0, cpu::sub_f32},
                      {"mul", 12.0, cpu::mul_f32},
                      {"div", 12.0, cpu::div_f32},
                      {"pow", 12.0, cpu::pow_f32},
                      {"minimum", 12.0, cpu::minimum_f32},
                      {"maximum", 12.0, cpu::maximum_f32},
                      {"neg", 8.0, unary_kernel<cpu::neg_f32>},
                      {"abs", 8.0, unary_kernel<cpu::abs_f32>},
                      {"square", 8.0, unary_kernel<cpu::square_f32>},
                      {"relu", 8.0, unary_kernel<cpu::relu_f32>},
                      {"leaky_relu", 8.0, leaky_relu_kernel, unary_kernel<reference::leaky_relu>},
                      {"gelu", 8.0, gelu_kernel, unary_kernel<reference::gelu>},
                      {"gelu_tanh", 8.0, gelu_tanh_kernel, unary_kernel<reference::gelu_tanh>},
                      {"silu", 8.0, unary_kernel<cpu::silu_f32>, unary_kernel<reference::silu>},
                      {"elu", 8.0, elu_kernel, unary_kernel<reference::elu>},
                      {"softplus", 8.0, softplus_kernel, unary_kernel<reference::softplus>},
                      {"sqrt", 8.0, unary_kernel<cpu::sqrt_f32>},
                      {"exp", 8.0, unary_kernel<cpu::exp_f32>},
                      {"sigmoid", 8.0, unary_kernel<cpu::sigmoid_f32>},
                      {"log", 8.0, unary_kernel<cpu::log_f32>},
                      {"rsqrt", 8.0, unary_kernel<cpu::rsqrt_f32>},
                      {"tanh", 8.0, unary_kernel<cpu::tanh_f32>},
                      {"sin", 8.0, unary_kernel<cpu::sin_f32>},
                      {"cos", 8.0, unary_kernel<cpu::cos_f32>},
                      {"reciprocal", 8.0, unary_kernel<cpu::reciprocal_f32>},
                      {"clamp", 8.0, clamp_kernel},
                      {"axpy", 12.0, axpy_kernel<cpu::axpy_f32>},
                      {"bias_add", 8.0, bias_add_kernel<cpu::bias_add_f32>},
                      {"dot", 8.0, nullptr}};

    std::vector<std::int64_t> sizes{1000, 10000, 32768, 32769, 40000, 50000, 75000, 100000,
                                    200000, 500000, 1000000, 2000000, 5000000, 10000000, 33554432};
    std::set<std::string> selected;
    bench::ElementwiseCutoffOption cutoff;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--help") {
            std::cout << "cpu_parallel_benchmark [operation ...] [--sizes=n,...] [--cutoff=n]\n"
                         "--cutoff affects selected elementwise operations; dot is unchanged.\n"
                         "0 forces the parallel branch for nonempty inputs; "
                         "9223372036854775807 forces serial.\n";
            return 0;
        }
        try {
            if (cutoff.parse(name)) {
                continue;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid --cutoff: " << e.what() << '\n';
            return 1;
        }
        if (name.rfind("--sizes=", 0) == 0) {
            try {
                sizes = parse_sizes(name.substr(8));
            } catch (const std::exception& e) {
                std::cerr << "Invalid --sizes: " << e.what() << '\n';
                return 1;
            }
            continue;
        }
        bool known = false;
        for (const Op& op : ops) {
            known = known || name == op.name;
        }
        if (!known) {
            std::cerr << "Unknown operation: " << name << ". Available operations:";
            for (const Op& op : ops) {
                std::cerr << ' ' << op.name;
            }
            std::cerr << '\n';
            return 1;
        }
        selected.insert(name);
    }
    auto wanted = [&](const char* name) { return selected.empty() || selected.count(name) != 0; };

    bench::Config config;
    config.slow_repeats = config.repeats; // Keep nine samples for cutoff comparisons too.
    std::vector<std::string> notes = {
        "impl: sequential = the kernel on one thread; parallel = the same kernel on N threads",
        "vs seq = the sequential median time / this row's median time",
        "elementwise parallel branch: n > the operation's cutoff; defaults await tuning",
        "500 ms worker-settling pause before each measurement; repeat after cooling"};
    for (const Op& op : ops) {
        if (wanted(op.name)) {
            notes.push_back(op.kernel == nullptr
                ? "dot cutoff (unchanged): " + std::to_string(cpu::parallel_threshold)
                : "elementwise cutoff: " + cutoff.configure(op.name));
        }
    }
    if (wanted("axpy")) {
        notes.push_back("axpy: alpha=" + std::to_string(axpy_alpha) + ", fused multiply-add");
    }
    if (wanted("bias_add")) {
        notes.push_back(
            "bias_add: features = largest power-of-two divisor of n up to 256; the first "
            "features entries of b are the bias; GB/s counts 8*n + 4*features bytes");
    }
    if (wanted("leaky_relu") || wanted("gelu") || wanted("gelu_tanh") || wanted("silu") ||
        wanted("elu") || wanted("softplus")) {
        notes.push_back("activations: leaky_relu slope=0.01, ELU alpha=1, "
                        "softplus beta=1 threshold=20; "
                        "GELU exact and tanh have separate rows; checked against float64 first");
    }
    bench::print_header("Sequential vs parallel on the CPU (float32)", config, notes);

    bench::Table table("GB/s", 1e9, "vs seq", "");

    for (const Op& op : ops) {
        if (!wanted(op.name)) {
            continue;
        }
        const bool is_dot = op.kernel == nullptr;
        const bool is_bias_add = std::string(op.name) == "bias_add";
        const std::string operation_name = op.name;
        const bool positive_input = operation_name == "sqrt" || operation_name == "rsqrt" ||
                                    operation_name == "log" || operation_name == "pow";
        table.section(op.name);

        for (std::int64_t n : sizes) {
            const auto len = static_cast<std::size_t>(n);
            const double bytes = op.bytes_per_element * static_cast<double>(n) +
                                 (is_bias_add ? 4.0 * static_cast<double>(bias_features(n)) : 0.0);
            bench::Arrays arrays(len, 4);
            float* a = arrays[0];
            float* b = arrays[1];
            float* out = arrays[2];
            float* reference = arrays[3];
            bench::fill_a(a, len);
            bench::fill_b(b, len);
            if (positive_input) {
                for (std::size_t i = 0; i < len; ++i) {
                    a[i] = std::fabs(a[i]) + 0.25f; // Time the valid finite domain.
                }
            }

            // One variable receives dot's result; elementwise ops fill `out` or `reference`.
            double value = 0.0;
            auto observe_out = [&] { return is_dot ? value : bench::checksum(out, len); };

            // The one-thread result is the reference for the multi-thread runs.
            auto run_sequential = [&] {
                if (is_dot) {
                    value = cpu::dot_f32(a, b, n);
                } else {
                    op.kernel(a, b, reference, n);
                }
            };
            cpu::set_max_threads(1);
            run_sequential();
            if (op.reference != nullptr) {
                op.reference(a, b, out, n);
                bench::check_close(out, reference, len, 3e-6,
                                   std::string(op.name) + " float64 reference");
            }
            const double reference_value = value;
            auto observe_reference = [&] {
                return is_dot ? value : bench::checksum(reference, len);
            };

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const bench::Sample seq_sample =
                bench::measure(config, run_sequential, observe_reference);
            const std::string label = std::to_string(n);
            table.row(label, "sequential", "-", seq_sample, bytes, &seq_sample);

            for (int threads : counts) {
                if (threads == 1) {
                    continue; // that is the sequential row
                }
                cpu::set_max_threads(threads);
                auto run_parallel = [&] {
                    if (is_dot) {
                        value = cpu::dot_f32(a, b, n);
                    } else {
                        op.kernel(a, b, out, n);
                    }
                };
                // Elementwise results must match the one-thread ones bit for bit; dot may differ
                // slightly because the partial sums are added in a different order.
                run_parallel();
                const std::string what =
                    std::string("parallel ") + op.name + " threads=" + std::to_string(threads);
                if (is_dot) {
                    bench::check_value(reference_value, value, 1e-4, what);
                } else {
                    bench::check_bits(reference, out, len, what);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                const bench::Sample par = bench::measure(config, run_parallel, observe_out);
                table.row(label, "parallel", std::to_string(threads), par, bytes, &seq_sample);
            }
            table.blank();
        }
    }
    cpu::set_max_threads(max);
    return 0;
}
