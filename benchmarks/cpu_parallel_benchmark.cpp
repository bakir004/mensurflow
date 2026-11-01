// Scalar vs sequential vs parallel on the CPU, for the elementwise ops and dot.
//   scalar:     sequential.cpp compiled without auto-vectorization (one element at a time).
//   sequential: the same loop with whatever SIMD the compiler finds. No OpenMP, no parallel_for.
//   parallel:   the kernels in elementwise.cpp / blas.cpp (parallel_for / parallel_sum) run with
//               1, 2, 4, ... threads. The 1-thread row shows what the parallel machinery costs
//               when there is nothing to gain.
// The two speedup columns divide the scalar and the sequential median by this row's median, so
// "vs scalar" is SIMD plus threads and "vs seq" is the threads alone.
// Kernels are called directly into preallocated buffers: no dispatcher, no allocation. GB/s counts
// the bytes each op must touch (binary: 12 per element, unary and dot: 8 per element, bias_add:
// 8*n + 4*features).
// Optional command-line names select operations, e.g. cpu_parallel_benchmark axpy bias_add.
// Built on the shared harness: see benchmarks/bench.hpp and docs/BENCHMARKING.md.
#include "bench.hpp"

#include "backends/cpu/blas.hpp"
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/scalar.hpp"
#include "backends/cpu/sequential.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>

namespace cpu = mensurflow::cpu;
namespace seq = mensurflow::cpu::sequential;
namespace scalar = mensurflow::cpu::scalar;

namespace {

using Kernel = void (*)(const float*, const float*, float*, std::int64_t);

// Unary operations use the same timing loop, with the unused second input ignored.
template <void (*Unary)(const float*, float*, std::int64_t)>
void unary_kernel(const float* x, const float*, float* out, std::int64_t n) {
    Unary(x, out, n);
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
    Kernel scalar;     // nullptr for dot, which returns a value instead of filling a buffer
    Kernel sequential; // likewise
    Kernel parallel;   // likewise
};

} // namespace

int main(int argc, char* argv[]) {
    const int max = cpu::max_threads();
    const std::vector<int> counts = bench::thread_counts(max);

    const Op ops[] = {{"add", 12.0, scalar::add_f32, seq::add_f32, cpu::add_f32},
                      {"sub", 12.0, scalar::sub_f32, seq::sub_f32, cpu::sub_f32},
                      {"mul", 12.0, scalar::mul_f32, seq::mul_f32, cpu::mul_f32},
                      {"div", 12.0, scalar::div_f32, seq::div_f32, cpu::div_f32},
                      {"neg", 8.0, unary_kernel<scalar::neg_f32>, unary_kernel<seq::neg_f32>,
                       unary_kernel<cpu::neg_f32>},
                      {"abs", 8.0, unary_kernel<scalar::abs_f32>, unary_kernel<seq::abs_f32>,
                       unary_kernel<cpu::abs_f32>},
                      {"square", 8.0, unary_kernel<scalar::square_f32>,
                       unary_kernel<seq::square_f32>, unary_kernel<cpu::square_f32>},
                      {"relu", 8.0, unary_kernel<scalar::relu_f32>, unary_kernel<seq::relu_f32>,
                       unary_kernel<cpu::relu_f32>},
                      {"sqrt", 8.0, unary_kernel<scalar::sqrt_f32>, unary_kernel<seq::sqrt_f32>,
                       unary_kernel<cpu::sqrt_f32>},
                      {"exp", 8.0, unary_kernel<scalar::exp_f32>, unary_kernel<seq::exp_f32>,
                       unary_kernel<cpu::exp_f32>},
                      {"sigmoid", 8.0, unary_kernel<scalar::sigmoid_f32>,
                       unary_kernel<seq::sigmoid_f32>, unary_kernel<cpu::sigmoid_f32>},
                      {"axpy", 12.0, axpy_kernel<scalar::axpy_f32>, axpy_kernel<seq::axpy_f32>,
                       axpy_kernel<cpu::axpy_f32>},
                      {"bias_add", 8.0, bias_add_kernel<scalar::bias_add_f32>,
                       bias_add_kernel<seq::bias_add_f32>, bias_add_kernel<cpu::bias_add_f32>},
                      {"dot", 8.0, nullptr, nullptr, nullptr}};

    std::set<std::string> selected;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
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

    const bench::Config config;
    std::vector<std::string> notes = {
        "impl: scalar = sequential.cpp without auto-vectorization; sequential = the same code with "
        "the compiler's SIMD; parallel = the parallel kernel with N threads",
        "vs scalar / vs seq = that baseline's median time / this row's median time",
        "parallel cutoff: " + std::to_string(cpu::default_grain) + " elements"};
    if (wanted("axpy")) {
        notes.push_back("axpy: alpha=" + std::to_string(axpy_alpha) + ", fused multiply-add");
    }
    if (wanted("bias_add")) {
        notes.push_back(
            "bias_add: features = largest power-of-two divisor of n up to 256; the first "
            "features entries of b are the bias; GB/s counts 8*n + 4*features bytes");
    }
    bench::print_header("Scalar vs sequential vs parallel on the CPU (float32)", config, notes);

    const std::int64_t sizes[] = {1000,    10000,   32768,   32769,    40000,
                                  50000,   75000,   100000,  200000,   500000,
                                  1000000, 2000000, 5000000, 10000000, 33554432};
    bench::Table table("GB/s");

    for (const Op& op : ops) {
        if (!wanted(op.name)) {
            continue;
        }
        const bool is_dot = op.sequential == nullptr;
        const bool is_bias_add = std::string(op.name) == "bias_add";
        const bool nonnegative_input = std::string(op.name) == "sqrt";
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
            if (nonnegative_input) {
                for (std::size_t i = 0; i < len; ++i) {
                    a[i] = std::fabs(a[i]); // Time valid square roots, not domain-error handling.
                }
            }

            // One variable receives dot's result; elementwise ops fill `out` or `reference`.
            double value = 0.0;
            auto observe_out = [&] { return is_dot ? value : bench::checksum(out, len); };

            // The sequential result is the reference for the scalar and the parallel runs.
            auto run_sequential = [&] {
                if (is_dot) {
                    value = seq::dot_f32(a, b, n);
                } else {
                    op.sequential(a, b, reference, n);
                }
            };
            run_sequential();
            const double reference_value = value;
            auto observe_reference = [&] {
                return is_dot ? value : bench::checksum(reference, len);
            };

            auto run_scalar = [&] {
                if (is_dot) {
                    value = scalar::dot_f32(a, b, n);
                } else {
                    op.scalar(a, b, out, n);
                }
            };
            run_scalar();
            if (is_dot) {
                bench::check_value(reference_value, value, 0.0, "scalar dot");
            } else {
                bench::check_bits(reference, out, len, std::string("scalar ") + op.name);
            }

            const bench::Sample scalar_sample = bench::measure(config, run_scalar, observe_out);
            const bench::Sample seq_sample =
                bench::measure(config, run_sequential, observe_reference);
            const std::string label = std::to_string(n);
            table.row(label, "scalar", "-", scalar_sample, bytes, &scalar_sample, &seq_sample);
            table.row(label, "sequential", "-", seq_sample, bytes, &scalar_sample, &seq_sample);

            for (int threads : counts) {
                cpu::set_max_threads(threads);
                auto run_parallel = [&] {
                    if (is_dot) {
                        value = cpu::dot_f32(a, b, n);
                    } else {
                        op.parallel(a, b, out, n);
                    }
                };
                // Elementwise results must match the sequential ones bit for bit; dot may differ
                // slightly because the partial sums are added in a different order.
                run_parallel();
                const std::string what =
                    std::string("parallel ") + op.name + " threads=" + std::to_string(threads);
                if (is_dot) {
                    bench::check_value(reference_value, value, 1e-4, what);
                } else {
                    bench::check_bits(reference, out, len, what);
                }
                const bench::Sample par = bench::measure(config, run_parallel, observe_out);
                table.row(label, "parallel", std::to_string(threads), par, bytes, &scalar_sample,
                          &seq_sample);
            }
            table.blank();
        }
    }
    cpu::set_max_threads(max);
    return 0;
}
