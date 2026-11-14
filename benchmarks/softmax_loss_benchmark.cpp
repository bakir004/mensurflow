// softmax, log_softmax, the losses and attention.
//   sequential  the kernel on one thread (the baseline).
//   parallel    the same kernel with N OpenMP threads; its result is checked bit for bit against the
//               sequential one first (one row is one thread's work, and reduced losses add fixed
//               chunks in order, so the bits do not depend on the thread count).
// Rate is GB/s of bytes the op must move: every input read once and the output written once
// (labels count 8 bytes per row). "vs seq" divides the sequential median time by this row's.
// Attention goes through the public API (two batched GEMMs around a masked softmax) and reports
// GFLOP/s of the two GEMMs; its sequential row is the same call on one thread.
//
//   softmax_loss_benchmark
#include "bench.hpp"

#include "backends/cpu/ops/loss/loss.hpp"
#include "backends/cpu/ops/softmax/softmax.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace mf = mensurflow;

namespace {

std::string shape(std::int64_t rows, std::int64_t cols) {
    return std::to_string(rows) + "x" + std::to_string(cols);
}

// One row per thread count for `run`, writing `out_floats` floats at `out`: the sequential
// baseline, then each thread count checked against it bit for bit.
template <class Run>
void sweep(bench::Table& table, const bench::Config& config, const std::string& label,
           const std::vector<int>& thread_counts, double work, const float* out,
           std::size_t out_floats, Run&& run) {
    cpu::set_max_threads(1);
    run();
    const std::vector<float> expected(out, out + out_floats);
    auto observe = [&] { return bench::checksum(out, out_floats); };
    const bench::Sample sequential = bench::measure(config, run, observe);
    table.row(label, "sequential", "1", sequential, work, &sequential);
    for (const int threads : thread_counts) {
        if (threads == 1) {
            continue;
        }
        cpu::set_max_threads(threads);
        run();
        bench::check_bits(expected.data(), out, out_floats,
                          label + " threads=" + std::to_string(threads));
        const bench::Sample parallel = bench::measure(config, run, observe);
        table.row(label, "parallel", std::to_string(threads), parallel, work, &sequential);
    }
    table.blank();
}

// Logits-like values in [-4, 4) and labels spread over the classes.
void fill_logits(float* data, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(static_cast<int>((i * 37) % 97) - 48) / 12.0f;
    }
}

std::vector<std::int64_t> make_labels(std::int64_t rows, std::int64_t classes) {
    std::vector<std::int64_t> labels(static_cast<std::size_t>(rows));
    for (std::int64_t r = 0; r < rows; ++r) {
        labels[static_cast<std::size_t>(r)] = (r * 13 + 5) % classes;
    }
    return labels;
}

} // namespace

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header("CPU softmax, losses and attention, float32", config,
                        {"sequential = one thread; parallel = the same kernel on N OpenMP threads",
                         "rate is GB/s of bytes moved (inputs read once, output written once);",
                         "attention: GFLOP/s of its two GEMMs, through the public API"});
    bench::Table table("GB/s", 1e9, "vs seq", "");

    // Classifier heads (1000 classes), attention scores (128 and 512 keys), a 32k vocabulary.
    const std::int64_t shapes[][2] = {
        {64, 1000}, {1024, 1000}, {4096, 128}, {16384, 512}, {32, 32000}};

    // ---- softmax / log_softmax forward --------------------------------------------------------
    for (const bool log : {false, true}) {
        table.section(std::string(log ? "log_softmax" : "softmax") +
                      " over the last axis [rows, cols]");
        for (const auto& s : shapes) {
            const std::int64_t rows = s[0], cols = s[1];
            const auto n = static_cast<std::size_t>(rows * cols);
            bench::Arrays arrays(n, 2);
            fill_logits(arrays[0], n);
            float* out = arrays[1];
            sweep(table, config, shape(rows, cols), thread_counts, 8.0 * static_cast<double>(n),
                  out, n, [&] {
                      (log ? cpu::log_softmax_f32 : cpu::softmax_f32)(arrays[0], out, rows, cols,
                                                                      1);
                  });
        }
    }

    // ---- backward from the forward output -----------------------------------------------------
    for (const bool log : {false, true}) {
        table.section(std::string(log ? "log_softmax" : "softmax") + "_backward [rows, cols]");
        const std::int64_t backward_shapes[][2] = {{1024, 1000}, {16384, 512}};
        for (const auto& s : backward_shapes) {
            const std::int64_t rows = s[0], cols = s[1];
            const auto n = static_cast<std::size_t>(rows * cols);
            bench::Arrays arrays(n, 4);
            fill_logits(arrays[0], n);
            (log ? cpu::log_softmax_f32 : cpu::softmax_f32)(arrays[0], arrays[1], rows, cols, 1);
            bench::fill_b(arrays[2], n);
            float* out = arrays[3];
            sweep(table, config, shape(rows, cols), thread_counts, 12.0 * static_cast<double>(n),
                  out, n, [&] {
                      (log ? cpu::log_softmax_backward_f32
                           : cpu::softmax_backward_f32)(arrays[1], arrays[2], out, rows, cols, 1);
                  });
        }
    }

    // ---- cross-entropy, mean over the batch ---------------------------------------------------
    table.section("cross_entropy, mean [rows, classes]");
    const std::int64_t ce_shapes[][2] = {{64, 1000}, {1024, 1000}, {8192, 1000}, {32, 32000}};
    for (const auto& s : ce_shapes) {
        const std::int64_t rows = s[0], classes = s[1];
        const auto n = static_cast<std::size_t>(rows * classes);
        bench::Arrays arrays(n, 1);
        fill_logits(arrays[0], n);
        const auto labels = make_labels(rows, classes);
        float loss = 0.0f;
        sweep(table, config, shape(rows, classes), thread_counts,
              4.0 * static_cast<double>(n) + 8.0 * static_cast<double>(rows), &loss, 1,
              [&] { cpu::cross_entropy_f32(arrays[0], labels.data(), &loss, rows, classes, 2); });
    }

    table.section("cross_entropy_backward, mean [rows, classes]");
    for (const auto& s : ce_shapes) {
        const std::int64_t rows = s[0], classes = s[1];
        const auto n = static_cast<std::size_t>(rows * classes);
        bench::Arrays arrays(n, 2);
        fill_logits(arrays[0], n);
        const auto labels = make_labels(rows, classes);
        const float upstream = 1.0f;
        float* out = arrays[1];
        sweep(table, config, shape(rows, classes), thread_counts,
              8.0 * static_cast<double>(n) + 8.0 * static_cast<double>(rows), out, n, [&] {
                  cpu::cross_entropy_backward_f32(arrays[0], labels.data(), &upstream, out, rows,
                                                  classes, 2);
              });
    }

    // ---- mse and bce, mean, and their backward ------------------------------------------------
    struct Elementwise {
        const char* name;
        void (*forward)(const float*, const float*, float*, std::int64_t, int);
        void (*backward)(const float*, const float*, const float*, float*, std::int64_t, int);
    };

    for (const Elementwise& op :
         {Elementwise{"mse_loss", cpu::mse_loss_f32, cpu::mse_loss_backward_f32},
          Elementwise{"binary_cross_entropy_with_logits", cpu::bce_with_logits_f32,
                      cpu::bce_with_logits_backward_f32}}) {
        table.section(std::string(op.name) + ", mean [n]");
        for (const std::int64_t n : {1000, 1 << 16, 1 << 20, 1 << 24}) {
            const auto size = static_cast<std::size_t>(n);
            bench::Arrays arrays(size, 2);
            fill_logits(arrays[0], size);
            for (std::size_t i = 0; i < size; ++i) {
                arrays[1][i] = static_cast<float>(i % 2); // 0/1 targets
            }
            float loss = 0.0f;
            sweep(table, config, std::to_string(n), thread_counts, 8.0 * static_cast<double>(n),
                  &loss, 1, [&] { op.forward(arrays[0], arrays[1], &loss, n, 2); });
        }
        table.section(std::string(op.name) + "_backward, mean [n]");
        for (const std::int64_t n : {1000, 1 << 16, 1 << 20, 1 << 24}) {
            const auto size = static_cast<std::size_t>(n);
            bench::Arrays arrays(size, 3);
            fill_logits(arrays[0], size);
            for (std::size_t i = 0; i < size; ++i) {
                arrays[1][i] = static_cast<float>(i % 2);
            }
            const float upstream = 1.0f;
            float* out = arrays[2];
            sweep(table, config, std::to_string(n), thread_counts, 12.0 * static_cast<double>(n),
                  out, size, [&] { op.backward(arrays[0], arrays[1], &upstream, out, n, 2); });
        }
    }

    // ---- attention through the public API -----------------------------------------------------
    bench::Table flops("GFLOP/s", 1e9, "vs seq", "");
    flops.section("attention [batch, heads, L, 64], causal, GFLOP/s of q k^T and p v");
    const std::int64_t attention_shapes[][3] = {{1, 8, 128}, {4, 12, 256}, {1, 16, 1024}};
    for (const auto& s : attention_shapes) {
        const std::int64_t batch = s[0], heads = s[1], length = s[2], depth = 64;
        const auto n = static_cast<std::size_t>(batch * heads * length * depth);
        std::vector<float> data(n);
        fill_logits(data.data(), n);
        const mf::Tensor q = mf::Tensor::from_vector(data, {batch, heads, length, depth});
        const mf::Tensor k = mf::Tensor::from_vector(data, {batch, heads, length, depth});
        const mf::Tensor v = mf::Tensor::from_vector(data, {batch, heads, length, depth});
        mf::Tensor out;
        auto run = [&] { out = mf::attention(q, k, v, nullptr, true); };
        const double gflop = 4.0 * static_cast<double>(batch * heads) * length * length * depth;
        cpu::set_max_threads(1);
        run();
        const std::vector<float> expected = out.to_vector<float>();
        auto observe = [&] { return static_cast<double>(out.to_vector<float>()[0]); };
        const bench::Sample sequential = bench::measure(config, run, observe);
        const std::string label =
            std::to_string(batch) + "x" + std::to_string(heads) + "x" + std::to_string(length);
        flops.row(label, "sequential", "1", sequential, gflop, &sequential);
        for (const int threads : thread_counts) {
            if (threads == 1) {
                continue;
            }
            cpu::set_max_threads(threads);
            run();
            const std::vector<float> got = out.to_vector<float>();
            bench::check_close(expected.data(), got.data(), got.size(), 1e-4,
                               label + " threads=" + std::to_string(threads));
            const bench::Sample parallel = bench::measure(config, run, observe);
            flops.row(label, "parallel", std::to_string(threads), parallel, gflop, &sequential);
        }
        flops.blank();
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}
