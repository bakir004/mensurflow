// max_pool2d and avg_pool2d, forward and backward: sequential versus parallel kernels on the
// ResNet stem pool, a 2x2/2 pool, global 7x7 pooling and a small pool, at batch 1 and 32.
//
//   pooling_benchmark [--threads=1,6,12,24] [--batches=1,32] [--only=name]
//
// Rate is GB/s of the arrays each pass must move once: max forward reads x and writes y and the
// int64 indices; max backward reads grad_y and the indices and writes grad_x; avg forward reads x
// and writes y; avg backward reads grad_y and writes grad_x. Compare with bandwidth_benchmark.
// The PyTorch side is benchmarks/reference/conv_pool.py.
#include "bench.hpp"

#include "backends/cpu/ops/conv/pooling.hpp"
#include "backends/cpu/threads.hpp"

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct Pool {
    const char* name;
    std::int64_t channels, h, k, stride, pad; // square input and window
};

const Pool pools[] = {
    {"r18.maxpool 3x3/2", 64, 112, 3, 2, 1},
    {"2x2/2 128ch 56px", 128, 56, 2, 2, 0},
    {"global 7x7 512ch", 512, 7, 7, 7, 0},
    {"small 3x3/1 8ch 16px", 8, 16, 3, 1, 1},
};

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        values.push_back(std::stoll(item));
    }
    return values;
}

// OpenMP workers keep spinning after a call; let them go idle before the next variant.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

double checksum(const std::int64_t* values, std::size_t n) {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total = total * 1315423911 + values[i];
    }
    return static_cast<double>(total);
}

} // namespace

int main(int argc, char** argv) {
    const int maximum_threads = cpu::max_threads();
    std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    std::vector<std::int64_t> batches = {1, 32};
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--threads=", 0) == 0) {
            thread_counts.clear();
            for (std::int64_t t : parse_list(arg.substr(10))) {
                thread_counts.push_back(static_cast<int>(t));
            }
        } else if (arg.rfind("--batches=", 0) == 0) {
            batches = parse_list(arg.substr(10));
        } else if (arg.rfind("--only=", 0) == 0) {
            only = arg.substr(7);
        }
    }

    const bench::Config config;
    bench::print_header("max_pool2d and avg_pool2d forward and backward, float32 NCHW", config,
                        {"seq = *_f32_sequential on one thread; par = *_f32_parallel",
                         "rate = GB/s of the arrays each pass moves once; ceiling: "
                         "bandwidth_benchmark"});
    bench::Table table("GB/s", 1e9, "vs seq", "");

    for (const Pool& pool : pools) {
        if (!only.empty() && std::string(pool.name).find(only) == std::string::npos) {
            continue;
        }
        for (const std::int64_t batch : batches) {
            cpu::Pool2dShape s;
            s.n = batch;
            s.c = pool.channels;
            s.h = s.w = pool.h;
            s.kh = s.kw = pool.k;
            s.stride_h = s.stride_w = pool.stride;
            s.pad_h = s.pad_w = pool.pad;
            s.h_out = s.w_out = (pool.h + 2 * pool.pad - pool.k) / pool.stride + 1;
            const auto n_in = static_cast<std::size_t>(s.n * s.c * s.h * s.w);
            const auto n_out = static_cast<std::size_t>(s.n * s.c * s.h_out * s.w_out);
            bench::Arrays arrays(n_in, 5);
            float* x = arrays[0];
            float* y = arrays[1];
            float* gy = arrays[2];
            float* gx = arrays[3];
            float* expected = arrays[4];
            std::vector<std::int64_t> indices(n_out), expected_indices(n_out);
            bench::fill_a(x, n_in);
            bench::fill_b(gy, n_out);
            const double max_forward_bytes = 4.0 * n_in + 12.0 * n_out;
            const double max_backward_bytes = 12.0 * n_out + 4.0 * n_in;
            const double avg_bytes = 4.0 * n_in + 4.0 * n_out;
            table.section(std::string(pool.name) + " N=" + std::to_string(batch));

            // Each pass: the sequential row, then the parallel rows checked against it.
            auto sweep = [&](const char* pass, double bytes, auto&& sequential, auto&& parallel,
                             const float* result, std::size_t length, auto&& observe) {
                cpu::set_max_threads(1);
                sequential();
                std::copy(result, result + length, expected);
                settle();
                const bench::Sample seq = bench::measure(config, sequential, observe);
                table.row(pass, "seq", "1", seq, bytes, &seq);
                for (int threads : thread_counts) {
                    cpu::set_max_threads(threads);
                    parallel();
                    bench::check_bits(expected, result, length, std::string(pool.name) + " " + pass);
                    settle();
                    const bench::Sample par = bench::measure(config, parallel, observe);
                    table.row(pass, "par", std::to_string(threads), par, bytes, &seq);
                }
            };
            sweep(
                "max forward", max_forward_bytes,
                [&] { cpu::max_pool2d_f32_sequential(x, y, expected_indices.data(), s); },
                [&] { cpu::max_pool2d_f32_parallel(x, y, indices.data(), s); }, y, n_out,
                [&] { return bench::checksum(y, n_out) + checksum(indices.data(), n_out); });
            sweep(
                "max backward", max_backward_bytes,
                [&] { cpu::max_pool2d_backward_f32_sequential(gy, expected_indices.data(), gx, s); },
                [&] { cpu::max_pool2d_backward_f32_parallel(gy, expected_indices.data(), gx, s); },
                gx, n_in, [&] { return bench::checksum(gx, n_in); });
            sweep(
                "avg forward", avg_bytes, [&] { cpu::avg_pool2d_f32_sequential(x, y, s); },
                [&] { cpu::avg_pool2d_f32_parallel(x, y, s); }, y, n_out,
                [&] { return bench::checksum(y, n_out); });
            // The gather may round differently from the scatter in a build that fuses multiply
            // and add, so the backward check allows rounding.
            cpu::set_max_threads(1);
            cpu::avg_pool2d_backward_f32_sequential(gy, gx, s);
            std::vector<float> want(gx, gx + n_in);
            settle();
            const bench::Sample seq = bench::measure(
                config, [&] { cpu::avg_pool2d_backward_f32_sequential(gy, gx, s); },
                [&] { return bench::checksum(gx, n_in); });
            table.row("avg backward", "seq", "1", seq, avg_bytes, &seq);
            for (int threads : thread_counts) {
                cpu::set_max_threads(threads);
                auto run = [&] { cpu::avg_pool2d_backward_f32_parallel(gy, gx, s); };
                run();
                bench::check_close(want.data(), gx, n_in, 1e-5, std::string(pool.name) + " avg backward");
                settle();
                const bench::Sample par =
                    bench::measure(config, run, [&] { return bench::checksum(gx, n_in); });
                table.row("avg backward", "par", std::to_string(threads), par, avg_bytes, &seq);
            }
        }
    }
    cpu::set_max_threads(maximum_threads);
}
