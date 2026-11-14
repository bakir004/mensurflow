// argmin (last axis) and var / std along an axis, forward and backward: sequential versus parallel
// kernels on shapes from real models and on small and large inputs.
//
//   argmin_var_std_benchmark [--threads=1,6,12,24] [--only=name]
//
// Rate is GB/s of the input read once (plus the gradient written once for backward); compare with
// bandwidth_benchmark. The PyTorch side is benchmarks/reference/argmin_var_std.py.
#include "bench.hpp"

#include "backends/cpu/ops/reduction/reduction.hpp"
#include "backends/cpu/threads.hpp"

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct ArgminShape {
    const char* name;
    std::int64_t rows, width;
};

const ArgminShape argmin_shapes[] = {
    {"classes 32x1000", 32, 1000},
    {"classes 1024x1000", 1024, 1000},
    {"k-means 65536x64", 65536, 64},
    {"long rows 4x2^22", 4, 1 << 22},
    {"small 8x100", 8, 100},
};

struct MomentShape {
    const char* name;
    std::vector<std::int64_t> shape;
    int axis;
};

const MomentShape moment_shapes[] = {
    {"layer-norm stats [6272,768] axis 1", {6272, 768}, 1},
    {"batch-norm stats [100352,64] axis 0", {100352, 64}, 0},
    {"middle axis [32,64,3136] axis 1", {32, 64, 3136}, 1},
    {"vector 2^24", {1 << 24}, 0},
    {"small [8,100] axis 1", {8, 100}, 1},
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
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--threads=", 0) == 0) {
            thread_counts.clear();
            for (std::int64_t t : parse_list(arg.substr(10))) {
                thread_counts.push_back(static_cast<int>(t));
            }
        } else if (arg.rfind("--only=", 0) == 0) {
            only = arg.substr(7);
        }
    }
    const bench::Config config;
    bench::print_header("argmin, var and std along an axis, float32", config,
                        {"seq = *_f32_sequential on one thread; par = *_f32_parallel",
                         "rate = GB/s of the input read once (+ grad_x written for backward)",
                         "var/std use correction = 1; std rows are timed like var"});
    bench::Table table("GB/s", 1e9, "vs seq", "");

    for (const ArgminShape& a : argmin_shapes) {
        if (!only.empty() && std::string(a.name).find(only) == std::string::npos) {
            continue;
        }
        const auto n = static_cast<std::size_t>(a.rows * a.width);
        bench::Arrays arrays(n, 1);
        float* x = arrays[0];
        bench::fill_a(x, n);
        x[n - 1] = -10.0f; // the last row's minimum at its very end
        std::vector<std::int64_t> want(static_cast<std::size_t>(a.rows)), got(want.size());
        table.section(std::string("argmin ") + a.name);
        cpu::set_max_threads(1);
        auto seq = [&] { cpu::argmin_f32_sequential(x, want.data(), a.rows, a.width); };
        seq();
        settle();
        const bench::Sample base =
            bench::measure(config, seq, [&] { return checksum(want.data(), want.size()); });
        table.row("argmin", "seq", "1", base, 4.0 * n, &base);
        for (int threads : thread_counts) {
            cpu::set_max_threads(threads);
            auto par = [&] { cpu::argmin_f32_parallel(x, got.data(), a.rows, a.width); };
            par();
            if (got != want) {
                throw std::runtime_error(std::string("argmin mismatch: ") + a.name);
            }
            settle();
            const bench::Sample sample =
                bench::measure(config, par, [&] { return checksum(got.data(), got.size()); });
            table.row("argmin", "par", std::to_string(threads), sample, 4.0 * n, &base);
        }
    }

    for (const MomentShape& m : moment_shapes) {
        if (!only.empty() && std::string(m.name).find(only) == std::string::npos) {
            continue;
        }
        std::int64_t total = 1;
        for (std::int64_t size : m.shape) {
            total *= size;
        }
        const auto n = static_cast<std::size_t>(total);
        const auto outputs = static_cast<std::size_t>(total / m.shape[static_cast<std::size_t>(m.axis)]);
        const auto rank = static_cast<std::int64_t>(m.shape.size());
        bench::Arrays arrays(n, 4);
        float* x = arrays[0];
        float* out = arrays[1];
        float* grad = arrays[2];
        float* grad_x = arrays[3];
        bench::fill_a(x, n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += 1000.0f; // a mean far from zero, as activations often have
        }
        bench::fill_b(grad, outputs);
        table.section(std::string("var/std ") + m.name);

        auto sweep = [&](const char* pass, double bytes, auto seq_kernel, auto par_kernel,
                         float* result, std::size_t length) {
            cpu::set_max_threads(1);
            seq_kernel();
            std::vector<float> want(result, result + length);
            settle();
            const bench::Sample base = bench::measure(
                config, seq_kernel, [&] { return bench::checksum(result, length); });
            table.row(pass, "seq", "1", base, bytes, &base);
            for (int threads : thread_counts) {
                cpu::set_max_threads(threads);
                par_kernel();
                bench::check_close(want.data(), result, length, 1e-5,
                                   std::string(m.name) + " " + pass);
                settle();
                const bench::Sample sample = bench::measure(
                    config, par_kernel, [&] { return bench::checksum(result, length); });
                table.row(pass, "par", std::to_string(threads), sample, bytes, &base);
            }
        };
        const std::int64_t* shape = m.shape.data();
        sweep(
            "var", 4.0 * n,
            [&] { cpu::var_axis_f32_sequential(x, out, shape, rank, m.axis, 1); },
            [&] { cpu::var_axis_f32_parallel(x, out, shape, rank, m.axis, 1); }, out, outputs);
        sweep(
            "std", 4.0 * n,
            [&] { cpu::std_axis_f32_sequential(x, out, shape, rank, m.axis, 1); },
            [&] { cpu::std_axis_f32_parallel(x, out, shape, rank, m.axis, 1); }, out, outputs);
        sweep(
            "var backward", 8.0 * n,
            [&] { cpu::var_backward_f32_sequential(x, grad, grad_x, shape, rank, m.axis, 1); },
            [&] { cpu::var_backward_f32_parallel(x, grad, grad_x, shape, rank, m.axis, 1); },
            grad_x, n);
        sweep(
            "std backward", 8.0 * n,
            [&] { cpu::std_backward_f32_sequential(x, grad, grad_x, shape, rank, m.axis, 1); },
            [&] { cpu::std_backward_f32_parallel(x, grad, grad_x, shape, rank, m.axis, 1); },
            grad_x, n);
    }
    cpu::set_max_threads(maximum_threads);
}
