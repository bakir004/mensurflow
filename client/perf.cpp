// A user's own performance check of mensurflow: only the public API and std::chrono, the way someone
// trying the library would measure it. Each op is warmed up once, then run until 0.3 s have passed
// (at least 5 times); the table shows the median time per call and the rate it implies.
//
// GPU work may finish after the op returns, so every timed call reads one element of its result,
// which waits for the device. Thread count: set OMP_NUM_THREADS before starting (default: all).
#include "mensurflow/mensurflow.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace mensurflow;
using Clock = std::chrono::steady_clock;

namespace {

// Reads one element, which waits until the op that produced `t` has finished on its device.
void wait_for(const Tensor& t) {
    if (t.numel() > 0) {
        (void)t.contiguous().view({t.numel()}).slice(0, 0, 1).to_vector<float>();
    }
}

// Median seconds per call of `run`, which returns the tensor it produced.
template <class Run> double time_it(Run&& run) {
    wait_for(run()); // warm-up: first-touch memory, GPU pipeline compilation
    std::vector<double> samples;
    const auto start = Clock::now();
    while (samples.size() < 5 ||
           std::chrono::duration<double>(Clock::now() - start).count() < 0.3) {
        const auto t0 = Clock::now();
        wait_for(run());
        samples.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
        if (samples.size() >= 1000) {
            break;
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void header(const char* title) {
    std::printf("\n== %s ==\n", title);
    std::printf("%-26s %-18s %-24s %12s %14s\n", "op", "size", "where", "median ms", "rate");
}

// rate_unit is "GB/s" (work in bytes) or "GFLOP/s" (work in flops).
void report(const std::string& op, const std::string& size, const std::string& where,
            double seconds, double work, const char* rate_unit) {
    std::printf("%-26s %-18s %-24s %12.4f %9.1f %s\n", op.c_str(), size.c_str(), where.c_str(),
                seconds * 1e3, work / seconds / 1e9, rate_unit);
}

std::vector<float> random_values(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (auto& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = static_cast<float>(seed >> 8) / 16777216.0f - 0.5f;
    }
    return v;
}

Tensor random_tensor(Shape shape, std::uint32_t seed, Device device = {}) {
    return Tensor::from_vector(random_values(numel(shape), seed), shape, device);
}

std::string mega(std::int64_t n) { return std::to_string(n / (1 << 20)) + "M floats"; }

} // namespace

int main() {
    std::printf("mensurflow performance check\n");
    std::printf("devices:");
    Device gpu{};
    bool has_gpu = false;
    for (Device d : list_devices()) {
        std::printf(" %s", to_string(d).c_str());
        if (d.type != DeviceType::CPU && !has_gpu) {
            gpu = d;
            has_gpu = true;
        }
    }
    std::printf("\n");

    // ---- dot: CPU, GPU with data already there, GPU with data copied over every call ----------
    header("dot product (memory-bound: reads two vectors)");
    for (const std::int64_t n : {std::int64_t{1} << 20, std::int64_t{1} << 24}) {
        const Tensor a = random_tensor({n}, 1);
        const Tensor b = random_tensor({n}, 2);
        const double bytes = 8.0 * static_cast<double>(n);
        report("dot", mega(n), "cpu", time_it([&] { return dot(a, b, Execution::CPU); }), bytes,
               "GB/s");
        if (has_gpu) {
            const Tensor ga = a.to(gpu);
            const Tensor gb = b.to(gpu);
            report("dot", mega(n), "gpu, data on gpu",
                   time_it([&] { return dot(ga, gb, Execution::GPU); }), bytes, "GB/s");
            report("dot", mega(n), "gpu, data on cpu",
                   time_it([&] { return dot(a, b, Execution::GPU); }), bytes, "GB/s");
        }
    }

    // ---- elementwise -----------------------------------------------------------------------------
    header("elementwise (memory-bound: reads two, writes one)");
    {
        const std::int64_t n = std::int64_t{1} << 24;
        const Tensor a = random_tensor({n}, 3);
        const Tensor b = random_tensor({n}, 4);
        const double bytes = 12.0 * static_cast<double>(n);
        report("add", mega(n), "cpu", time_it([&] { return add(a, b, Execution::CPU); }), bytes,
               "GB/s");
        report("mul", mega(n), "cpu", time_it([&] { return mul(a, b, Execution::CPU); }), bytes,
               "GB/s");
        report("relu", mega(n), "cpu", time_it([&] { return relu(a, Execution::CPU); }),
               8.0 * static_cast<double>(n), "GB/s");
        report("exp", mega(n), "cpu", time_it([&] { return exp(a, Execution::CPU); }),
               8.0 * static_cast<double>(n), "GB/s");
        if (has_gpu) {
            const Tensor ga = a.to(gpu);
            const Tensor gb = b.to(gpu);
            report("add", mega(n), "gpu, data on gpu",
                   time_it([&] { return add(ga, gb, Execution::GPU); }), bytes, "GB/s");
        }
    }

    // ---- reductions ------------------------------------------------------------------------------
    header("reductions");
    {
        const std::int64_t n = std::int64_t{1} << 24;
        const Tensor a = random_tensor({n}, 5);
        const double bytes = 4.0 * static_cast<double>(n);
        report("sum", mega(n), "cpu", time_it([&] { return sum(a); }), bytes, "GB/s");
        report("max", mega(n), "cpu", time_it([&] { return max(a); }), bytes, "GB/s");
        const Tensor m = random_tensor({4096, 4096}, 6);
        report("sum along rows (axis 1)", "4096x4096", "cpu", time_it([&] { return sum(m, 1); }),
               bytes, "GB/s");
    }

    // ---- matrix multiplication ---------------------------------------------------------------------
    header("matrix multiplication (compute-bound: 2*n^3 flops)");
    for (const std::int64_t n : {128, 512, 1024, 2048}) {
        const Tensor a = random_tensor({n, n}, 7);
        const Tensor b = random_tensor({n, n}, 8);
        const double flops =
            2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        const std::string size = std::to_string(n) + "x" + std::to_string(n);
        report("matmul", size, "cpu", time_it([&] { return matmul(a, b); }), flops, "GFLOP/s");
    }
    {
        const Tensor a = random_tensor({1024, 1024}, 9);
        const Tensor b = random_tensor({1024, 1024}, 10);
        report("matmul, b transposed view", "1024x1024", "cpu",
               time_it([&] { return matmul(a, b.transpose(0, 1)); }), 2.0 * 1024 * 1024 * 1024,
               "GFLOP/s");
    }

    // ---- batched matmul vs a loop of matmuls --------------------------------------------------------
    header("batched matmul vs calling matmul per matrix");

    struct Batch {
        std::int64_t batch, m, n, k;
    };

    for (const Batch s : {Batch{1024, 8, 8, 8}, Batch{64, 128, 128, 64}, Batch{8, 512, 512, 512}}) {
        const Tensor a = random_tensor({s.batch, s.m, s.k}, 11);
        const Tensor b = random_tensor({s.batch, s.k, s.n}, 12);
        const double flops = 2.0 * static_cast<double>(s.batch) * static_cast<double>(s.m) *
                             static_cast<double>(s.n) * static_cast<double>(s.k);
        const std::string size = std::to_string(s.batch) + "x" + std::to_string(s.m) + "x" +
                                 std::to_string(s.n) + "x" + std::to_string(s.k);
        report("gemm_strided_batched", size, "cpu",
               time_it([&] { return gemm_strided_batched(a, b); }), flops, "GFLOP/s");
        // What a user without a batched op writes: slice out each matrix and multiply it.
        std::vector<Tensor> lefts, rights;
        for (std::int64_t i = 0; i < s.batch; ++i) {
            lefts.push_back(a.slice(0, i, i + 1).view({s.m, s.k}));
            rights.push_back(b.slice(0, i, i + 1).view({s.k, s.n}));
        }
        report("matmul in a loop", size, "cpu", time_it([&] {
                   Tensor last;
                   for (std::size_t i = 0; i < lefts.size(); ++i) {
                       last = matmul(lefts[i], rights[i]);
                   }
                   return last;
               }),
               flops, "GFLOP/s");
    }

    // ---- data movement -------------------------------------------------------------------------------
    header("data movement (memory-bound: reads input, writes output)");
    {
        const Tensor x = random_tensor({4096, 4096}, 13);
        const Tensor y = random_tensor({4096, 4096}, 14);
        const double moved = 2.0 * 2.0 * 4.0 * 4096 * 4096;
        report("concat, axis 0", "2 x 4096x4096", "cpu", time_it([&] { return concat({x, y}, 0); }),
               moved, "GB/s");
        report("concat, axis 1", "2 x 4096x4096", "cpu", time_it([&] { return concat({x, y}, 1); }),
               moved, "GB/s");
        report("pad by 1 on every side", "4096x4096", "cpu",
               time_it([&] { return pad(x, {1, 1, 1, 1}); }), 2.0 * 4.0 * 4096 * 4096, "GB/s");
        const Tensor images = random_tensor({32, 64, 56, 56}, 15);
        report("pad images by 1 pixel", "32x64x56x56", "cpu",
               time_it([&] { return pad(images, {1, 1, 1, 1}); }),
               4.0 * 32 * 64 * (56 * 56 + 58 * 58), "GB/s");
        report("split into 8 (views)", "4096x4096", "cpu",
               time_it([&] { return split(x, 512)[0]; }), 0.0, "GB/s");
        report("transpose + contiguous", "4096x4096", "cpu",
               time_it([&] { return x.transpose(0, 1).contiguous(); }), 2.0 * 4.0 * 4096 * 4096,
               "GB/s");
    }

    std::printf(
        "\nRates: GB/s counts bytes read and written once; GFLOP/s counts multiply-adds as 2.\n");
    return 0;
}
