// Compares full row-major matmul API call times for the CPU and Auto backend.
// Reports median timings for several shapes and validates results outside timing.
#include "prs/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

// The synchronous API returns after all work and output transfers are complete.
template <typename F> double time_ms(F operation) {
    const auto start = Clock::now();
    operation();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void validate(const std::vector<float>& cpu, const std::vector<float>& device) {
    // Tolerate float accumulation differences while rejecting nonfinite results.
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(device[i]) ||
            std::abs(cpu[i] - device[i]) > 1e-4f * std::max(1.0f, std::abs(cpu[i]))) {
            throw std::runtime_error("Matrix multiplication correctness check failed");
        }
    }
}

void benchmark(std::size_t m, std::size_t n, std::size_t k, prs::Backend backend) {
    std::vector<float> a(m * k), b(k * n), cpu(m * n), device(m * n);
    // Reproducible inputs use binary fractions to limit input rounding error.
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 8;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8;
    }
    auto run_cpu = [&] { prs::matmul(a, b, cpu, m, n, k, prs::Backend::CPU); };
    auto run_device = [&] { prs::matmul(a, b, device, m, n, k, backend); };
    run_cpu();
    run_device(); // Exclude first-use context/pipeline creation.
    validate(cpu, device);
    std::vector<double> cpu_times, device_times;
    for (int trial = 0; trial < 5; ++trial) {
        // Alternate execution order to reduce systematic ordering bias.
        if (trial % 2 == 0) {
            cpu_times.push_back(time_ms(run_cpu));
            device_times.push_back(time_ms(run_device));
        } else {
            device_times.push_back(time_ms(run_device));
            cpu_times.push_back(time_ms(run_cpu));
        }
        validate(cpu, device); // Outside timing after every trial.
    }
    // The third sorted sample is the median of the five measured calls.
    std::sort(cpu_times.begin(), cpu_times.end());
    std::sort(device_times.begin(), device_times.end());
    std::cout << std::setw(8) << m << std::setw(8) << n << std::setw(8) << k << std::setw(14)
              << cpu_times[2] << std::setw(14) << device_times[2] << std::setw(14)
              << cpu_times[2] / device_times[2] << '\n';
}
} // namespace

int main() {
    try {
        // Auto selects CUDA, Metal, or CPU; CPU fallback compares two CPU runs.
        const auto backend = prs::resolve_backend();
        std::cout << "Custom float32 matrix multiplication, C[M,N] = A[M,K] * B[K,N]\n"
                  << "Selected backend: " << prs::backend_name(backend) << '\n'
                  << "CPU: serial i-k-j loop; GPU: tiled 16x16 kernel.\n"
                  << "Median of 5 calls after warmup. GPU includes allocations and copies.\n"
                  << "Not a comparison with optimized CPU BLAS or cuBLAS.\n\n"
                  << std::setw(8) << "M" << std::setw(8) << "N" << std::setw(8) << "K"
                  << std::setw(14) << "CPU ms" << std::setw(14) << "Backend ms" << std::setw(14)
                  << "Speedup" << '\n'
                  << std::fixed << std::setprecision(3);
        if (backend == prs::Backend::CPU) {
            std::cout << "No GPU available; comparing two CPU runs.\n";
        }
        // Include partial GPU tiles, a small square case, and a larger workload.
        benchmark(17, 23, 19, backend);
        benchmark(64, 64, 64, backend);
        benchmark(1024, 1024, 1024, backend);
        std::cout << "Correctness checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
