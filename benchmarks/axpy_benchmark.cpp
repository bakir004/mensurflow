#include "prs/axpy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double measure(const std::vector<float>& x, std::vector<float>& y,
               prs::Backend backend, int batch) {
    const auto start = Clock::now();
    for (int i = 0; i < batch; ++i) prs::axpy(0.5f, x, y, backend);
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / batch;
}

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void benchmark(std::size_t n, int batch, prs::Backend gpu) {
    std::vector<float> x(n), cpu(n, 0), device(n, 0);
    for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<float>(i % 17) * 0.25f;
    // Exclude first-use shader compilation and warm up both backends.
    for (int i = 0; i < 3; ++i) {
        prs::axpy(0.5f, x, cpu, prs::Backend::CPU);
        prs::axpy(0.5f, x, device, gpu);
    }
    constexpr int trials = 11;
    std::vector<double> cpu_times, gpu_times;
    for (int trial = 0; trial < trials; ++trial) {
        // Alternate ordering to reduce systematic ordering bias.
        if (trial % 2 == 0) {
            cpu_times.push_back(measure(x, cpu, prs::Backend::CPU, batch));
            gpu_times.push_back(measure(x, device, gpu, batch));
        } else {
            gpu_times.push_back(measure(x, device, gpu, batch));
            cpu_times.push_back(measure(x, cpu, prs::Backend::CPU, batch));
        }
    }
    // Validate outside the timed section. Values are exactly representable here.
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = (3 + trials * batch) * 0.5f * x[i];
        if (cpu[i] != expected || device[i] != expected) {
            throw std::runtime_error("Benchmark correctness check failed");
        }
    }
    const double cpu_ms = median(cpu_times), gpu_ms = median(gpu_times);
    std::cout << std::setw(12) << n << std::setw(16) << cpu_ms
              << std::setw(16) << gpu_ms << std::setw(16) << cpu_ms / gpu_ms << '\n';
}
} // namespace

int main() {
    try {
        const auto gpu = prs::metal_available() ? prs::Backend::Metal : prs::Backend::CUDA;
        if (gpu == prs::Backend::CUDA && !prs::cuda_available()) {
            throw std::runtime_error("No GPU backend/device available");
        }
        std::cout << "GPU: " << (gpu == prs::Backend::Metal ? "Metal" : "CUDA")
                  << "\nMedian full API call time, milliseconds (11 trials).\n"
                  << "Includes GPU allocations, copies, execution, and synchronization.\n"
                  << "Excludes first-use compilation; speedup = CPU time / GPU time.\n"
                  << "CPU uses OpenMP if found at build time; otherwise a serial loop.\n\n"
                  << std::setw(12) << "Elements" << std::setw(16) << "CPU ms"
                  << std::setw(16) << "GPU ms" << std::setw(16) << "GPU speedup" << '\n'
                  << std::fixed << std::setprecision(6);
        benchmark(1024, 100, gpu);
        benchmark(16 * 1024 * 1024, 1, gpu);
        std::cout << "Correctness checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
