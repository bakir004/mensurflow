// Compare CPU and automatically selected backend performance through prs::axpy().
// Report median full-call time and verify that repeated in-place updates stay correct.
#include "prs/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
// A monotonic clock avoids timing changes caused by system-clock adjustments.
using Clock = std::chrono::steady_clock;

// Batch short operations for steadier timing, then return milliseconds per API call.
// GPU calls include temporary allocations, transfers, execution, and synchronization.
double measure(const std::vector<float>& x, std::vector<float>& y,
               prs::Backend backend, int batch) {
    const auto start = Clock::now();
    for (int i = 0; i < batch; ++i) prs::axpy(0.5f, x, y, backend);
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / batch;
}

// The middle of an odd-sized sample set is less sensitive to occasional slow calls.
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Keep separate outputs so CPU and selected-backend calls perform equal AXPY updates.
// The gpu parameter can also be CPU when Auto finds no usable GPU.
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
    // A ratio above 1 means the selected backend was faster than the CPU measurement.
    std::cout << std::setw(12) << n << std::setw(16) << cpu_ms
              << std::setw(16) << gpu_ms << std::setw(16) << cpu_ms / gpu_ms << '\n';
}
} // namespace

int main() {
    try {
        // Auto selects available hardware once; CPU fallback keeps the benchmark runnable.
        const auto gpu = prs::resolve_backend();
        std::cout << "Selected backend: " << prs::backend_name(gpu)
                  << "\nMedian full API call time, milliseconds (11 trials).\n"
                  << "Includes GPU allocations, copies, execution, and synchronization.\n"
                  << "Excludes first-use compilation; speedup = CPU time / GPU time.\n"
                  << "CPU uses OpenMP if found at build time; otherwise a serial loop.\n\n"
                  << std::setw(12) << "Elements" << std::setw(16) << "CPU ms"
                  << std::setw(16) << "Backend ms" << std::setw(16) << "Speedup" << '\n'
                  << std::fixed << std::setprecision(6);
        if (gpu == prs::Backend::CPU)
            std::cout << "No GPU available; comparing two CPU runs.\n";
        // Batch the small workload; time the large workload one API call at a time.
        benchmark(1024, 100, gpu);
        benchmark(16 * 1024 * 1024, 1, gpu);
        std::cout << "Correctness checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
