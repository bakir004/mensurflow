// Compares full AXPY API call times for the CPU and the automatically selected
// backend, with warmup, repeated measurements, and untimed correctness checks.
#include "prs/ops.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

// Batch small workloads to reduce timer overhead, then report time per call.
// GPU timings include allocation, data transfer, execution, and synchronization.
double measure(const std::vector<float>& x, std::vector<float>& y, prs::Backend backend,
               int batch) {
    const auto start = Clock::now();
    for (int i = 0; i < batch; ++i) {
        prs::axpy(0.5f, x, y, backend);
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / batch;
}

// Trials have odd counts, so the middle sorted sample is the median.
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void benchmark(std::size_t n, int batch, prs::Backend backend) {
    std::vector<float> x(n), cpu(n, 0), device(n, 0);
    // Binary fractions keep repeated updates exact and inputs reproducible.
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 17) * 0.25f;
    }
    // Exclude first-use shader compilation and warm up both backends.
    for (int i = 0; i < 3; ++i) {
        prs::axpy(0.5f, x, cpu, prs::Backend::CPU);
        prs::axpy(0.5f, x, device, backend);
    }
    constexpr int trials = 11;
    std::vector<double> cpu_times, backend_times;
    for (int trial = 0; trial < trials; ++trial) {
        // Alternate ordering to reduce systematic ordering bias.
        if (trial % 2 == 0) {
            cpu_times.push_back(measure(x, cpu, prs::Backend::CPU, batch));
            backend_times.push_back(measure(x, device, backend, batch));
        } else {
            backend_times.push_back(measure(x, device, backend, batch));
            cpu_times.push_back(measure(x, cpu, prs::Backend::CPU, batch));
        }
    }
    // Validate outside the timed section. Values are exactly representable here.
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = static_cast<float>(3 + trials * batch) * 0.5f * x[i];
        if (cpu[i] != expected || device[i] != expected) {
            throw std::runtime_error("Benchmark correctness check failed");
        }
    }
    const double cpu_ms = median(cpu_times), backend_ms = median(backend_times);
    std::cout << std::setw(12) << n << std::setw(16) << cpu_ms << std::setw(16) << backend_ms
              << std::setw(16) << cpu_ms / backend_ms << '\n';
}
} // namespace

int main() {
    try {
        // Auto selects CUDA, Metal, or CPU according to runtime availability.
        const auto backend = prs::resolve_backend();
        std::cout << "Selected backend: " << prs::backend_name(backend)
                  << "\nMedian full API call time, milliseconds (11 trials).\n"
                  << "Includes GPU allocations, copies, execution, and synchronization.\n"
                  << "Excludes first-use compilation; speedup = CPU time / GPU time.\n"
                  << "CPU uses OpenMP if found at build time; otherwise a serial loop.\n\n"
                  << std::setw(12) << "Elements" << std::setw(16) << "CPU ms" << std::setw(16)
                  << "Backend ms" << std::setw(16) << "Speedup" << '\n'
                  << std::fixed << std::setprecision(6);
        if (backend == prs::Backend::CPU) {
            std::cout << "No GPU available; comparing two CPU runs.\n";
        }
        // Contrast launch-dominated small inputs with a large memory workload.
        benchmark(1024, 100, backend);
        benchmark(std::size_t{16} * 1024 * 1024, 1, backend);
        std::cout << "Correctness checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
