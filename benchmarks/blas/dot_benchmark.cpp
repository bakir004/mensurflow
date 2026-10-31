// Benchmarks mensurflow::dot on every available device, through the public API, so the numbers
// include the dispatcher. Three questions:
//   1. How fast is each device, and how much faster than the CPU?            (table 1)
//   2. What does the abstraction cost per call?                              (table 2)
//   3. On the CPU, how does the API compare with calling the kernel directly
//      and with a plain serial loop?                                          (table 3)
// Each timed call includes reading the scalar result back, so GPU work is finished when the
// clock stops. Results are verified against a float64 reference outside the timing.
#include "backends/cpu/blas.hpp"
#include "backends/cpu/parallel.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/blas.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace mensurflow;

namespace {

using Clock = std::chrono::steady_clock;

const char* compiler() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#elif defined(_MSC_VER)
    return "msvc";
#else
    return "unknown";
#endif
}

// Median seconds per call. Fast calls are repeated until a sample lasts at least ~2 ms.
template <typename F> double median_seconds(F&& call) {
    call(); // warm up caches, thread pool, and any lazy GPU initialization
    int reps = 1;
    for (;;) {
        const auto start = Clock::now();
        for (int r = 0; r < reps; ++r) {
            call();
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        if (elapsed >= 2e-3 || reps >= (1 << 20)) {
            break;
        }
        reps *= 2;
    }
    std::vector<double> samples;
    for (int s = 0; s < 7; ++s) {
        const auto start = Clock::now();
        for (int r = 0; r < reps; ++r) {
            call();
        }
        samples.push_back(std::chrono::duration<double>(Clock::now() - start).count() / reps);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::vector<float> make_input(std::int64_t n, float scale) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = (static_cast<float>(i % 13) - 6.0f) / 8 * scale;
    }
    return v;
}

// Sink so the compiler cannot discard the direct and serial calls.
volatile float sink = 0;

} // namespace

int main() {
    const auto devices = list_devices();
    const std::int64_t sizes[] = {1, 1000, 100000, 1000000, 10000000, 33554432};

    std::cout << "dot benchmark (float32), through mensurflow::dot\n"
              << "compiler: " << compiler() << ", OpenMP: " << (cpu::has_openmp() ? "yes" : "no")
              << ", CPU threads: " << cpu::max_threads()
              << ", hardware threads: " << std::thread::hardware_concurrency() << "\nDevices:";
    for (Device d : devices) {
        std::cout << ' ' << to_string(d);
    }
    std::cout << "\nMedian of 7 samples; each call includes reading the result back.\n"
              << "GB/s counts 8 bytes per element (two float reads); GFLOP/s counts 2.\n"
              << std::fixed << std::setprecision(3);

    std::vector<double> cpu_seconds(std::size(sizes), 0.0);

    // 1. Throughput per device.
    std::cout << "\n== 1. Throughput by device ==\n"
              << std::setw(10) << "device" << std::setw(12) << "n" << std::setw(12) << "ms"
              << std::setw(10) << "GB/s" << std::setw(10) << "GFLOP/s" << std::setw(12)
              << "vs cpu:0" << '\n';
    for (Device device : devices) {
        for (std::size_t k = 0; k < std::size(sizes); ++k) {
            const std::int64_t n = sizes[k];
            const auto a = make_input(n, 1.0f);
            const auto b = make_input(n, 0.5f);
            const Tensor ta = Tensor::from_vector(a, {n}, device);
            const Tensor tb = Tensor::from_vector(b, {n}, device);

            const double seconds = median_seconds([&] { sink = dot(ta, tb).item<float>(); });

            // Checked after timing so verification never skews the measurement.
            const auto want = reference::dot(a.data(), b.data(), n);
            if (std::abs(static_cast<double>(dot(ta, tb).item<float>()) - want.value) >
                reference::dot_tolerance(want)) {
                std::cerr << to_string(device) << " n=" << n << ": result differs from reference\n";
                return 1;
            }
            if (device == Device{DeviceType::CPU, 0}) {
                cpu_seconds[k] = seconds;
            }
            const double elements = static_cast<double>(n);
            std::cout << std::setw(10) << to_string(device) << std::setw(12) << n << std::setw(12)
                      << seconds * 1e3 << std::setw(10) << 8.0 * elements / seconds / 1e9
                      << std::setw(10) << 2.0 * elements / seconds / 1e9 << std::setw(11)
                      << cpu_seconds[k] / seconds << "x\n";
        }
        std::cout << '\n';
    }

    // 2. Fixed cost of one call: allocate the scalar, validate, dispatch, run, read back.
    std::cout << "== 2. Cost of one call on 1 element (pure overhead) ==\n"
              << std::setw(10) << "device" << std::setw(14) << "microseconds" << '\n';
    for (Device device : devices) {
        const Tensor one = Tensor::from_vector(std::vector<float>{2.0f}, {1}, device);
        const double seconds = median_seconds([&] { sink = dot(one, one).item<float>(); });
        std::cout << std::setw(10) << to_string(device) << std::setw(14) << seconds * 1e6 << '\n';
    }

    // 3. The same CPU work three ways: the abstraction's price on the CPU.
    std::cout << "\n== 3. CPU: API vs direct kernel vs serial loop ==\n"
              << std::setw(12) << "n" << std::setw(12) << "api ms" << std::setw(12) << "kernel ms"
              << std::setw(12) << "serial ms" << std::setw(14) << "api/kernel" << '\n';
    for (std::int64_t n : sizes) {
        const auto a = make_input(n, 1.0f);
        const auto b = make_input(n, 0.5f);
        const Tensor ta = Tensor::from_vector(a, {n});
        const Tensor tb = Tensor::from_vector(b, {n});
        const double api = median_seconds([&] { sink = dot(ta, tb).item<float>(); });
        const double kernel = median_seconds([&] { sink = cpu::dot_f32(a.data(), b.data(), n); });
        const double serial = median_seconds(
            [&] { sink = static_cast<float>(reference::dot(a.data(), b.data(), n).value); });
        std::cout << std::setw(12) << n << std::setw(12) << api * 1e3 << std::setw(12)
                  << kernel * 1e3 << std::setw(12) << serial * 1e3 << std::setw(13) << api / kernel
                  << "x\n";
    }
    return 0;
}
