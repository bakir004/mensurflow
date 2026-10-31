// Measures add and mul across sizes and thread counts. Prints the machine and build details first
// so results from different team members can be compared. Elementwise ops are memory-bound, so
// the GB/s column (3 arrays of 4-byte floats touched per element) is the number to watch.
// The last table runs mensurflow::add/mul on every available device.
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/parallel.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/elementwise.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

using namespace mensurflow::cpu;
using mensurflow::Device;
using mensurflow::DeviceType;
using mensurflow::list_devices;
using mensurflow::Tensor;
using mensurflow::to_string;

namespace {

using Clock = std::chrono::steady_clock;
using Kernel = void (*)(const float*, const float*, float*, std::int64_t);

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

// Median seconds per call. Small inputs finish in nanoseconds, so each sample repeats the call
// until it lasts at least ~2 ms.
double median_seconds(Kernel kernel, const float* a, const float* b, float* out, std::int64_t n) {
    kernel(a, b, out, n); // warm up caches and the thread pool
    int reps = 1;
    for (;;) {
        const auto start = Clock::now();
        for (int r = 0; r < reps; ++r) {
            kernel(a, b, out, n);
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
            kernel(a, b, out, n);
        }
        samples.push_back(std::chrono::duration<double>(Clock::now() - start).count() / reps);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Same timing scheme for any callable, used for the public mensurflow API.
template <typename F> double median_call(F&& call) {
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

// Sink so the compiler cannot discard results that are only read for synchronization.
volatile float sink = 0;

} // namespace

int main() {
    const int default_threads = max_threads();
    std::vector<int> thread_counts{1};
    for (int t = 2; t < default_threads; t *= 2) {
        thread_counts.push_back(t);
    }
    if (default_threads > 1) {
        thread_counts.push_back(default_threads);
    }

    std::cout << "Elementwise float32 benchmark\n"
              << "compiler: " << compiler() << "\nOpenMP: " << (has_openmp() ? "yes" : "no")
              << ", max threads: " << default_threads
              << ", hardware threads: " << std::thread::hardware_concurrency()
              << ", parallel cutoff: " << default_grain << " elements\n"
              << "Median of 7 samples. GB/s counts 12 bytes per element (2 reads, 1 write).\n\n"
              << std::setw(6) << "op" << std::setw(12) << "n" << std::setw(9) << "threads"
              << std::setw(12) << "ms" << std::setw(10) << "GB/s" << std::setw(10) << "speedup"
              << '\n'
              << std::fixed << std::setprecision(3);

    struct Op {
        const char* name;
        Kernel kernel;
        Kernel reference;
    };

    const Op ops[] = {{"add", add_f32, reference::add}, {"mul", mul_f32, reference::mul}};

    for (const Op& op : ops) {
        for (std::int64_t n : {1000, 10000, 100000, 1000000, 10000000, 33554432}) {
            std::vector<float> a(static_cast<std::size_t>(n)), b(a.size()), out(a.size()),
                expected(a.size());
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] = static_cast<float>(i % 13) / 8 - 0.75f;
                b[i] = static_cast<float>(i % 17) / 8 - 1.0f;
            }
            op.reference(a.data(), b.data(), expected.data(), n);

            double single_thread = 0;
            for (int threads : thread_counts) {
                set_max_threads(threads);
                const double seconds = median_seconds(op.kernel, a.data(), b.data(), out.data(), n);
                // Checked after timing, so verification never skews the measurement.
                if (std::memcmp(out.data(), expected.data(), out.size() * sizeof(float)) != 0) {
                    std::cerr << op.name << " n=" << n << " threads=" << threads
                              << ": result differs from reference\n";
                    return 1;
                }
                if (threads == 1) {
                    single_thread = seconds;
                }
                std::cout << std::setw(6) << op.name << std::setw(12) << n << std::setw(9)
                          << threads << std::setw(12) << seconds * 1e3 << std::setw(10)
                          << 12.0 * static_cast<double>(n) / seconds / 1e9 << std::setw(9)
                          << single_thread / seconds << "x\n";
            }
            std::cout << '\n';
        }
    }
    // Through the public API on every device: the numbers a user of mensurflow::add/mul sees,
    // including allocating the result and dispatching. Each call reads one element of the
    // result back, which waits for a GPU to finish.
    set_max_threads(default_threads);
    std::cout << "== Public API on every device ==\n"
              << std::setw(10) << "device" << std::setw(6) << "op" << std::setw(12) << "n"
              << std::setw(12) << "ms" << std::setw(10) << "GB/s" << std::setw(12) << "vs cpu:0"
              << '\n';

    struct ApiOp {
        const char* name;
        Tensor (*apply)(const Tensor&, const Tensor&);
        Kernel reference;
    };

    const ApiOp api_ops[] = {{"add", mensurflow::add, reference::add},
                             {"mul", mensurflow::mul, reference::mul}};
    const std::int64_t api_sizes[] = {1000, 100000, 1000000, 10000000};
    std::vector<double> cpu_seconds(std::size(api_ops) * std::size(api_sizes), 0.0);
    for (const Device device : list_devices()) {
        for (std::size_t o = 0; o < std::size(api_ops); ++o) {
            for (std::size_t k = 0; k < std::size(api_sizes); ++k) {
                const std::int64_t n = api_sizes[k];
                std::vector<float> a(static_cast<std::size_t>(n)), b(a.size()), expected(a.size());
                for (std::size_t i = 0; i < a.size(); ++i) {
                    a[i] = static_cast<float>(i % 13) / 8 - 0.75f;
                    b[i] = static_cast<float>(i % 17) / 8 - 1.0f;
                }
                api_ops[o].reference(a.data(), b.data(), expected.data(), n);
                const Tensor ta = Tensor::from_vector(a, {n}, device);
                const Tensor tb = Tensor::from_vector(b, {n}, device);

                const double seconds = median_call(
                    [&] { sink = api_ops[o].apply(ta, tb).slice(0, 0, 1).item<float>(); });

                // Checked after timing; vendor libraries may differ from the reference in the
                // last bit, so only the CPU must match exactly.
                const auto got = api_ops[o].apply(ta, tb).to_vector<float>();
                for (std::size_t i = 0; i < got.size(); ++i) {
                    const float tolerance =
                        device.type == DeviceType::CPU ? 0.0f : 4e-7f * std::abs(expected[i]);
                    if (std::abs(got[i] - expected[i]) > tolerance) {
                        std::cerr << to_string(device) << ' ' << api_ops[o].name << " n=" << n
                                  << ": result differs from reference at " << i << '\n';
                        return 1;
                    }
                }
                const std::size_t slot = o * std::size(api_sizes) + k;
                if (device == Device{DeviceType::CPU, 0}) {
                    cpu_seconds[slot] = seconds;
                }
                std::cout << std::setw(10) << to_string(device) << std::setw(6) << api_ops[o].name
                          << std::setw(12) << n << std::setw(12) << seconds * 1e3 << std::setw(10)
                          << 12.0 * static_cast<double>(n) / seconds / 1e9 << std::setw(11)
                          << cpu_seconds[slot] / seconds << "x\n";
            }
        }
        std::cout << '\n';
    }
    return 0;
}
