// mensurflow::dot on every available device, through the public API, so the numbers include the
// dispatcher. Three questions:
//   1. How fast is each device, and how much faster than the CPU?              (section 1)
//   2. What does one call cost when there is no work to do?                    (section 2)
//   3. On the CPU, what does the API cost compared with calling the kernel directly, and what
//      do SIMD and threads each add?                                           (section 3)
// Each timed call includes reading the scalar result back, so GPU work is finished when the clock
// stops. Built on the shared harness: see benchmarks/bench.hpp and docs/BENCHMARKS.md.
#include "bench.hpp"

#include "backends/cpu/blas.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/scalar.hpp"
#include "backends/cpu/sequential.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/blas.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
using mensurflow::Device;
using mensurflow::DeviceType;
using mensurflow::Tensor;

int main() {
    const auto devices = mensurflow::list_devices();
    const std::int64_t sizes[] = {1, 1000, 100000, 1000000, 10000000, 33554432};
    const bench::Config config;

    std::string device_list;
    for (Device d : devices) {
        device_list += " " + mensurflow::to_string(d);
    }
    bench::print_header(
        "dot (float32) through mensurflow::dot", config,
        {"devices:" + device_list,
         "each call includes reading the result back; GB/s counts 8 bytes per element "
         "(two float reads)"});

    // 1. Throughput per device. Speedup is against the same size on cpu:0.
    bench::Table by_device("GB/s", 1e9, "vs cpu:0", "");
    by_device.section("1. dot on every device");
    for (std::int64_t n : sizes) {
        const auto len = static_cast<std::size_t>(n);
        std::vector<float> a(len), b(len);
        bench::fill_a(a.data(), len);
        bench::fill_b(b.data(), len);
        const auto want = reference::dot(a.data(), b.data(), n);

        bench::Sample cpu_sample;
        bool have_cpu = false;
        for (Device device : devices) {
            const Tensor ta = Tensor::from_vector(a, {n}, device);
            const Tensor tb = Tensor::from_vector(b, {n}, device);
            double result = 0.0;
            auto run = [&] { result = mensurflow::dot(ta, tb).item<float>(); };
            run();
            // Float accumulation error is relative to the sum of |x*y|, not to the result.
            bench::check_value(want.value, result, 2e-4,
                               mensurflow::to_string(device) + " dot n=" + std::to_string(n),
                               want.magnitude);
            const bench::Sample sample = bench::measure(config, run, [&] { return result; });
            if (device == Device{DeviceType::CPU, 0}) {
                cpu_sample = sample;
                have_cpu = true;
            }
            by_device.row(std::to_string(n), mensurflow::to_string(device), "-", sample,
                          8.0 * static_cast<double>(n), have_cpu ? &cpu_sample : nullptr);
        }
        by_device.blank();
    }

    // 2. Fixed cost of one call: allocate the scalar, validate, dispatch, run, read back.
    bench::Table overhead("GB/s", 1e9, "", "");
    overhead.section("2. one call on 1 element (pure overhead)");
    for (Device device : devices) {
        const Tensor one = Tensor::from_vector(std::vector<float>{2.0f}, {1}, device);
        double result = 0.0;
        auto run = [&] { result = mensurflow::dot(one, one).item<float>(); };
        const bench::Sample sample = bench::measure(config, run, [&] { return result; });
        overhead.row("1", mensurflow::to_string(device), "-", sample, 0.0);
    }

    // 3. The same CPU work through the API and called directly. "vs api" is the API's median time
    // divided by this row's, so above 1x means faster than going through the API.
    bench::Table kernels("GB/s", 1e9, "vs api", "");
    kernels.section("3. CPU: API vs the kernels called directly");
    for (std::int64_t n : sizes) {
        const auto len = static_cast<std::size_t>(n);
        std::vector<float> a(len), b(len);
        bench::fill_a(a.data(), len);
        bench::fill_b(b.data(), len);
        const Tensor ta = Tensor::from_vector(a, {n});
        const Tensor tb = Tensor::from_vector(b, {n});
        const double bytes = 8.0 * static_cast<double>(n);
        double result = 0.0;
        auto observe = [&] { return result; };

        auto run_api = [&] { result = mensurflow::dot(ta, tb).item<float>(); };
        auto run_parallel = [&] { result = cpu::dot_f32(a.data(), b.data(), n); };
        auto run_sequential = [&] { result = cpu::sequential::dot_f32(a.data(), b.data(), n); };
        auto run_scalar = [&] { result = cpu::scalar::dot_f32(a.data(), b.data(), n); };

        run_sequential();
        const double expected = result;
        run_parallel();
        bench::check_value(expected, result, 1e-4, "parallel dot kernel n=" + std::to_string(n));
        run_scalar();
        bench::check_value(expected, result, 1e-4, "scalar dot kernel n=" + std::to_string(n));
        const bench::Sample api = bench::measure(config, run_api, observe);
        const std::string label = std::to_string(n);
        kernels.row(label, "api", "-", api, bytes, &api);
        kernels.row(label, "parallel", std::to_string(cpu::max_threads()),
                    bench::measure(config, run_parallel, observe), bytes, &api);
        kernels.row(label, "sequential", "-", bench::measure(config, run_sequential, observe),
                    bytes, &api);
        kernels.row(label, "scalar", "-", bench::measure(config, run_scalar, observe), bytes, &api);
        kernels.blank();
    }
    return 0;
}
