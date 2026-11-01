// mensurflow::add and mul on every available device, through the public API, so the numbers are
// what a user of the library sees: allocating the result, dispatching, and running the kernel.
// Each call reads one element of the result back, which waits for a GPU to finish. GB/s counts 12
// bytes per element (two reads, one write). Speedup is against the same size on cpu:0.
//
// For the CPU kernels alone (scalar vs sequential vs parallel, every op, thread sweep) use
// cpu_parallel_benchmark. Built on the shared harness: see benchmarks/bench.hpp and
// docs/BENCHMARKING.md.
#include "bench.hpp"

#include "mensurflow/mensurflow.hpp"
#include "reference/elementwise.hpp"

#include <cstdint>
#include <string>
#include <vector>

using mensurflow::Device;
using mensurflow::DeviceType;
using mensurflow::Tensor;

namespace {

using Binary = Tensor (*)(const Tensor&, const Tensor&);
using Reference = void (*)(const float*, const float*, float*, std::int64_t);

struct Op {
    const char* name;
    Binary apply;
    Reference reference;
};

} // namespace

int main() {
    const auto devices = mensurflow::list_devices();
    const std::int64_t sizes[] = {1000, 100000, 1000000, 10000000};
    const bench::Config config;

    std::string device_list;
    for (Device d : devices) {
        device_list += " " + mensurflow::to_string(d);
    }
    bench::print_header("add and mul on every device (float32) through the public API", config,
                        {"devices:" + device_list,
                         "each call includes allocating the result and reading one element back"});

    const Op ops[] = {{"add", mensurflow::add, reference::add},
                      {"mul", mensurflow::mul, reference::mul}};
    bench::Table table("GB/s", 1e9, "vs cpu:0", "");

    for (const Op& op : ops) {
        table.section(op.name);
        for (std::int64_t n : sizes) {
            const auto len = static_cast<std::size_t>(n);
            std::vector<float> a(len), b(len), expected(len);
            bench::fill_a(a.data(), len);
            bench::fill_b(b.data(), len);
            op.reference(a.data(), b.data(), expected.data(), n);

            bench::Sample cpu_sample;
            bool have_cpu = false;
            for (Device device : devices) {
                const Tensor ta = Tensor::from_vector(a, {n}, device);
                const Tensor tb = Tensor::from_vector(b, {n}, device);

                // The CPU must match the reference exactly; a vendor library may differ in the
                // last bit.
                const std::vector<float> got = op.apply(ta, tb).to_vector<float>();
                const std::string what =
                    mensurflow::to_string(device) + " " + op.name + " n=" + std::to_string(n);
                if (device.type == DeviceType::CPU) {
                    bench::check_bits(expected.data(), got.data(), len, what);
                } else {
                    bench::check_close(expected.data(), got.data(), len, 1e-6, what);
                }

                double result = 0.0;
                auto run = [&] { result = op.apply(ta, tb).slice(0, 0, 1).item<float>(); };
                const bench::Sample sample = bench::measure(config, run, [&] { return result; });
                if (device == Device{DeviceType::CPU, 0}) {
                    cpu_sample = sample;
                    have_cpu = true;
                }
                table.row(std::to_string(n), mensurflow::to_string(device), "-", sample,
                          12.0 * static_cast<double>(n), have_cpu ? &cpu_sample : nullptr);
            }
            table.blank();
        }
    }
    return 0;
}
