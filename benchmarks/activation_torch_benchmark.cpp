// Native C++ comparison: direct MensurFlow CPU kernels and preallocated ATen out operations.
// Both sides share input/output addresses, parameters, thread settings, observer and timer.
// ATen operator dispatch is timed; Tensor allocation/wrapping and Python are outside the timer.
#include "bench.hpp"

#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/threads.hpp"
#include "reference/elementwise.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/Version.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct Operation {
    const char* name;
    void (*ours)(const float*, float*, std::int64_t);
    void (*torch)(const at::Tensor&, at::Tensor&);
    void (*reference)(const float*, float*, std::int64_t);
};

const Operation operations[] = {
    {"leaky_relu",
     [](const float* x, float* y, std::int64_t n) { cpu::leaky_relu_f32(x, y, n, 0.01f); },
     [](const at::Tensor& x, at::Tensor& y) { at::leaky_relu_out(y, x, 0.01f); },
     reference::leaky_relu},
    {"gelu",
     [](const float* x, float* y, std::int64_t n) { cpu::gelu_f32(x, y, n); },
     [](const at::Tensor& x, at::Tensor& y) { at::gelu_out(y, x, "none"); }, reference::gelu},
    {"gelu_tanh",
     [](const float* x, float* y, std::int64_t n) { cpu::gelu_f32(x, y, n, true); },
     [](const at::Tensor& x, at::Tensor& y) { at::gelu_out(y, x, "tanh"); }, reference::gelu_tanh},
    {"silu", cpu::silu_f32,
     [](const at::Tensor& x, at::Tensor& y) { at::silu_out(y, x); }, reference::silu},
    {"elu",
     [](const float* x, float* y, std::int64_t n) { cpu::elu_f32(x, y, n, 1.0f); },
     [](const at::Tensor& x, at::Tensor& y) { at::elu_out(y, x, 1.0f, 1.0f, 1.0f); },
     reference::elu},
    {"softplus",
     [](const float* x, float* y, std::int64_t n) { cpu::softplus_f32(x, y, n, 1.0f, 20.0f); },
     [](const at::Tensor& x, at::Tensor& y) { at::softplus_out(y, x, 1.0f, 20.0f); },
     reference::softplus}};

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t parsed = 0;
        const auto value = std::stoll(token, &parsed);
        if (value <= 0 || parsed != token.size()) {
            throw std::invalid_argument("list values must be positive integers");
        }
        values.push_back(value);
    }
    if (values.empty() || text.back() == ',') {
        throw std::invalid_argument("expected a nonempty comma-separated list");
    }
    return values;
}

void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

int run(int argc, char** argv) {
    const int maximum_threads = cpu::max_threads();
    std::vector<std::int64_t> sizes{1000, 32769, 1000000};
    std::vector<int> threads = bench::thread_counts(maximum_threads);
    std::set<std::string> selected;
    bool reverse = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "activation_torch_benchmark [operation ...] [--sizes=1000,32769,1000000] "
                         "[--threads=1,2,6,12] [--reverse]\n"
                         "Operations: leaky_relu gelu gelu_tanh silu elu softplus\n"
                         "--reverse times ATen first; default times MensurFlow first.\n"
                         "Both libraries use preallocated F32 buffers; no Python is timed.\n";
            return 0;
        }
        if (arg == "--reverse") {
            reverse = true;
        } else if (arg.rfind("--sizes=", 0) == 0) {
            sizes = parse_list(arg.substr(8));
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads.clear();
            for (auto value : parse_list(arg.substr(10))) {
                if (value > maximum_threads) {
                    throw std::invalid_argument("requested threads exceed initial OpenMP maximum");
                }
                threads.push_back(static_cast<int>(value));
            }
        } else {
            bool known = false;
            for (const auto& op : operations) {
                known = known || arg == op.name;
            }
            if (!known) {
                throw std::invalid_argument("unknown operation or option: " + arg);
            }
            selected.insert(arg);
        }
    }
    // Three padded arrays: input, shared output and independent float64 reference.
    const auto max_stride = std::vector<float>().max_size() / 3;
    for (auto n : sizes) {
        if (max_stride <= 1279 || static_cast<std::uint64_t>(n) > max_stride - 1279) {
            throw std::invalid_argument("size exceeds the padded three-array allocation limit");
        }
    }

    c10::InferenceMode inference_guard;
    at::set_num_interop_threads(1);
    bench::Config config;
    config.slow_repeats = config.repeats; // Keep nine samples even for slow reference rows.
    bench::print_header(
        "Native C++ CPU activation comparison with PyTorch", config,
        {std::string("LibTorch version: ") + TORCH_VERSION,
         "scope: direct MensurFlow CPU calls vs preallocated C++ ATen out operator calls",
         "identical F32 input/output addresses; allocation, Tensor wrapping and Python excluded",
         "8 bytes/element; same requested intra-op threads; inter-op threads=1",
         "leaky_relu slope=0.01; ELU alpha=1; softplus beta=1 threshold=20",
         "GELU exact and tanh are separate; each side checked against float64 before timing",
         "vs torch = ATen median / MensurFlow median; above 1 means MensurFlow faster",
         std::string("measurement order: ") +
             (reverse ? "ATen then MensurFlow" : "MensurFlow then ATen"),
         "500 ms worker-settling pause before each library; repeat in a separate cooled session"});
    std::cout << "\nLibTorch build configuration:\n" << at::show_config()
              << "\nInitial LibTorch parallel configuration:\n" << at::get_parallel_info() << '\n';

    bench::Table table("GB/s", 1e9, "vs torch", "");
    for (const auto& op : operations) {
        if (!selected.empty() && selected.count(op.name) == 0) {
            continue;
        }
        table.section(op.name);
        for (auto n : sizes) {
            const auto len = static_cast<std::size_t>(n);
            bench::Arrays arrays(len, 3);
            float* input = arrays[0];
            float* output = arrays[1];
            float* expected = arrays[2];
            bench::fill_a(input, len);
            op.reference(input, expected, n);
            // from_blob borrows memory; arrays stay alive until both sides finish.
            const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            const at::Tensor x = at::from_blob(input, {n}, options);
            at::Tensor out = at::from_blob(output, {n}, options);
            auto ours = [&] { op.ours(input, output, n); };
            auto torch = [&] { op.torch(x, out); };
            auto observe = [&] { return bench::checksum(output, len); };
            for (int count : threads) {
                cpu::set_max_threads(count);
                at::set_num_threads(count);
                if (cpu::max_threads() != count || at::get_num_threads() != count) {
                    throw std::runtime_error("libraries did not accept the same thread count");
                }
                ours();
                bench::check_close(expected, output, len, 3e-6,
                                   std::string(op.name) + " MensurFlow vs float64");
                torch();
                bench::check_close(expected, output, len, 3e-6,
                                   std::string(op.name) + " ATen vs float64");
                if (out.data_ptr<float>() != output) {
                    throw std::runtime_error("ATen replaced the preallocated output buffer");
                }
                bench::Sample own_sample, torch_sample;
                auto measure_ours = [&] {
                    settle();
                    own_sample = bench::measure(config, ours, observe);
                };
                auto measure_torch = [&] {
                    settle();
                    torch_sample = bench::measure(config, torch, observe);
                };
                if (reverse) {
                    measure_torch();
                    measure_ours();
                } else {
                    measure_ours();
                    measure_torch();
                }
                const double bytes = 8.0 * static_cast<double>(n);
                table.row(std::to_string(n), "mensurflow", std::to_string(count), own_sample,
                          bytes, &torch_sample);
                table.row(std::to_string(n), "pytorch", std::to_string(count), torch_sample,
                          bytes, &torch_sample);
                std::cout.flush();
            }
            table.blank();
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "activation_torch_benchmark: " << e.what() << '\n';
        return 1;
    }
}
