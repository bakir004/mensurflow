// Auto-tunes a three-layer network's forward pass for the shapes it will run with, on the GPU if
// there is one, and stores the results in mensurflow_tuning.txt (in the current directory) for the
// next run. Run it twice: the second run reuses the stored results instead of measuring again.
//
//   tune_mlp [batch] [cpu]      (batch 256 by default; "cpu" tunes the CPU even with a GPU)
#include "mensurflow/mensurflow.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace mf = mensurflow;

namespace {

mf::Tensor random_matrix(std::int64_t rows, std::int64_t cols, std::uint32_t seed,
                         mf::Device device) {
    std::vector<float> values(static_cast<std::size_t>(rows * cols));
    std::uint32_t state = seed;
    for (float& v : values) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(static_cast<int>(state >> 9) % 2001 - 1000) / 1000.0f / 32.0f;
    }
    return mf::Tensor::from_vector(values, {rows, cols}, device);
}

} // namespace

int main(int argc, char** argv) {
    const std::int64_t batch = argc > 1 ? std::atoll(argv[1]) : 256;
    mf::Device device{mf::DeviceType::CPU, 0};
    const bool cpu_only = argc > 2 && std::string(argv[2]) == "cpu";
    for (const mf::Device d : mf::list_devices()) {
        if (d.type != mf::DeviceType::CPU && !cpu_only) {
            device = d;
            break;
        }
    }
    const mf::Execution on =
        device.type == mf::DeviceType::CPU ? mf::Execution::CPU : mf::Execution::GPU;

    // The algorithm, written once and run many times: 1024 -> 4096 -> 1024 -> 512 with additive
    // biases (given as full [batch, width] tensors, since bias_add runs only on the CPU so far).
    const mf::Algorithm network([on](const std::vector<mf::Tensor>& in) {
        mf::Tensor h = mf::add(mf::matmul(in[0], in[1], on), in[4], on);
        h = mf::add(mf::matmul(h, in[2], on), in[5], on);
        return std::vector<mf::Tensor>{mf::matmul(h, in[3], on)};
    });
    const std::vector<mf::Tensor> inputs = {
        random_matrix(batch, 1024, 1, device), random_matrix(1024, 4096, 2, device),
        random_matrix(4096, 1024, 3, device),  random_matrix(1024, 512, 4, device),
        random_matrix(batch, 4096, 5, device), random_matrix(batch, 1024, 6, device)};

    mf::Tuner tuner("mensurflow_tuning.txt");
    std::cout << "device " << mf::to_string(device) << ", batch " << batch << ", " << tuner.size()
              << " results already stored in " << tuner.path() << "\n\n";
    mf::TuneOptions options;
    options.verbose = true;
    const mf::TuneReport report = tuner.tune(network, inputs, options);

    std::printf("\n%-8s %-10s %-26s %10s %10s %8s  %s\n", "op", "device", "call", "default ms",
                "tuned ms", "speedup", "settings");
    for (const mf::TunedCall& call : report.calls) {
        std::printf("%-8s %-10s %-26s %10.4f %10.4f %7.2fx  %s%s\n", call.op.c_str(),
                    call.device.c_str(), call.key.c_str(), call.default_ms, call.best_ms,
                    call.default_ms / call.best_ms, mf::to_string(call.best_settings).c_str(),
                    call.reused ? "  (stored earlier)" : "");
    }
    std::printf("\nwhole network: %.4f ms with the defaults, %.4f ms tuned (%.2fx)%s\n",
                report.default_ms, report.tuned_ms, report.default_ms / report.tuned_ms,
                report.kept ? "" : "; slower, so the new results were dropped");

    // Using the results: run() applies them to every call they cover.
    const auto out = network.run(inputs, tuner)[0];
    std::cout << "output " << mf::to_string(out.shape()) << ", saved to " << tuner.path() << '\n';
    return 0;
}
