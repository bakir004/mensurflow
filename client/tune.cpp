// Auto-tuning from a user's point of view, with only the public API: write the computation once as
// an Algorithm, tune it for the inputs it will run with, then run it with the stored settings.
//
//   client_tune [results file]      (client_tuning.txt by default)
//
// Run it twice: the first run measures and writes the file; the second finds every call already
// stored and measures nothing. Delete the file to tune again from scratch.
#include "mensurflow/mensurflow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace mensurflow;
using Clock = std::chrono::steady_clock;

namespace {

// Median milliseconds of `run` over a few calls; reading one element waits for a GPU to finish.
template <class Run> double time_ms(Run&& run) {
    std::vector<double> samples;
    for (int i = 0; i < 7; ++i) {
        const auto t0 = Clock::now();
        const Tensor out = run();
        (void)out.view({out.numel()}).slice(0, 0, 1).to_vector<float>();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "client_tuning.txt";

    // 1. What can be tuned in this build, on this machine.
    std::cout << "tunable kernels:\n";
    for (const TunableKernelInfo& k : tunable_kernels()) {
        std::cout << "  " << k.name << " on " << k.device << ", settings:";
        for (const std::string& p : k.params) {
            std::cout << ' ' << p;
        }
        std::cout << '\n';
    }

    // 2. The computation, written once: a two-layer network, 512 -> 1024 -> 256, batch 256. Every op
    //    keeps the default Execution::Auto, so tuning also chooses the device of each one (without
    //    a stored choice, Auto runs on the CPU).
    const Algorithm network([](const std::vector<Tensor>& in) {
        const Tensor hidden = relu(matmul(in[0], in[1]));
        return std::vector<Tensor>{matmul(hidden, in[2])};
    });
    const std::vector<Tensor> inputs = {uniform({256, 512}, -1, 1, 1),
                                        uniform({512, 1024}, -0.05f, 0.05f, 2),
                                        uniform({1024, 256}, -0.05f, 0.05f, 3)};

    // 3. The tuner reads the results stored earlier (a missing file is fine).
    Tuner tuner(path);
    std::cout << '\n' << tuner.size() << " results already stored in " << tuner.path() << "\n\n";

    // 4. Tune for exactly these inputs. Every option has a default; these are the common ones.
    TuneOptions options;
    options.search = SearchMethod::BinarySampling; // or Exhaustive: every point, much slower
    options.give_up = 1.3;  // stop timing a point once it is 1.3x the best so far (0: never)
    options.verbose = true; // a line per call as it is found
    const TuneReport report = tuner.tune(network, inputs, options);

    // 5. What it found, one entry per distinct call. A kernel call has the kernel's settings; the
    //    device of an Execution::Auto call (device "auto") is one setting, gpu=0 or gpu=1, timed
    //    with the copies to the device and back.
    std::cout << "\nresults:\n";
    for (const TunedCall& call : report.calls) {
        std::printf("  %s on %s, %s%s\n", call.op.c_str(), call.device.c_str(), call.key.c_str(),
                    call.reused ? "  (stored earlier)" : "");
        std::printf("      default %-34s %8.3f ms\n", to_string(call.default_settings).c_str(),
                    call.default_ms);
        std::printf("      tuned   %-34s %8.3f ms  (%.2fx", to_string(call.best_settings).c_str(),
                    call.best_ms, call.default_ms / call.best_ms);
        if (!call.reused) {
            std::printf(", %zu of %zu points measured", call.points.size(), call.space_size);
        }
        std::printf(")\n");
    }
    std::printf("\nwhole network: %.3f ms with the defaults, %.3f ms tuned%s\n", report.default_ms,
                report.tuned_ms, report.kept ? "" : " (slower, so the new results were dropped)");

    // 6. Using the results: pass the tuner to run(). Calls it has results for get them; anything
    //    else (other sizes, other ops) runs with the defaults. run() without a tuner ignores them.
    const double plain = time_ms([&] { return network.run(inputs)[0]; });
    const double tuned = time_ms([&] { return network.run(inputs, tuner)[0]; });
    std::printf("measured here: %.3f ms -> %.3f ms (%.2fx)\n", plain, tuned, plain / tuned);

    // Same inputs, same result, up to float rounding where another device or kernel adds the
    // products in another order (the tuner rejects any point that differs by more).
    const std::vector<float> a = network.run(inputs)[0].to_vector<float>();
    const std::vector<float> b = network.run(inputs, tuner)[0].to_vector<float>();
    float largest = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        largest = std::max(largest, std::abs(a[i] - b[i]));
    }
    std::printf("largest difference from the untuned result: %g\n", largest);

    // 7. The file is plain text, one line per result under this machine's name; tune() saved it.
    std::cout << "\n" << tuner.path() << " (this machine):\n" << tuner.describe();
    return 0;
}
