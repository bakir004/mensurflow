#include "bench.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#include <vector>

namespace cpu = mensurflow::cpu;
using mensurflow::Tensor;

int main() {
    const bench::Config config;
    bench::print_header("MensurFlow optimizer and normalization kernels, float32", config,
                        {"CPU public APIs; median/min over the standard harness"});
    bench::Table table("GB/s", 1e9);
    const std::int64_t n = 1 << 20;
    std::vector<float> values(static_cast<std::size_t>(n), 1.0f);
    const Tensor x = Tensor::from_vector(values, {n / 1024, 1024});
    const Tensor g = Tensor::ones(x.shape());
    Tensor w = Tensor::ones({1024}), m = Tensor::zeros(x.shape()), v = Tensor::zeros(x.shape());
    table.section("normalization");
    for (int threads : bench::thread_counts(cpu::max_threads())) {
        cpu::set_max_threads(threads);
        Tensor layer_out;
        auto layer = bench::measure(config, [&] { layer_out = mensurflow::layer_norm(x, {1024}, &w); },
                                    [&] { return layer_out.to_vector<float>()[0]; });
        table.row("layer_norm", "mensurflow", std::to_string(threads), layer,
                  3.0 * n * sizeof(float));
        Tensor rms_out;
        auto rms = bench::measure(config, [&] { rms_out = mensurflow::rms_norm(x, {1024}, &w); },
                                  [&] { return rms_out.to_vector<float>()[0]; });
        table.row("rms_norm", "mensurflow", std::to_string(threads), rms,
                  3.0 * n * sizeof(float));
    }
    table.section("optimizers");
    for (int threads : bench::thread_counts(cpu::max_threads())) {
        cpu::set_max_threads(threads);
        Tensor p;
        auto adam = bench::measure(config, [&] {
            std::vector<float> reset(static_cast<std::size_t>(n), 1.0f);
            m = Tensor::zeros(x.shape()); v = Tensor::zeros(x.shape());
            p = Tensor::from_vector(reset, x.shape());
            mensurflow::adam(p, g, m, v, 1e-3f, 1);
        }, [&] { return p.to_vector<float>()[0]; });
        table.row("adam", "mensurflow", std::to_string(threads), adam,
                  5.0 * n * sizeof(float));
    }
    return 0;
}
