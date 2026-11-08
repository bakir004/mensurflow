#include "bench.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace cpu = mensurflow::cpu;
using mensurflow::Tensor;

std::vector<float> fill(std::size_t n, int period, float offset) {
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<float>(i % static_cast<std::size_t>(period)) / 8.0f - offset;
    }
    return values;
}

double observe(const Tensor& tensor) {
    const std::vector<float> values = tensor.to_vector<float>();
    double total = 0.0;
    for (float value : values) {
        total += value;
    }
    return total;
}

void check_result(const std::vector<float>& expected, const Tensor& actual, const std::string& name) {
    const std::vector<float> values = actual.to_vector<float>();
    if (values.size() != expected.size()) {
        throw std::runtime_error(name + ": output size mismatch");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 2e-5f) {
            throw std::runtime_error(name + ": output mismatch at " + std::to_string(i));
        }
    }
}

template <class Run>
void run_operation(bench::Table& table, const bench::Config& config, const std::string& name,
                   const std::vector<int>& thread_counts, double bytes, Run&& run) {
    Tensor result;
    auto execute = [&] { result = run(); };
    auto observe_result = [&] { return observe(result); };

    cpu::set_max_threads(1);
    execute();
    const std::vector<float> expected = result.to_vector<float>();
    const bench::Sample sequential = bench::measure(config, execute, observe_result);
    table.row(name, "mensurflow", "1", sequential, bytes);
    for (int threads : thread_counts) {
        if (threads == 1) {
            continue;
        }
        cpu::set_max_threads(threads);
        execute();
        check_result(expected, result, name + " threads=" + std::to_string(threads));
        const bench::Sample parallel = bench::measure(config, execute, observe_result);
        table.row(name, "mensurflow", std::to_string(threads), parallel, bytes);
    }
}

} // namespace

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header(
        "MensurFlow activation backward and tensor movement, float32",
        config,
        {"public CPU APIs; allocation and result readback are outside timing checks",
         "same shapes as benchmarks/reference/activation_movement.py",
         "rate is GB/s: two input reads and one output write for backward; one read and one write for movement"});
    bench::Table table("GB/s", 1e9);

    const std::int64_t n = 1 << 20;
    const auto values = fill(static_cast<std::size_t>(n), 17, 1.0f);
    const auto gradient_values = fill(static_cast<std::size_t>(n), 13, 0.5f);
    const Tensor x = Tensor::from_vector(values, {n});
    const Tensor gradient = Tensor::from_vector(gradient_values, {n});
    table.section("activation backward");
    run_operation(table, config, "relu_backward", thread_counts, 12.0 * n,
                  [&] { return mensurflow::relu_backward(x, gradient); });
    run_operation(table, config, "sigmoid_backward", thread_counts, 12.0 * n,
                  [&] { return mensurflow::sigmoid_backward(x, gradient); });
    run_operation(table, config, "tanh_backward", thread_counts, 12.0 * n,
                  [&] { return mensurflow::tanh_backward(x, gradient); });
    run_operation(table, config, "gelu_backward", thread_counts, 12.0 * n,
                  [&] { return mensurflow::gelu_backward(x, gradient); });
    run_operation(table, config, "silu_backward", thread_counts, 12.0 * n,
                  [&] { return mensurflow::silu_backward(x, gradient); });

    const auto matrix_values = fill(2048 * 2048, 17, 1.0f);
    const Tensor matrix = Tensor::from_vector(matrix_values, {2048, 2048});
    table.section("contiguous and permute");
    run_operation(table, config, "transpose_contiguous", thread_counts,
                  2.0 * matrix_values.size() * sizeof(float),
                  [&] { return matrix.transpose(0, 1).contiguous(); });
    const auto cube_values = fill(64 * 128 * 256, 17, 1.0f);
    const Tensor cube = Tensor::from_vector(cube_values, {64, 128, 256});
    run_operation(table, config, "permute_contiguous", thread_counts,
                  2.0 * cube_values.size() * sizeof(float),
                  [&] { return cube.permute({2, 0, 1}).contiguous(); });

    const auto selection_values = fill(1024 * 1024, 17, 1.0f);
    const Tensor selection_input = Tensor::from_vector(selection_values, {1024, 1024});
    std::vector<std::int64_t> columns(1024);
    for (std::int64_t i = 0; i < 1024; ++i) {
        columns[static_cast<std::size_t>(i)] = 1023 - i;
    }
    const Tensor column_indices = Tensor::from_vector(columns, {1024});
    std::vector<std::int64_t> gather_values(1024 * 1024);
    for (std::size_t i = 0; i < gather_values.size(); ++i) {
        gather_values[i] = columns[i % columns.size()];
    }
    const Tensor gather_indices = Tensor::from_vector(gather_values, {1024, 1024});
    table.section("gather and index_select");
    run_operation(table, config, "index_select", thread_counts,
                  2.0 * selection_values.size() * sizeof(float),
                  [&] { return mensurflow::index_select(selection_input, column_indices, 1); });
    run_operation(table, config, "gather", thread_counts,
                  2.0 * selection_values.size() * sizeof(float),
                  [&] { return mensurflow::gather(selection_input, gather_indices, 1); });

    cpu::set_max_threads(maximum_threads);
    return 0;
}
