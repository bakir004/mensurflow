// Public CPU APIs, optionally compared with native allocating ATen operations.
// Allocation and validation are timed on both sides; setup, checks and readback are not.
#include "bench.hpp"

#include "backends/cpu/threads.hpp"
#include "mensurflow/ops/indexing.hpp"
#include "mensurflow/ops/random.hpp"

#ifdef MENSURFLOW_TORCH_REFERENCE
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/native_dropout_backward.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;
using mensurflow::Execution;
using mensurflow::Tensor;

namespace {

constexpr std::int64_t classes = 16;
constexpr std::int64_t vocabulary = 512;
constexpr std::int64_t width = 32;
constexpr std::uint64_t seed = 20261111;
constexpr float probability = 0.25f;
constexpr float scale = 1.0f / (1.0f - probability);

enum class Kind { Uniform, Normal, Bernoulli, Dropout, DropoutBackward, OneHot, Embedding,
                  EmbeddingBackward };

struct Operation {
    const char* name;
    Kind kind;
};

constexpr Operation operations[] = {
    {"uniform", Kind::Uniform}, {"normal", Kind::Normal},
    {"bernoulli", Kind::Bernoulli}, {"dropout", Kind::Dropout},
    {"dropout_backward", Kind::DropoutBackward}, {"one_hot", Kind::OneHot},
    {"embedding", Kind::Embedding}, {"embedding_backward", Kind::EmbeddingBackward}};

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t parsed = 0;
        const auto value = std::stoll(token, &parsed);
        if (value <= 0 || parsed != token.size() ||
            std::find(values.begin(), values.end(), value) != values.end()) {
            throw std::invalid_argument("expected distinct positive integers");
        }
        values.push_back(value);
    }
    if (values.empty() || text.back() == ',') {
        throw std::invalid_argument("expected a nonempty comma-separated list");
    }
    return values;
}

void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

struct Data {
    std::int64_t n;
    std::int64_t label_count;
    std::int64_t lookup_count;
    std::vector<float> values;
    std::vector<float> weights;
    std::vector<float> gradients;
    std::vector<std::int64_t> labels;
    std::vector<std::int64_t> indices;
    std::vector<std::uint8_t> mask;
    Tensor x;
    Tensor weight;
    Tensor gradient;
    Tensor label;
    Tensor index;
    Tensor saved_mask;

#ifdef MENSURFLOW_TORCH_REFERENCE
    at::Tensor torch_x;
    at::Tensor torch_weight;
    at::Tensor torch_gradient;
    at::Tensor torch_label;
    at::Tensor torch_index;
    at::Tensor torch_mask;
#endif

    explicit Data(std::int64_t count)
        : n(count), label_count(1 + (n - 1) / classes),
          lookup_count(1 + (n - 1) / width), values(static_cast<std::size_t>(n)),
          weights(static_cast<std::size_t>(vocabulary * width)),
          gradients(static_cast<std::size_t>(lookup_count * width)),
          labels(static_cast<std::size_t>(label_count)),
          indices(static_cast<std::size_t>(lookup_count)), mask(static_cast<std::size_t>(n)) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 4.0f;
            mask[i] = i % 4 != 0;
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<float>(static_cast<int>(i % 37) - 18) / 4.0f;
        }
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            gradients[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 4.0f;
        }
        for (std::size_t i = 0; i < labels.size(); ++i) {
            labels[i] = static_cast<std::int64_t>((7 * i + 3) % classes);
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            indices[i] = static_cast<std::int64_t>((17 * i + i / 7) % vocabulary);
        }
        x = Tensor::from_vector(values, {n});
        weight = Tensor::from_vector(weights, {vocabulary, width});
        gradient = Tensor::from_vector(gradients, {lookup_count, width});
        label = Tensor::from_vector(labels, {label_count});
        index = Tensor::from_vector(indices, {lookup_count});
        saved_mask = Tensor::from_host(mask.data(), {n}, mensurflow::DType::Bool);
#ifdef MENSURFLOW_TORCH_REFERENCE
        const auto fp = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const auto integer = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
        torch_x = at::from_blob(values.data(), {n}, fp);
        torch_weight = at::from_blob(weights.data(), {vocabulary, width}, fp);
        torch_gradient = at::from_blob(gradients.data(), {lookup_count, width}, fp);
        torch_label = at::from_blob(labels.data(), {label_count}, integer);
        torch_index = at::from_blob(indices.data(), {lookup_count}, integer);
        torch_mask = at::from_blob(mask.data(), {n}, at::TensorOptions().dtype(at::kBool));
#endif
    }

    std::string shape(Kind kind) const {
        if (kind == Kind::OneHot) {
            return std::to_string(label_count) + "x" + std::to_string(classes);
        }
        if (kind == Kind::Embedding || kind == Kind::EmbeddingBackward) {
            return std::to_string(lookup_count) + "x" + std::to_string(width);
        }
        return std::to_string(n);
    }

    double elements(Kind kind) const {
        if (kind == Kind::OneHot) {
            return static_cast<double>(label_count * classes);
        }
        if (kind == Kind::Embedding || kind == Kind::EmbeddingBackward) {
            return static_cast<double>(lookup_count * width);
        }
        return static_cast<double>(n);
    }

    mensurflow::Shape output_shape(Kind kind) const {
        if (kind == Kind::OneHot) {
            return {label_count, classes};
        }
        if (kind == Kind::Embedding) {
            return {lookup_count, width};
        }
        if (kind == Kind::EmbeddingBackward) {
            return {vocabulary, width};
        }
        return {n};
    }
};

std::vector<float> deterministic_reference(Kind kind, const Data& data) {
    if (kind == Kind::DropoutBackward) {
        std::vector<float> expected(data.values.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expected[i] = data.mask[i] ? data.values[i] * scale : 0.0f;
        }
        return expected;
    }
    if (kind == Kind::Embedding) {
        std::vector<float> expected(data.gradients.size());
        for (std::size_t i = 0; i < data.indices.size(); ++i) {
            for (std::int64_t column = 0; column < width; ++column) {
                expected[i * width + column] = data.weights[data.indices[i] * width + column];
            }
        }
        return expected;
    }
    if (kind == Kind::EmbeddingBackward) {
        std::vector<float> expected(data.weights.size(), 0.0f);
        for (std::size_t i = 0; i < data.indices.size(); ++i) {
            for (std::int64_t column = 0; column < width; ++column) {
                expected[data.indices[i] * width + column] += data.gradients[i * width + column];
            }
        }
        return expected;
    }
    return {};
}

void check_random(const std::vector<float>& values, Kind kind, const std::string& what) {
    double sum = 0.0;
    double squares = 0.0;
    for (float value : values) {
        if (!std::isfinite(value) ||
            (kind == Kind::Uniform && (value < -1.0f || value >= 1.0f)) ||
            (kind == Kind::Bernoulli && value != 0.0f && value != 1.0f)) {
            throw std::runtime_error(what + ": invalid random sample");
        }
        sum += value;
        squares += static_cast<double>(value) * value;
    }
    const double count = static_cast<double>(values.size());
    const double expected_mean = kind == Kind::Bernoulli ? probability : 0.0;
    const double expected_second = kind == Kind::Uniform ? 1.0 / 3.0
                                   : kind == Kind::Normal ? 1.0 : probability;
    // Loose ten-standard-error sanity checks, not a statistical certification of either RNG.
    const double tolerance = 10.0 / std::sqrt(count);
    if (std::abs(sum / count - expected_mean) > tolerance ||
        std::abs(squares / count - expected_second) > 3.0 * tolerance) {
        throw std::runtime_error(what + ": unexpected sample moments");
    }
}

void check_float_result(const std::vector<float>& actual, const std::vector<std::uint8_t>& mask,
                        Kind kind, const Data& data, const std::string& what) {
    if (kind == Kind::Uniform || kind == Kind::Normal || kind == Kind::Bernoulli) {
        check_random(actual, kind, what);
    } else if (kind == Kind::Dropout) {
        if (mask.size() != actual.size()) {
            throw std::runtime_error(what + ": dropout mask shape");
        }
        std::size_t kept = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float expected = mask[i] ? data.values[i] * scale : 0.0f;
            if (mask[i] > 1 || actual[i] != expected) {
                throw std::runtime_error(what + ": dropout formula mismatch");
            }
            kept += mask[i];
        }
        const double count = static_cast<double>(actual.size());
        const double standard_error = std::sqrt(probability * (1.0 - probability) / count);
        if (std::abs(static_cast<double>(kept) / count - (1.0 - probability)) >
            10.0 * standard_error) {
            throw std::runtime_error(what + ": unexpected dropout keep rate");
        }
    } else {
        const auto expected = deterministic_reference(kind, data);
        if (actual.size() != expected.size()) {
            throw std::runtime_error(what + ": output shape mismatch");
        }
        bench::check_close(expected.data(), actual.data(), actual.size(), 2e-6, what);
    }
}

void check_one_hot(const std::int64_t* values, const Data& data, const std::string& what) {
    for (std::int64_t row = 0; row < data.label_count; ++row) {
        for (std::int64_t column = 0; column < classes; ++column) {
            if (values[row * classes + column] != (column == data.labels[row] ? 1 : 0)) {
                throw std::runtime_error(what + ": one_hot mismatch");
            }
        }
    }
}

Tensor execute(Kind kind, const Data& data, Tensor& mask) {
    using namespace mensurflow;
    switch (kind) {
    case Kind::Uniform:
        return uniform({data.n}, -1.0f, 1.0f, seed, 0, Execution::CPU);
    case Kind::Normal:
        return normal({data.n}, 0.0f, 1.0f, seed, 0, Execution::CPU);
    case Kind::Bernoulli:
        return bernoulli({data.n}, probability, seed, 0, Execution::CPU);
    case Kind::Dropout: {
        auto result = dropout(data.x, probability, true, seed, 0, Execution::CPU);
        mask = result.mask;
        return result.output;
    }
    case Kind::DropoutBackward:
        return dropout_backward(data.x, data.saved_mask, probability, true, Execution::CPU);
    case Kind::OneHot:
        return one_hot(data.label, classes, Execution::CPU);
    case Kind::Embedding:
        return embedding(data.weight, data.index, Execution::CPU);
    case Kind::EmbeddingBackward:
        return embedding_backward(data.gradient, data.index, vocabulary, Execution::CPU);
    }
    throw std::logic_error("invalid benchmark operation");
}

double observe(const Tensor& output) {
    if (output.dtype() == mensurflow::DType::I64) {
        const auto values = output.to_vector<std::int64_t>();
        double total = 0.0;
        for (auto value : values) {
            total += static_cast<double>(value);
        }
        return total;
    }
    const auto values = output.to_vector<float>();
    return bench::checksum(values.data(), values.size());
}

void check(const Tensor& output, const Tensor& saved_mask, Kind kind, const Data& data) {
    const auto expected_dtype = kind == Kind::OneHot ? mensurflow::DType::I64
                                                    : mensurflow::DType::F32;
    if (!output.defined() || output.shape() != data.output_shape(kind) ||
        output.dtype() != expected_dtype) {
        throw std::runtime_error("MensurFlow: output shape or dtype mismatch");
    }
    if (kind == Kind::OneHot) {
        const auto values = output.to_vector<std::int64_t>();
        check_one_hot(values.data(), data, "MensurFlow");
        return;
    }
    std::vector<std::uint8_t> mask;
    if (kind == Kind::Dropout) {
        if (!saved_mask.defined() || saved_mask.shape() != mensurflow::Shape{data.n} ||
            saved_mask.dtype() != mensurflow::DType::Bool) {
            throw std::runtime_error("MensurFlow: dropout mask shape or dtype mismatch");
        }
        mask.resize(static_cast<std::size_t>(saved_mask.numel()));
        saved_mask.copy_to_host(mask.data());
    }
    check_float_result(output.to_vector<float>(), mask, kind, data, "MensurFlow");
}

#ifdef MENSURFLOW_TORCH_REFERENCE
at::Tensor execute_torch(Kind kind, const Data& data, at::Tensor& mask, at::Generator& generator) {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    switch (kind) {
    case Kind::Uniform:
        return at::empty({data.n}, options).uniform_(-1.0, 1.0, generator);
    case Kind::Normal:
        return at::empty({data.n}, options).normal_(0.0, 1.0, generator);
    case Kind::Bernoulli:
        return at::empty({data.n}, options).bernoulli_(probability, generator);
    case Kind::Dropout: {
        auto result = at::native_dropout(data.torch_x, probability, true);
        mask = std::get<1>(result);
        return std::get<0>(result);
    }
    case Kind::DropoutBackward:
        return at::native_dropout_backward(data.torch_x, data.torch_mask, scale);
    case Kind::OneHot:
        return at::one_hot(data.torch_label, classes);
    case Kind::Embedding:
        return at::embedding(data.torch_weight, data.torch_index, -1, false, false);
    case Kind::EmbeddingBackward:
        return at::embedding_dense_backward(data.torch_gradient, data.torch_index, vocabulary,
                                            -1, false);
    }
    throw std::logic_error("invalid benchmark operation");
}

double observe_torch(const at::Tensor& output) {
    if (output.scalar_type() == at::kLong) {
        const auto* values = output.data_ptr<std::int64_t>();
        double total = 0.0;
        for (std::int64_t i = 0; i < output.numel(); ++i) {
            total += static_cast<double>(values[i]);
        }
        return total;
    }
    return bench::checksum(output.data_ptr<float>(), static_cast<std::size_t>(output.numel()));
}

void check_torch(const at::Tensor& output, const at::Tensor& saved_mask, Kind kind,
                 const Data& data) {
    const auto shape = data.output_shape(kind);
    const auto expected_dtype = kind == Kind::OneHot ? at::kLong : at::kFloat;
    if (!output.defined() || output.sizes() != at::IntArrayRef(shape) ||
        output.scalar_type() != expected_dtype || !output.is_contiguous()) {
        throw std::runtime_error("PyTorch: output shape, dtype or layout mismatch");
    }
    if (kind == Kind::OneHot) {
        check_one_hot(output.data_ptr<std::int64_t>(), data, "PyTorch");
        return;
    }
    const auto* values = output.data_ptr<float>();
    std::vector<std::uint8_t> mask;
    if (kind == Kind::Dropout) {
        if (!saved_mask.defined() || saved_mask.dim() != 1 || saved_mask.size(0) != data.n ||
            saved_mask.scalar_type() != at::kBool || !saved_mask.is_contiguous()) {
            throw std::runtime_error("PyTorch: dropout mask shape, dtype or layout mismatch");
        }
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(saved_mask.data_ptr<bool>());
        mask.assign(bytes, bytes + saved_mask.numel());
    }
    check_float_result(std::vector<float>(values, values + output.numel()), mask, kind, data,
                       "PyTorch");
}
#endif

int run(int argc, char** argv) {
    const int maximum_threads = cpu::max_threads();
    std::vector<std::int64_t> sizes{1000, 32769, 8388608};
    std::vector<int> threads = bench::thread_counts(maximum_threads);
    std::set<std::string> selected;
    bool reverse = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "random_indexing_benchmark [operation ...] [--sizes=1000,32769,8388608] "
                         "[--threads=1,2,6,12] [--reverse]\n"
                         "Operations: uniform normal bernoulli dropout dropout_backward "
                         "one_hot embedding embedding_backward\n"
                         "Sizes are output-element targets; one_hot rounds up to 16, embedding "
                         "to 32. Embedding backward prints its upstream-gradient shape.\n"
                         "--reverse times PyTorch first when its optional reference is enabled.\n";
            return 0;
        }
        if (arg == "--reverse") {
            reverse = true;
        } else if (arg.rfind("--sizes=", 0) == 0) {
            sizes = parse_list(arg.substr(8));
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads.clear();
            for (auto count : parse_list(arg.substr(10))) {
                if (count > maximum_threads) {
                    throw std::invalid_argument("requested threads exceed initial OpenMP maximum");
                }
                threads.push_back(static_cast<int>(count));
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
    for (auto n : sizes) {
        if (static_cast<std::uint64_t>(n) >
            std::min(std::vector<float>().max_size(), std::vector<std::int64_t>().max_size()) -
                static_cast<std::size_t>(width)) {
            throw std::invalid_argument("size exceeds benchmark allocation limit");
        }
    }

    bench::Config config;
    config.slow_repeats = config.repeats;
    std::vector<std::string> notes{
        "public CPU APIs; allocation and validation timed; setup/checks/readback excluded",
        "uniform [-1,1), normal mean=0 std=1, Bernoulli p=0.25; dropout p=0.25 training=true",
        "RNG calls include generation, not seeding; MensurFlow seed=20261111 offset=0 each call",
        "same-size F32 inputs, I64 indices, Bool dropout mask, I64 one_hot outputs",
        "one_hot classes=16; embedding vocabulary=512 width=32; repeated indices accumulate",
        "embedding_backward shape is upstream gradient; output is always 512x32",
        "Melems/s counts generated/output elements; embedding_backward counts upstream elements",
        "all deterministic outputs checked against independent formulas before timing",
        "500 ms settling pause before each measurement; nine samples even for slow calls"};
#ifdef MENSURFLOW_TORCH_REFERENCE
    c10::InferenceMode guard;
    at::set_num_interop_threads(1);
    at::manual_seed(seed);
    auto generator = at::make_generator<at::CPUGeneratorImpl>(seed);
    notes.push_back(std::string("LibTorch version: ") + TORCH_VERSION);
    notes.push_back("RNG streams differ (Philox vs PyTorch CPU); bounds/moments checked independently");
    notes.push_back("dropout forward includes saved mask on both sides; masks checked independently");
    notes.push_back("dropout backward uses one shared predetermined mask on both sides");
    notes.push_back("vs torch = PyTorch median / MensurFlow median; above 1 favors MensurFlow");
    notes.push_back(reverse ? "measurement order: PyTorch then MensurFlow"
                            : "measurement order: MensurFlow then PyTorch");
#else
    notes.push_back("LibTorch reference disabled; vs seq compares the same public API on one thread");
    (void)reverse;
#endif
    bench::print_header("MensurFlow random and indexing operations (Aldin)", config, notes);
#ifdef MENSURFLOW_TORCH_REFERENCE
    std::cout << "\nLibTorch build configuration:\n" << at::show_config()
              << "\nInitial LibTorch parallel configuration:\n" << at::get_parallel_info() << '\n';
    bench::Table table("Melems/s", 1e6, "vs torch", "vs seq");
#else
    bench::Table table("Melems/s", 1e6, "vs seq", "");
#endif

    for (const auto& op : operations) {
        if (!selected.empty() && selected.count(op.name) == 0) {
            continue;
        }
        table.section(op.name);
        for (auto n : sizes) {
            Data data(n);
            Tensor output, mask;
            auto ours = [&] { output = execute(op.kind, data, mask); };
            auto observe_ours = [&] { return observe(output); };
            cpu::set_max_threads(1);
            ours();
            check(output, mask, op.kind, data);
            settle();
            const auto sequential = bench::measure(config, ours, observe_ours);
#ifdef MENSURFLOW_TORCH_REFERENCE
            at::Tensor torch_output, torch_mask;
            auto torch = [&] { torch_output = execute_torch(op.kind, data, torch_mask, generator); };
            auto observe_reference = [&] { return observe_torch(torch_output); };
#endif
            for (int count : threads) {
                cpu::set_max_threads(count);
                ours();
                check(output, mask, op.kind, data);
#ifdef MENSURFLOW_TORCH_REFERENCE
                at::set_num_threads(count);
                torch();
                check_torch(torch_output, torch_mask, op.kind, data);
                bench::Sample own_sample, torch_sample;
                auto measure_ours = [&] {
                    settle();
                    own_sample = bench::measure(config, ours, observe_ours);
                };
                auto measure_reference = [&] {
                    settle();
                    torch_sample = bench::measure(config, torch, observe_reference);
                };
                if (reverse) {
                    measure_reference();
                    measure_ours();
                } else {
                    measure_ours();
                    measure_reference();
                }
                table.row(data.shape(op.kind), "mensurflow", std::to_string(count), own_sample,
                          data.elements(op.kind), &torch_sample, &sequential);
                table.row(data.shape(op.kind), "pytorch", std::to_string(count), torch_sample,
                          data.elements(op.kind), &torch_sample, &sequential);
#else
                settle();
                const auto sample = bench::measure(config, ours, observe_ours);
                table.row(data.shape(op.kind), "mensurflow", std::to_string(count), sample,
                          data.elements(op.kind), &sequential);
#endif
                std::cout.flush();
            }
            table.blank();
        }
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "random_indexing_benchmark: " << error.what() << '\n';
        return 1;
    }
}
