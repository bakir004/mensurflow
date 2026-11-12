// Runner behind the observability dashboard. It uses only the public mensurflow API (plus the trace
// hook) and prints JSON on stdout; observability/serve.py calls it for each request.
//
//   observe list
//       devices, whether tracing is compiled in, and every op with its size unit, default sizes and
//       per-device support (each op is tried once at a tiny size on each device).
//   observe run --op NAME --device cpu:0 [--execution auto|cpu|gpu] [--sizes 1024,65536]
//               [--min-time 0.25]
//       times the op at each size: a warm-up call, then calls until min-time has passed (at least 5).
//       Every timed call reads one element of the result, which waits for a GPU to finish. One more
//       call runs with the trace hook installed, so the result shows which kernels it dispatched.
//   observe call --op NAME --device cpu:0 [--execution ...] [--size N] [--calls 25]
//       traces single calls: each call gets a "call" span (from entering the op to having its result),
//       split into "<op>()" (the function returning) and "read result" (waiting for the device), with
//       the library's spans inside: the dispatch, the kernel, allocations and copies. Also times the
//       same calls with tracing detached, so the tracing overhead is visible.
//   observe program --device cpu:0 [--execution ...] [--iterations 3]
//       runs a small MLP step (forward and backward) and returns its full trace.
//
// The thread count comes from OMP_NUM_THREADS, which serve.py sets per request.
#include "mensurflow/mensurflow.hpp"
#include "trace_collector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace mensurflow;
using Clock = std::chrono::steady_clock;

namespace {

// ---- inputs ---------------------------------------------------------------------------------------

std::vector<float> values(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    for (float& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = 0.1f +
            static_cast<float>(seed >> 8) / 16777216.0f; // [0.1, 1.1): safe for log, sqrt, div
    }
    return v;
}

Tensor rnd(Shape shape, std::uint32_t seed, Device device = {}) {
    return Tensor::from_vector(values(numel(shape), seed), shape, device);
}

Tensor indices(Shape shape, std::int64_t limit, std::uint32_t seed, Device device) {
    std::vector<std::int64_t> v(static_cast<std::size_t>(numel(shape)));
    for (auto& x : v) {
        seed = seed * 1664525u + 1013904223u;
        x = static_cast<std::int64_t>(seed >> 8) % limit;
    }
    return Tensor::from_vector(v, shape, device);
}

// A [rows, 256] matrix with about n elements.
Shape rows_of_256(std::int64_t n) { return {std::max<std::int64_t>(1, n / 256), 256}; }

// ---- op table -------------------------------------------------------------------------------------

struct Case {
    std::vector<Tensor> inputs;
    std::function<Tensor()> run;
    double flops = 0; // 0: memory-bound, rate is GB/s
};

using Make = std::function<Case(std::int64_t size, Device device, Execution execution)>;

struct OpSpec {
    std::string name;
    std::string group;
    std::string unit; // what `size` means
    std::vector<std::int64_t> sizes;
    Make make;
};

const std::vector<std::int64_t> element_sizes = {1 << 10, 1 << 14, 1 << 18,
                                                 1 << 20, 1 << 22, 1 << 24};
const std::vector<std::int64_t> matrix_sizes = {32, 64, 128, 256, 512, 1024};
constexpr const char* elements = "elements";
constexpr const char* side = "matrix side N";

using Unary = Tensor (*)(const Tensor&, Execution);
using Binary = Tensor (*)(const Tensor&, const Tensor&, Execution);

OpSpec unary(const char* name, const char* group, Unary f) {
    return {name, group, elements, element_sizes, [f](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd({n}, 1, d);
                return Case{{x}, [=] { return f(x, e); }};
            }};
}

OpSpec binary(const char* name, const char* group, Binary f) {
    return {name, group, elements, element_sizes, [f](std::int64_t n, Device d, Execution e) {
                Tensor a = rnd({n}, 1, d);
                Tensor b = rnd({n}, 2, d);
                return Case{{a, b}, [=] { return f(a, b, e); }};
            }};
}

using AxisReduce = Tensor (*)(const Tensor&, int, bool, Execution);

OpSpec axis_reduce(const char* name, AxisReduce f) {
    return {name, "reduction", elements, element_sizes, [f](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd(rows_of_256(n), 1, d);
                return Case{{x}, [=] { return f(x, 1, false, e); }};
            }};
}

std::vector<OpSpec> make_ops() {
    std::vector<OpSpec> ops;
    const auto add_op = [&](OpSpec spec) { ops.push_back(std::move(spec)); };

    // BLAS
    add_op({"dot", "blas", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
                Tensor a = rnd({n}, 1, d), b = rnd({n}, 2, d);
                return Case{{a, b}, [=] { return dot(a, b, e); }, 2.0 * n};
            }});
    add_op({"scal", "blas", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd({n}, 1, d);
                return Case{{x}, [=] { return scal(1.5f, x, e); }};
            }});
    add_op(unary("asum", "blas", [](const Tensor& x, Execution e) { return asum(x, e); }));
    add_op(unary("nrm2", "blas", [](const Tensor& x, Execution e) { return nrm2(x, e); }));
    add_op(unary("copy", "blas", [](const Tensor& x, Execution e) { return copy(x, e); }));
    add_op({"matvec", "blas", side, matrix_sizes, [](std::int64_t m, Device d, Execution e) {
                Tensor a = rnd({m, m}, 1, d), x = rnd({m}, 2, d);
                return Case{{a, x}, [=] { return matvec(a, x, e); }, 2.0 * m * m};
            }});
    add_op({"gemv", "blas", side, matrix_sizes, [](std::int64_t m, Device d, Execution e) {
                Tensor a = rnd({m, m}, 1, d), x = rnd({m}, 2, d), y = rnd({m}, 3, d);
                return Case{
                    {a, x, y}, [=] { return gemv(a, x, 1.5f, &y, 0.5f, false, e); }, 2.0 * m * m};
            }});
    add_op({"ger", "blas", side, matrix_sizes, [](std::int64_t m, Device d, Execution e) {
                Tensor x = rnd({m}, 1, d), y = rnd({m}, 2, d), a = rnd({m, m}, 3, d);
                return Case{{x, y, a}, [=] { return ger(1.5f, x, y, &a, e); }, 2.0 * m * m};
            }});
    add_op({"matmul", "blas", side, matrix_sizes, [](std::int64_t m, Device d, Execution e) {
                Tensor a = rnd({m, m}, 1, d), b = rnd({m, m}, 2, d);
                return Case{{a, b}, [=] { return matmul(a, b, e); }, 2.0 * m * m * m};
            }});
    add_op({"gemm", "blas", side, matrix_sizes, [](std::int64_t m, Device d, Execution e) {
                Tensor a = rnd({m, m}, 1, d), b = rnd({m, m}, 2, d), c = rnd({m, m}, 3, d);
                return Case{{a, b, c},
                            [=] { return gemm(a, b, 1.5f, &c, 0.5f, false, true, e); },
                            2.0 * m * m * m};
            }});
    add_op({"gemm_strided_batched",
            "blas",
            "matrix side N (batch 16)",
            {8, 16, 32, 64, 128, 256},
            [](std::int64_t m, Device d, Execution e) {
                Tensor a = rnd({16, m, m}, 1, d), b = rnd({16, m, m}, 2, d);
                return Case{{a, b},
                            [=] {
                                return gemm_strided_batched(a, b, 1.0f, nullptr, 0.0f, false, false,
                                                            e);
                            },
                            16 * 2.0 * m * m * m};
            }});

    // Elementwise, binary
    add_op(binary("add", "elementwise", add));
    add_op(binary("sub", "elementwise", sub));
    add_op(binary("mul", "elementwise", mul));
    add_op(binary("div", "elementwise", div));
    add_op(binary("pow", "elementwise", pow));
    add_op(binary("minimum", "elementwise", minimum));
    add_op(binary("maximum", "elementwise", maximum));
    add_op(binary("eq", "compare", eq));
    add_op(binary("ne", "compare", ne));
    add_op(binary("lt", "compare", lt));
    add_op(binary("le", "compare", le));
    add_op(binary("gt", "compare", gt));
    add_op(binary("ge", "compare", ge));
    add_op(
        {"axpy", "elementwise", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
             Tensor x = rnd({n}, 1, d), y = rnd({n}, 2, d);
             return Case{{x, y}, [=] { return axpy(1.5f, x, y, e); }, 2.0 * n};
         }});
    add_op({"bias_add", "elementwise", elements, element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd(rows_of_256(n), 1, d), bias = rnd({256}, 2, d);
                return Case{{x, bias}, [=] { return bias_add(x, bias, e); }};
            }});
    add_op({"where", "compare", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd({n}, 1, d), y = rnd({n}, 2, d);
                Tensor cond = gt(rnd({n}, 1), rnd({n}, 2), Execution::CPU).to(d);
                return Case{{cond, x, y}, [=] { return where(cond, x, y, e); }};
            }});
    add_op({"clamp", "elementwise", elements, element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd({n}, 1, d);
                return Case{{x}, [=] { return clamp(x, 0.3f, 0.7f, e); }};
            }});
    add_op(
        {"cast", "elementwise", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
             Tensor x = rnd({n}, 1, d);
             return Case{{x}, [=] { return cast(x, DType::I32, e); }};
         }});

    // Elementwise, unary (math and activations)
    add_op(unary("neg", "elementwise", neg));
    add_op(unary("abs", "elementwise", abs));
    add_op(unary("square", "elementwise", square));
    add_op(unary("sqrt", "elementwise", sqrt));
    add_op(unary("rsqrt", "elementwise", rsqrt));
    add_op(unary("reciprocal", "elementwise", reciprocal));
    add_op(unary("exp", "elementwise", exp));
    add_op(unary("log", "elementwise", log));
    add_op(unary("sin", "elementwise", sin));
    add_op(unary("cos", "elementwise", cos));
    add_op(unary("tanh", "activation", tanh));
    add_op(unary("sigmoid", "activation", sigmoid));
    add_op(unary("relu", "activation", relu));
    add_op(unary("silu", "activation", silu));
    add_op(unary("leaky_relu", "activation",
                 [](const Tensor& x, Execution e) { return leaky_relu(x, 0.01f, e); }));
    add_op(unary("gelu", "activation",
                 [](const Tensor& x, Execution e) { return gelu(x, false, e); }));
    add_op(
        unary("elu", "activation", [](const Tensor& x, Execution e) { return elu(x, 1.0f, e); }));
    add_op(unary("softplus", "activation",
                 [](const Tensor& x, Execution e) { return softplus(x, 1.0f, 20.0f, e); }));
    add_op(binary("relu_backward", "activation", relu_backward));
    add_op(binary("sigmoid_backward", "activation", sigmoid_backward));
    add_op(binary("tanh_backward", "activation", tanh_backward));
    add_op(binary("silu_backward", "activation", silu_backward));
    add_op(binary("gelu_backward", "activation", [](const Tensor& x, const Tensor& g, Execution e) {
        return gelu_backward(x, g, false, e);
    }));

    // Reductions
    add_op(unary("sum", "reduction", [](const Tensor& x, Execution e) { return sum(x, e); }));
    add_op(unary("mean", "reduction", [](const Tensor& x, Execution e) { return mean(x, e); }));
    add_op(unary("max", "reduction", [](const Tensor& x, Execution e) { return max(x, e); }));
    add_op(unary("min", "reduction", [](const Tensor& x, Execution e) { return min(x, e); }));
    add_op(unary("prod", "reduction", [](const Tensor& x, Execution e) { return prod(x, e); }));
    add_op(axis_reduce("sum(axis)", sum));
    add_op(axis_reduce("mean(axis)", mean));
    add_op(axis_reduce("max(axis)", max));
    add_op(axis_reduce("min(axis)", min));
    add_op(axis_reduce("prod(axis)", prod));
    add_op(
        {"argmax", "reduction", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
             Tensor x = rnd(rows_of_256(n), 1, d);
             return Case{{x}, [=] { return argmax(x, -1, e); }};
         }});

    // Movement
    add_op(
        {"concat", "movement", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
             Tensor a = rnd(rows_of_256(n / 2), 1, d), b = rnd(rows_of_256(n / 2), 2, d);
             return Case{{a, b}, [=] { return concat({a, b}, 0, e); }};
         }});
    add_op({"split", "movement", elements, element_sizes, [](std::int64_t n, Device d, Execution) {
                Tensor x = rnd(rows_of_256(n), 1, d);
                return Case{{x}, [=] { return split(x, std::int64_t{64}, 1)[0]; }};
            }});
    add_op({"pad", "movement", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
                Tensor x = rnd(rows_of_256(n), 1, d);
                return Case{{x}, [=] { return pad(x, {1, 1, 1, 1}, 0.0f, e); }};
            }});
    add_op(
        {"gather", "movement", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
             const Shape shape = rows_of_256(n);
             Tensor x = rnd(shape, 1, d), idx = indices(shape, 256, 7, d);
             return Case{{x, idx}, [=] { return gather(x, idx, 1, e); }};
         }});
    add_op({"index_select", "movement", elements, element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                const Shape shape = rows_of_256(n);
                Tensor x = rnd(shape, 1, d), idx = indices({shape[0]}, shape[0], 7, d);
                return Case{{x, idx}, [=] { return index_select(x, idx, 0, e); }};
            }});

    // Indexing
    add_op({"one_hot", "indexing", "output elements", element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                Tensor idx = indices({std::max<std::int64_t>(1, n / 16)}, 16, 7, d);
                return Case{{idx}, [=] { return one_hot(idx, 16, e); }};
            }});
    add_op({"embedding", "indexing", "output elements", element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                Tensor w = rnd({1024, 64}, 1, d);
                Tensor idx = indices({std::max<std::int64_t>(1, n / 64)}, 1024, 7, d);
                return Case{{w, idx}, [=] { return embedding(w, idx, e); }};
            }});
    add_op({"embedding_backward", "indexing", "gradient elements", element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                const std::int64_t count = std::max<std::int64_t>(1, n / 64);
                Tensor g = rnd({count, 64}, 1, d), idx = indices({count}, 1024, 7, d);
                return Case{{g, idx}, [=] { return embedding_backward(g, idx, 1024, e); }};
            }});

    // Random
    add_op({"uniform", "random", elements, element_sizes, [](std::int64_t n, Device, Execution e) {
                return Case{{}, [=] { return uniform({n}, 0.0f, 1.0f, 42, 0, e); }};
            }});
    add_op({"normal", "random", elements, element_sizes, [](std::int64_t n, Device, Execution e) {
                return Case{{}, [=] { return normal({n}, 0.0f, 1.0f, 42, 0, e); }};
            }});
    add_op(
        {"bernoulli", "random", elements, element_sizes, [](std::int64_t n, Device, Execution e) {
             return Case{{}, [=] { return bernoulli({n}, 0.5f, 42, 0, e); }};
         }});
    add_op(
        {"dropout", "random", elements, element_sizes, [](std::int64_t n, Device d, Execution e) {
             Tensor x = rnd({n}, 1, d);
             return Case{{x}, [=] { return dropout(x, 0.5f, true, 42, 0, e).output; }};
         }});
    add_op({"dropout_backward", "random", elements, element_sizes,
            [](std::int64_t n, Device d, Execution e) {
                Tensor g = rnd({n}, 1, d);
                Tensor mask = dropout(rnd({n}, 2), 0.5f, true, 42, 0, Execution::CPU).mask.to(d);
                return Case{{g, mask}, [=] { return dropout_backward(g, mask, 0.5f, true, e); }};
            }});
    return ops;
}

// ---- timing ---------------------------------------------------------------------------------------

// Reads one element, which waits until the op that produced `t` has finished on its device.
void wait_for(const Tensor& t) {
    if (t.defined() && t.numel() > 0) {
        unsigned char buffer[16];
        t.contiguous().view({t.numel()}).slice(0, 0, 1).copy_to_host(buffer);
    }
}

struct Timing {
    std::vector<double> samples_ms; // sorted
    std::int64_t output_bytes = 0;
};

Timing time_it(const std::function<Tensor()>& run, double min_time) {
    Timing timing;
    Tensor out = run(); // warm-up: first-touch memory, GPU pipeline compilation
    wait_for(out);
    timing.output_bytes = static_cast<std::int64_t>(out.nbytes());
    const auto start = Clock::now();
    while (timing.samples_ms.size() < 5 ||
           std::chrono::duration<double>(Clock::now() - start).count() < min_time) {
        const auto t0 = Clock::now();
        wait_for(run());
        timing.samples_ms.push_back(
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        if (timing.samples_ms.size() >= 5000) {
            break;
        }
    }
    std::sort(timing.samples_ms.begin(), timing.samples_ms.end());
    return timing;
}

double quantile(const std::vector<double>& sorted, double q) {
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}

// ---- helpers --------------------------------------------------------------------------------------

using observe::json_escape;
using observe::shape_json;

std::string num(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.9g", v);
    return buffer;
}

int threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

Execution parse_execution(const std::string& text, Device device) {
    if (text == "cpu") {
        return Execution::CPU;
    }
    if (text == "gpu") {
        return Execution::GPU;
    }
    if (text == "auto") {
        return Execution::Auto;
    }
    return device.type == DeviceType::CPU ? Execution::CPU : Execution::GPU;
}

std::string execution_label(Execution e) {
    return e == Execution::CPU ? "cpu" : e == Execution::GPU ? "gpu" : "auto";
}

std::vector<std::int64_t> parse_sizes(const std::string& text) {
    std::vector<std::int64_t> sizes;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            sizes.push_back(std::stoll(item));
        }
    }
    return sizes;
}

std::string sizes_json(const std::vector<std::int64_t>& sizes) {
    std::string out = "[";
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        out += (i ? "," : "") + std::to_string(sizes[i]);
    }
    return out + "]";
}

std::map<std::string, std::string> parse_flags(int argc, char** argv) {
    std::map<std::string, std::string> flags;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::invalid_argument("expected a --flag, got " + key);
        }
        flags[key.substr(2)] = argv[i + 1];
    }
    return flags;
}

std::string get(const std::map<std::string, std::string>& flags, const std::string& key,
                const std::string& fallback) {
    const auto it = flags.find(key);
    return it == flags.end() ? fallback : it->second;
}

const OpSpec& find_op(const std::vector<OpSpec>& ops, const std::string& name) {
    for (const OpSpec& op : ops) {
        if (op.name == name) {
            return op;
        }
    }
    throw std::invalid_argument("unknown op: " + name);
}

std::string header_json() {
    std::string out = "\"tracing\":";
    out += observe::TraceCollector::enabled() ? "true" : "false";
    out += ",\"threads\":" + std::to_string(threads());
    out += ",\"hardware_threads\":" + std::to_string(std::thread::hardware_concurrency());
#ifdef OBSERVE_SCALAR_BUILD
    out += ",\"build\":\"scalar\""; // library compiled without auto-vectorization
#else
    out += ",\"build\":\"simd\"";
#endif
    out += ",\"devices\":[";
    const std::vector<Device> devices = list_devices();
    for (std::size_t i = 0; i < devices.size(); ++i) {
        out += (i ? ",\"" : "\"") + to_string(devices[i]) + "\"";
    }
    return out + "]";
}

// ---- commands -------------------------------------------------------------------------------------

int cmd_list(const std::vector<OpSpec>& ops) {
    std::cout << "{" << header_json() << ",\"ops\":[";
    const std::vector<Device> devices = list_devices();
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const OpSpec& op = ops[i];
        std::cout << (i ? ",\n" : "\n") << "{\"name\":\"" << json_escape(op.name)
                  << "\",\"group\":\"" << op.group << "\",\"unit\":\"" << op.unit
                  << "\",\"sizes\":" << sizes_json(op.sizes) << ",\"support\":{";
        for (std::size_t k = 0; k < devices.size(); ++k) {
            const Device d = devices[k];
            std::string status = "ok";
            try {
                Case c = op.make(op.unit == std::string(elements) ? 4096 : 16, d,
                                 parse_execution("", d));
                wait_for(c.run());
            } catch (const std::exception& error) {
                status = error.what();
            }
            std::cout << (k ? "," : "") << "\"" << to_string(d) << "\":\"" << json_escape(status)
                      << "\"";
        }
        std::cout << "}}";
    }
    std::cout << "]}\n";
    return 0;
}

int cmd_run(const std::vector<OpSpec>& ops, const std::map<std::string, std::string>& flags) {
    const OpSpec& op = find_op(ops, get(flags, "op", ""));
    const Device device = parse_device(get(flags, "device", "cpu:0"));
    const Execution execution = parse_execution(get(flags, "execution", "default"), device);
    const std::string size_text = get(flags, "sizes", "");
    const std::vector<std::int64_t> sizes = size_text.empty() ? op.sizes : parse_sizes(size_text);
    const double min_time = std::stod(get(flags, "min-time", "0.25"));

    std::cout << "{" << header_json() << ",\"op\":\"" << json_escape(op.name) << "\",\"unit\":\""
              << op.unit << "\",\"device\":\"" << to_string(device) << "\",\"execution\":\""
              << execution_label(execution) << "\",\"results\":[";
    observe::TraceCollector collector;
    collector.stop();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const std::int64_t size = sizes[i];
        std::cout << (i ? ",\n" : "\n") << "{\"size\":" << size;
        try {
            Case c = op.make(size, device, execution);
            std::int64_t input_bytes = 0;
            std::string shapes = "[";
            for (std::size_t k = 0; k < c.inputs.size(); ++k) {
                input_bytes += static_cast<std::int64_t>(c.inputs[k].nbytes());
                shapes += (k ? "," : "") + shape_json(c.inputs[k].shape());
            }
            shapes += "]";
            const Timing t = time_it(c.run, min_time);

            collector.clear();
            collector.start();
            wait_for(c.run());
            collector.stop();

            const double median = quantile(t.samples_ms, 0.5);
            const double bytes = static_cast<double>(input_bytes + t.output_bytes);
            const bool compute = c.flops > 0 && op.unit != std::string(elements);
            const double rate =
                compute ? c.flops / (median * 1e-3) / 1e9 : bytes / (median * 1e-3) / 1e9;

            std::cout << ",\"inputs\":" << shapes << ",\"bytes\":" << num(bytes)
                      << ",\"flops\":" << num(c.flops) << ",\"calls\":" << t.samples_ms.size()
                      << ",\"median_ms\":" << num(median)
                      << ",\"min_ms\":" << num(t.samples_ms.front())
                      << ",\"max_ms\":" << num(t.samples_ms.back())
                      << ",\"p10_ms\":" << num(quantile(t.samples_ms, 0.1))
                      << ",\"p90_ms\":" << num(quantile(t.samples_ms, 0.9))
                      << ",\"rate\":" << num(rate) << ",\"rate_unit\":\""
                      << (compute ? "GFLOP/s" : "GB/s") << "\",\"samples_ms\":[";
            const std::size_t step = std::max<std::size_t>(1, t.samples_ms.size() / 200);
            for (std::size_t k = 0; k < t.samples_ms.size(); k += step) {
                std::cout << (k ? "," : "") << num(t.samples_ms[k]);
            }
            std::cout << "],\"trace\":" << collector.events_json();
        } catch (const std::exception& error) {
            std::cout << ",\"error\":\"" << json_escape(error.what()) << "\"";
        }
        std::cout << "}";
        std::cout.flush();
    }
    std::cout << "]}\n";
    return 0;
}

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}

observe::Record client_span(const std::string& name, std::int64_t start, std::int64_t end) {
    observe::Record r;
    r.name = name;
    r.device = "client";
    r.start_ns = start;
    r.end_ns = end;
    return r;
}

int cmd_call(const std::vector<OpSpec>& ops, const std::map<std::string, std::string>& flags) {
    const OpSpec& op = find_op(ops, get(flags, "op", ""));
    const Device device = parse_device(get(flags, "device", "cpu:0"));
    const Execution execution = parse_execution(get(flags, "execution", "default"), device);
    const std::string size_text = get(flags, "size", "");
    const std::int64_t size =
        size_text.empty() ? op.sizes[op.sizes.size() / 2] : std::stoll(size_text);
    const int calls = std::max(1, std::stoi(get(flags, "calls", "25")));

    Case c = op.make(size, device, execution);
    std::string shapes = "[";
    for (std::size_t k = 0; k < c.inputs.size(); ++k) {
        shapes += (k ? "," : "") + shape_json(c.inputs[k].shape());
    }
    shapes += "]";
    for (int i = 0; i < 3; ++i) {
        wait_for(c.run()); // warm-up
    }

    std::vector<double> untraced;
    for (int i = 0; i < calls; ++i) {
        const std::int64_t t0 = now_ns();
        wait_for(c.run());
        untraced.push_back(static_cast<double>(now_ns() - t0) / 1e6);
    }

    observe::TraceCollector collector;
    collector.stop();
    std::vector<double> traced;
    std::vector<std::string> call_json;
    for (int i = 0; i < calls; ++i) {
        collector.clear();
        collector.start();
        const std::int64_t t0 = now_ns();
        Tensor out = c.run();
        const std::int64_t t1 = now_ns();
        wait_for(out);
        const std::int64_t t2 = now_ns();
        collector.stop();
        collector.add(client_span("call", t0, t2));
        collector.add(client_span(op.name + "()", t0, t1));
        collector.add(client_span("read result", t1, t2));
        traced.push_back(static_cast<double>(t2 - t0) / 1e6);
        call_json.push_back("{\"total_ms\":" + num(traced.back()) +
                            ",\"events\":" + collector.events_json(t0) + "}");
    }
    std::sort(untraced.begin(), untraced.end());
    std::vector<double> sorted = traced;
    std::sort(sorted.begin(), sorted.end());

    std::cout << "{" << header_json() << ",\"op\":\"" << json_escape(op.name) << "\",\"unit\":\""
              << op.unit << "\",\"size\":" << size << ",\"inputs\":" << shapes << ",\"device\":\""
              << to_string(device) << "\",\"execution\":\"" << execution_label(execution)
              << "\",\"untraced_median_ms\":" << num(quantile(untraced, 0.5))
              << ",\"traced_median_ms\":" << num(quantile(sorted, 0.5)) << ",\"calls\":[";
    for (std::size_t i = 0; i < call_json.size(); ++i) {
        std::cout << (i ? ",\n" : "\n") << call_json[i];
    }
    std::cout << "]}\n";
    return 0;
}

// One training step of a small MLP: the kind of program the trace view is for.
int cmd_program(const std::map<std::string, std::string>& flags) {
    const Device device = parse_device(get(flags, "device", "cpu:0"));
    const Execution e = parse_execution(get(flags, "execution", "default"), device);
    const int iterations = std::stoi(get(flags, "iterations", "3"));

    observe::TraceCollector collector;
    collector.stop();
    std::string error;
    auto begin = Clock::now(); // reset when the traced part starts, after the inputs are made
    try {
        // demo:begin (serve.py shows the lines between these markers in the Trace tab)
        Tensor x = rnd({256, 512}, 1, device);
        Tensor w1 = rnd({512, 1024}, 2, device);
        Tensor b1 = rnd({1024}, 3, device);
        Tensor w2 = rnd({1024, 64}, 4, device);
        w1.set_requires_grad();
        w2.set_requires_grad();
        begin = Clock::now();
        collector.start();
        for (int i = 0; i < iterations; ++i) {
            Tensor h = relu(bias_add(matmul(x, w1, e), b1, e), e);
            Tensor logits = matmul(h, w2, e);
            Tensor p = exp(logits, e);
            Tensor z = sum(p, 1, true, e);
            Tensor loss = mean(div(p, z, e), e);
            wait_for(loss);
            try {
                loss.backward();
                wait_for(w1.grad());
            } catch (const std::exception& backward_error) {
                error = std::string("backward: ") + backward_error.what();
            }
            Tensor joined = concat({h, h}, 1, e);
            wait_for(pad(joined, {1, 1}, 0.0f, e));
            wait_for(dropout(h, 0.1f, true, 7, 0, e).output);
        }
        // demo:end
    } catch (const std::exception& run_error) {
        error = run_error.what();
    }
    collector.stop();
    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    std::cout << "{" << header_json() << ",\"device\":\"" << to_string(device)
              << "\",\"execution\":\"" << execution_label(e) << "\",\"total_ms\":" << num(total_ms)
              << ",\"error\":\"" << json_escape(error) << "\",\"traceEvents\":"
              << collector.events_json(
                     std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch())
                         .count())
              << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "list";
        const std::vector<OpSpec> ops = make_ops();
        const auto flags = parse_flags(argc, argv);
        if (command == "list") {
            return cmd_list(ops);
        }
        if (command == "run") {
            return cmd_run(ops, flags);
        }
        if (command == "call") {
            return cmd_call(ops, flags);
        }
        if (command == "program") {
            return cmd_program(flags);
        }
        throw std::invalid_argument("unknown command: " + command +
                                    " (list, run, call or program)");
    } catch (const std::exception& error) {
        std::cout << "{\"fatal\":\"" << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
