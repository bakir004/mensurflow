// Measure forced serial/parallel branches of existing kernels at matched thread counts.
#include "bench.hpp"

#include "backends/cpu/ops/elementwise/cast.hpp"
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"
#include "backends/cpu/threads.hpp"
#include "reference/cast.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <variant>

namespace cpu = mensurflow::cpu;
using mensurflow::DType;
using ParallelOp = cpu::ElementwiseOp;

namespace {

// Actual element objects, separated at fixed offsets modulo 4 KB like bench::Arrays.
class Buffer {
  public:
    Buffer(std::size_t n, DType type, std::size_t page_offset) {
        switch (type) {
        case DType::F32: allocate<float>(n, page_offset); break;
        case DType::I32: allocate<std::int32_t>(n, page_offset); break;
        case DType::F16:
        case DType::BF16: allocate<std::uint16_t>(n, page_offset); break;
        case DType::Bool: allocate<std::uint8_t>(n, page_offset); break;
        default: throw std::invalid_argument("unsupported buffer dtype");
        }
    }
    template <typename T> T* data() { return static_cast<T*>(data_); }
    void* data() { return data_; }
    static std::size_t max_elements() {
        return std::min({capacity<float>(), capacity<std::int32_t>(),
                         capacity<std::uint16_t>(), capacity<std::uint8_t>()});
    }
  private:
    template <typename T> static std::size_t capacity() {
        const auto maximum = std::vector<T>().max_size();
        return maximum > 4096 / sizeof(T) ? maximum - 4096 / sizeof(T) : 0;
    }
    template <typename T> void allocate(std::size_t n, std::size_t page_offset) {
        if (n > capacity<T>()) {
            throw std::invalid_argument("size exceeds typed buffer limit");
        }
        auto& values = storage_.emplace<std::vector<T>>(n + 4096 / sizeof(T));
        const auto base = reinterpret_cast<std::uintptr_t>(values.data());
        const auto padding = (page_offset + 4096 - (base & 4095)) & 4095;
        data_ = values.data() + padding / sizeof(T);
    }
    std::variant<std::vector<float>, std::vector<std::int32_t>,
                 std::vector<std::uint16_t>, std::vector<std::uint8_t>> storage_;
    void* data_ = nullptr;
};

using FloatKernel = void (*)(const float*, const float*, float*, std::int64_t);
using ComparisonKernel = void (*)(const float*, const float*, std::uint8_t*, std::int64_t);
enum class Kind { Float, Comparison, Where, Bias, Cast };
struct Case {
    std::string name;
    ParallelOp op;
    Kind kind = Kind::Float;
    FloatKernel kernel = nullptr;
    ComparisonKernel compare = nullptr;
    DType source = DType::F32;
    DType destination = DType::F32;
    std::int64_t features = 0;
};

template <void (*Unary)(const float*, float*, std::int64_t)>
void unary(const float* x, const float*, float* out, std::int64_t n) { Unary(x, out, n); }
void leaky_relu(const float* x, const float*, float* out, std::int64_t n) {
    cpu::leaky_relu_f32(x, out, n, 0.01f);
}
void gelu(const float* x, const float*, float* out, std::int64_t n) {
    cpu::gelu_f32(x, out, n);
}
void gelu_tanh(const float* x, const float*, float* out, std::int64_t n) {
    cpu::gelu_f32(x, out, n, true);
}
void elu(const float* x, const float*, float* out, std::int64_t n) {
    cpu::elu_f32(x, out, n, 1.0f);
}
void softplus(const float* x, const float*, float* out, std::int64_t n) {
    cpu::softplus_f32(x, out, n, 1.0f, 20.0f);
}
void clamp(const float* x, const float*, float* out, std::int64_t n) {
    cpu::clamp_f32(x, out, n, -1.0f, 1.0f);
}
void axpy(const float* x, const float* y, float* out, std::int64_t n) {
    cpu::axpy_f32(1.25f, x, y, out, n);
}

std::vector<Case> cases() {
    std::vector<Case> result;
    const auto add = [&](const char* name, ParallelOp op, FloatKernel kernel) {
        result.push_back({name, op, Kind::Float, kernel});
    };
#define BINARY(Name) add(#Name, ParallelOp::Name, cpu::Name##_f32)
#define UNARY(Name) add(#Name, ParallelOp::Name, unary<cpu::Name##_f32>)
    BINARY(add); BINARY(sub); BINARY(mul); BINARY(div); BINARY(pow);
    BINARY(minimum); BINARY(maximum);
    UNARY(neg); UNARY(abs); UNARY(square); UNARY(relu); UNARY(sqrt); UNARY(exp);
    UNARY(sigmoid); UNARY(log); UNARY(rsqrt); UNARY(tanh); UNARY(sin); UNARY(cos);
    UNARY(reciprocal); UNARY(silu);
#undef BINARY
#undef UNARY
    add("leaky_relu", ParallelOp::leaky_relu, leaky_relu);
    add("gelu", ParallelOp::gelu, gelu);
    add("gelu_tanh", ParallelOp::gelu_tanh, gelu_tanh);
    add("elu", ParallelOp::elu, elu);
    add("softplus", ParallelOp::softplus, softplus);
    add("clamp", ParallelOp::clamp, clamp);
    add("axpy", ParallelOp::axpy, axpy);
    for (auto features : {1, 64, 256}) {
        Case op{"bias_add_f" + std::to_string(features), ParallelOp::bias_add, Kind::Bias};
        op.features = features;
        result.push_back(op);
    }
#define COMPARE(Name) \
    result.push_back({#Name, ParallelOp::Name, Kind::Comparison, nullptr, cpu::Name##_f32, \
                      DType::F32, DType::Bool})
    COMPARE(eq); COMPARE(ne); COMPARE(lt); COMPARE(le); COMPARE(gt); COMPARE(ge);
#undef COMPARE
    result.push_back({"where", ParallelOp::where, Kind::Where});
    for (DType source : {DType::F32, DType::I32, DType::F16, DType::BF16}) {
        for (DType destination : {DType::F32, DType::I32, DType::F16, DType::BF16}) {
            result.push_back({std::string("cast_") + mensurflow::dtype_name(source) + "_to_" +
                              mensurflow::dtype_name(destination), ParallelOp::cast,
                              Kind::Cast, nullptr, nullptr, source, destination});
        }
    }
    return result;
}

const char* policy_name(ParallelOp op) {
    return cpu::elementwise_parallel_policies[static_cast<std::size_t>(op)].name;
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty() || std::find(values.begin(), values.end(), token) != values.end()) {
            throw std::invalid_argument("list values must be nonempty and unique");
        }
        values.push_back(token);
    }
    if (values.empty() || text.back() == ',') {
        throw std::invalid_argument("expected a nonempty comma-separated list");
    }
    return values;
}

std::int64_t integer(const std::string& text, bool allow_zero = false) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected an unsigned integer");
    }
    const auto value = std::stoll(text);
    if (!allow_zero && value == 0) {
        throw std::invalid_argument("expected a positive integer");
    }
    return value;
}

std::vector<std::int64_t> numbers(const std::string& text) {
    std::vector<std::int64_t> result;
    for (const auto& token : split(text)) {
        const auto value = integer(token);
        if (std::find(result.begin(), result.end(), value) != result.end()) {
            throw std::invalid_argument("numeric list values must be unique");
        }
        result.push_back(value);
    }
    return result;
}

struct Buffers {
    Buffer x, y, out, mask;
    Buffers(const Case& op, std::size_t n)
        : x(n, op.source, 0), y(op.kind == Kind::Cast ? 0 : n, DType::F32, 1024),
          out(n, op.destination, 2048), mask(op.kind == Kind::Where ? n : 0, DType::Bool, 3072) {
        if (op.kind == Kind::Cast) {
            for (std::size_t i = 0; i < n; ++i) {
                const auto value = static_cast<int>(i % 4093) - 2046;
                reference::casts::store(x.data(), op.source, i,
                    reference::casts::encode(op.source == DType::I32 ? value : value / 4.0L,
                                             op.source));
            }
        } else {
            bench::fill_a(x.data<float>(), n);
            bench::fill_b(y.data<float>(), n);
            if (op.op == ParallelOp::sqrt || op.op == ParallelOp::rsqrt ||
                op.op == ParallelOp::log || op.op == ParallelOp::pow) {
                for (std::size_t i = 0; i < n; ++i) {
                    x.data<float>()[i] = std::fabs(x.data<float>()[i]) + 0.25f;
                }
            }
            if (op.kind == Kind::Where) {
                for (std::size_t i = 0; i < n; ++i) {
                    mask.data<std::uint8_t>()[i] = i % 5 < 2 ? 1 : 0;
                }
            }
        }
    }
    void run(const Case& op, std::int64_t n) {
        switch (op.kind) {
        case Kind::Float:
            return op.kernel(x.data<float>(), y.data<float>(), out.data<float>(), n);
        case Kind::Comparison:
            return op.compare(x.data<float>(), y.data<float>(), out.data<std::uint8_t>(), n);
        case Kind::Where:
            return cpu::where_f32(mask.data<std::uint8_t>(), x.data<float>(), y.data<float>(),
                                  out.data<float>(), n);
        case Kind::Bias:
            return cpu::bias_add_f32(x.data<float>(), y.data<float>(), out.data<float>(), n,
                                     op.features);
        case Kind::Cast:
            return cpu::cast_buffer(x.data(), op.source, out.data(), op.destination, n);
        }
    }
};

void emit(const Case& op, std::int64_t n, int threads,
          const bench::Sample& serial, const bench::Sample& parallel) {
    std::cout << std::setprecision(17) << "CUT\t" << op.name << '\t' << policy_name(op.op)
              << '\t' << n << '\t' << threads << '\t'
              << serial.median_s << '\t' << serial.min_s << '\t' << serial.max_s << '\t'
              << parallel.median_s << '\t' << parallel.min_s << '\t' << parallel.max_s << '\t'
              << serial.median_s / parallel.median_s << '\t'
              << serial.cpu_over_wall << '\t' << parallel.cpu_over_wall << '\t'
              << serial.calls_per_sample << '\t' << parallel.calls_per_sample;
    for (const auto* sample : {&serial, &parallel}) {
        std::cout << '\t';
        for (std::size_t i = 0; i < sample->runs_s.size(); ++i) {
            std::cout << (i ? "," : "") << sample->runs_s[i];
        }
    }
    std::cout << '\n' << std::flush;
}

int run(int argc, char** argv) {
    const auto available = cases();
    const int original_threads = cpu::max_threads();
    auto sizes = std::vector<std::int64_t>{512, 4096, 32768, 262144};
    std::vector<int> threads;
    for (auto count : {2, 6, 12}) {
        count = std::min(count, original_threads);
        if (std::find(threads.begin(), threads.end(), count) == threads.end()) {
            threads.push_back(count);
        }
    }
    std::set<std::string> selected;
    bool reverse = false;
    int settle_ms = 500;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--list") {
            for (const auto& op : available) {
                std::cout << op.name << '\t' << policy_name(op.op) << '\t' << op.features << '\n';
            }
            return 0;
        }
        if (arg == "--help") {
            std::cout << "parallel_cutoff_benchmark [--cases=name,...] [--sizes=n,...] "
                         "[--threads=n,...] [--reverse] [--settle-ms=500] [--list]\n"
                         "Forced serial/parallel branches share output addresses and requested threads.\n"
                         "Defaults: all cases, sizes512,4096,32768,262144, threads2,6,12 clipped to max.\n"
                         "Official measurements use500ms settle; 0 is only for smoke checks.\n";
            return 0;
        }
        if (arg == "--reverse") {
            reverse = true;
        } else if (arg.rfind("--cases=", 0) == 0) {
            for (const auto& name : split(arg.substr(8))) {
                const auto found = std::find_if(available.begin(), available.end(),
                                                [&](const Case& op) { return op.name == name; });
                if (found == available.end()) {
                    throw std::invalid_argument("unknown case: " + name);
                }
                selected.insert(name);
            }
        } else if (arg.rfind("--sizes=", 0) == 0) {
            sizes = numbers(arg.substr(8));
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads.clear();
            for (auto count : numbers(arg.substr(10))) {
                if (count > original_threads) {
                    throw std::invalid_argument("thread count exceeds initial OpenMP maximum");
                }
                threads.push_back(static_cast<int>(count));
            }
        } else if (arg.rfind("--settle-ms=", 0) == 0) {
            const auto value = integer(arg.substr(12), true);
            if (value > 60000) {
                throw std::invalid_argument("settle milliseconds must be in[0,60000]");
            }
            settle_ms = static_cast<int>(value);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    for (auto n : sizes) {
        if (static_cast<std::uint64_t>(n) > Buffer::max_elements() ||
            static_cast<std::uint64_t>(n) > std::vector<unsigned char>().max_size() / 4) {
            throw std::invalid_argument("size exceeds benchmark buffer limit");
        }
    }
    auto operations = available;
    operations.erase(std::remove_if(operations.begin(), operations.end(), [&](const Case& op) {
        return !selected.empty() && selected.count(op.name) == 0;
    }), operations.end());
    if (reverse) {
        std::reverse(operations.begin(), operations.end());
        std::reverse(sizes.begin(), sizes.end());
        std::reverse(threads.begin(), threads.end());
    }
    bench::Config config;
    config.slow_repeats = config.repeats;
    bench::print_header("Matched-thread serial/parallel cutoff measurement", config,
        {"cutoffs: serial=INT64_MAX, parallel=0; same input/output addresses per pair",
         "modes: " + std::string(reverse ? "parallel then serial" : "serial then parallel"),
         "settle milliseconds before each mode: " + std::to_string(settle_ms),
         "bias_add fixes features at1/64/256; non-divisible n is skipped",
         "cast: all16pairs, integer I32 sources and quarter-increment floating sources",
         "independent typed buffers at page offsets0/1024/2048; where mask offset3072",
         "float input: shared fill_a/fill_b; sqrt/rsqrt/log/pow receive abs(x)+0.25",
         "serial/parallel outputs are checked bitwise outside timing; no allocation is timed"});
    std::cout << "CUT_HEADER\tcase\tpolicy\tn\tthreads\tserial_median_s\tserial_min_s\tserial_max_s"
                 "\tparallel_median_s\tparallel_min_s\tparallel_max_s\tspeedup\tserial_cpu_wall"
                 "\tparallel_cpu_wall\tserial_calls\tparallel_calls\tserial_runs_s\tparallel_runs_s\n";
    for (const auto& op : operations) {
        const auto original_cutoff = cpu::elementwise_parallel_cutoff(op.op);
        for (auto n : sizes) {
            if (op.features != 0 && n % op.features != 0) {
                std::cout << "SKIP\t" << op.name << '\t' << n << "\tnot divisible by features\n";
                continue;
            }
            Buffers buffers(op, static_cast<std::size_t>(n));
            const auto bytes = static_cast<std::size_t>(n) * mensurflow::dtype_size(op.destination);
            std::vector<unsigned char> expected(bytes);
            auto observe = [&] {
                const auto* output = buffers.out.data<std::uint8_t>();
                std::uint64_t checksum = 0;
                for (std::size_t i = 0; i < bytes; ++i) { checksum += output[i]; }
                return static_cast<double>(checksum);
            };
            auto kernel = [&] { buffers.run(op, n); };
            for (int count : threads) {
                cpu::set_max_threads(count);
                cpu::set_elementwise_parallel_cutoff(op.op, std::numeric_limits<std::int64_t>::max());
                kernel();
                std::memcpy(expected.data(), buffers.out.data(), bytes);
                cpu::set_elementwise_parallel_cutoff(op.op, 0);
                std::memset(buffers.out.data(), 0xa5, bytes);
                kernel();
                if (std::memcmp(expected.data(), buffers.out.data(), bytes) != 0) {
                    throw std::runtime_error(op.name + ": serial/parallel outputs differ");
                }
                auto measure = [&](std::int64_t cutoff) {
                    cpu::set_elementwise_parallel_cutoff(op.op, cutoff);
                    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
                    return bench::measure(config, kernel, observe);
                };
                bench::Sample serial, parallel;
                if (reverse) {
                    parallel = measure(0);
                    serial = measure(std::numeric_limits<std::int64_t>::max());
                } else {
                    serial = measure(std::numeric_limits<std::int64_t>::max());
                    parallel = measure(0);
                }
                emit(op, n, count, serial, parallel);
            }
        }
        cpu::set_elementwise_parallel_cutoff(op.op, original_cutoff);
    }
    cpu::set_max_threads(original_threads);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) {
        std::cerr << "parallel_cutoff_benchmark: " << e.what() << '\n';
        return 1;
    }
}
