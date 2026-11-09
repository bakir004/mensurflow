// Direct CPU casts into preallocated buffers, optionally paired with native ATen copy_.
#include "bench.hpp"
#include "elementwise_cutoffs.hpp"

#include "backends/cpu/ops/elementwise/cast.hpp"
#include "backends/cpu/threads.hpp"
#include "reference/cast.hpp"

#ifdef MENSURFLOW_TORCH_REFERENCE
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/Version.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#endif

#include <chrono>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace ref = reference::casts;
using mensurflow::DType;

namespace {

const DType types[] = {DType::F32, DType::I32, DType::F16, DType::BF16};

// Keep real objects of the source/destination type, with the same page-offset spacing used by
// the shared harness. Each independent buffer needs at most one extra 4 KB page of elements.
class CastBuffer {
  public:
    CastBuffer(std::size_t n, DType type, std::size_t page_offset) {
        switch (type) {
        case DType::F32:
            allocate<float>(n, page_offset);
            break;
        case DType::I32:
            allocate<std::int32_t>(n, page_offset);
            break;
        case DType::F16:
        case DType::BF16:
            allocate<std::uint16_t>(n, page_offset);
            break;
        default:
            throw std::invalid_argument("unsupported cast buffer dtype");
        }
    }

    CastBuffer(const CastBuffer&) = delete;
    CastBuffer& operator=(const CastBuffer&) = delete;

    void* data() { return data_; }

    static std::size_t max_elements() {
        return std::min({capacity<float>(), capacity<std::int32_t>(), capacity<std::uint16_t>()});
    }

  private:
    static constexpr std::size_t page_bytes = 4096;

    template <typename T> static std::size_t capacity() {
        constexpr auto padding = page_bytes / sizeof(T);
        const auto maximum = std::vector<T>().max_size();
        return maximum > padding ? maximum - padding : 0;
    }

    template <typename T> void allocate(std::size_t n, std::size_t page_offset) {
        if (n > capacity<T>()) {
            throw std::invalid_argument("size exceeds typed cast buffer allocation limit");
        }
        auto& values = storage_.emplace<std::vector<T>>(n + page_bytes / sizeof(T));
        const auto base = reinterpret_cast<std::uintptr_t>(values.data());
        const auto padding_bytes =
            (page_offset + page_bytes - (base & (page_bytes - 1))) & (page_bytes - 1);
        data_ = values.data() + padding_bytes / sizeof(T);
    }

    std::variant<std::vector<float>, std::vector<std::int32_t>, std::vector<std::uint16_t>>
        storage_;
    void* data_ = nullptr;
};

std::string name(DType source, DType destination) {
    return std::string(mensurflow::dtype_name(source)) + "_to_" +
           mensurflow::dtype_name(destination);
}

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("expected positive integer list values");
        }
        const auto value = std::stoll(token);
        if (value <= 0 || std::find(values.begin(), values.end(), value) != values.end()) {
            throw std::invalid_argument("list values must be positive and unique");
        }
        values.push_back(value);
    }
    if (values.empty() || text.back() == ',') {
        throw std::invalid_argument("expected a nonempty comma-separated list");
    }
    return values;
}

void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

#ifdef MENSURFLOW_TORCH_REFERENCE
at::ScalarType torch_type(DType type) {
    switch (type) {
    case DType::F32:
        return at::kFloat;
    case DType::I32:
        return at::kInt;
    case DType::F16:
        return at::kHalf;
    case DType::BF16:
        return at::kBFloat16;
    default:
        throw std::invalid_argument("unsupported reference dtype");
    }
}
#endif

int run(int argc, char** argv) {
    const int maximum_threads = cpu::max_threads();
    auto threads = bench::thread_counts(maximum_threads);
    std::vector<std::int64_t> sizes{1000, 32769, 4000003};
    std::set<std::string> selected;
    bench::ElementwiseCutoffOption cutoff;
    bool reverse = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "cast_benchmark [f32_to_bf16 ...] [--sizes=n,...] [--threads=n,...] "
                         "[--cutoff=n] [--reverse]\n"
                         "All 16 pairs among f32/i32/f16/bf16 are supported.\n"
                         "--reverse times ATen first in the optional native reference build.\n";
            return 0;
        }
        if (cutoff.parse(arg)) {
            continue;
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
            for (auto source : types) {
                for (auto destination : types) {
                    known = known || arg == name(source, destination);
                }
            }
            if (!known) {
                throw std::invalid_argument("unknown conversion or option: " + arg);
            }
            selected.insert(arg);
        }
    }
    const auto maximum_elements = CastBuffer::max_elements();
    for (auto n : sizes) {
        if (static_cast<std::uint64_t>(n) > maximum_elements) {
            throw std::invalid_argument("size exceeds typed cast buffer allocation limit");
        }
    }
    std::vector<std::string> notes{
        "MensurFlow cutoff: " + cutoff.configure("cast"),
        "preallocated buffers; direct CPU kernels; allocation, wrapping and Python are not timed",
        "rate: source bytes + destination bytes per element; logical traffic, not DRAM bandwidth",
        "input: repeating integers [-2046,2046] for I32; quarter increments for floating sources",
        "source values are encoded independently; all buffers use their actual element type",
        "every conversion checked against a long-double mathematical reference before timing",
        "nine samples; 500 ms worker-settling pause before each measurement; repeat after cooling"};
#ifdef MENSURFLOW_TORCH_REFERENCE
    c10::InferenceMode inference_guard;
    at::set_num_interop_threads(1);
    notes.push_back(std::string("LibTorch version: ") + TORCH_VERSION);
    notes.push_back("ATen TensorIterator GRAIN_SIZE=" + std::to_string(at::internal::GRAIN_SIZE));
    notes.push_back("reference: native ATen copy_ including dispatch; same buffers and threads");
    notes.push_back(reverse ? "order: ATen then MensurFlow" : "order: MensurFlow then ATen");
    const char* ratio_label = "vs torch";
#else
    (void)reverse;
    if (std::find(threads.begin(), threads.end(), 1) == threads.end()) {
        threads.push_back(1);
    }
    const char* ratio_label = "vs seq";
#endif
    std::sort(threads.begin(), threads.end());
    bench::Config config;
    config.slow_repeats = config.repeats;
    bench::print_header("CPU cast comparison", config, notes);
#ifdef MENSURFLOW_TORCH_REFERENCE
    std::cout << "\nLibTorch build configuration:\n" << at::show_config()
              << "\nInitial LibTorch parallel configuration:\n" << at::get_parallel_info() << '\n';
#endif
    bench::Table table("GB/s", 1e9, ratio_label, "");
    for (auto source : types) {
        for (auto destination : types) {
            const auto operation = name(source, destination);
            if (!selected.empty() && selected.count(operation) == 0) {
                continue;
            }
            table.section(operation);
            for (auto n : sizes) {
                const auto len = static_cast<std::size_t>(n);
                CastBuffer input_buffer(len, source, 0);
                CastBuffer output_buffer(len, destination, 1024);
                CastBuffer expected_buffer(len, destination, 2048);
                void* input = input_buffer.data();
                void* output = output_buffer.data();
                void* expected = expected_buffer.data();
                for (std::size_t i = 0; i < len; ++i) {
                    const auto integer = static_cast<int>(i % 4093) - 2046;
                    const long double value = source == DType::I32 ? integer : integer / 4.0L;
                    ref::store(input, source, i, ref::encode(value, source));
                }
                ref::cast(input, source, expected, destination, len);
                auto ours = [&] { cpu::cast_buffer(input, source, output, destination, n); };
                auto observe = [&] {
                    double checksum = 0;
                    for (std::size_t i = 0; i < len; ++i) {
                        checksum += ref::load(output, destination, i);
                    }
                    return checksum;
                };
                const auto check = [&](const char* library) {
                    if (!ref::equal(expected, output, destination, len, source == destination)) {
                        throw std::runtime_error(operation + " " + library + " reference mismatch");
                    }
                };
#ifdef MENSURFLOW_TORCH_REFERENCE
                const auto x =
                    at::from_blob(input, {n}, at::TensorOptions().dtype(torch_type(source)));
                auto out =
                    at::from_blob(output, {n}, at::TensorOptions().dtype(torch_type(destination)));
                auto torch = [&] { out.copy_(x); };
#else
                bench::Sample baseline;
#endif
                for (int count : threads) {
                    cpu::set_max_threads(count);
#ifdef MENSURFLOW_TORCH_REFERENCE
                    at::set_num_threads(count);
                    if (cpu::max_threads() != count || at::get_num_threads() != count) {
                        throw std::runtime_error("libraries did not accept the same thread count");
                    }
#endif
                    std::memset(output, 0xa5, len * mensurflow::dtype_size(destination));
                    ours();
                    check("MensurFlow");
                    bench::Sample own_sample;
                    const auto measure_ours = [&] {
                        settle();
                        own_sample = bench::measure(config, ours, observe);
                    };
#ifdef MENSURFLOW_TORCH_REFERENCE
                    std::memset(output, 0xa5, len * mensurflow::dtype_size(destination));
                    torch();
                    check("ATen");
                    if (out.data_ptr() != output) {
                        throw std::runtime_error("ATen replaced the output buffer");
                    }
                    bench::Sample torch_sample;
                    const auto measure_torch = [&] {
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
                    const auto* comparison = &torch_sample;
#else
                    measure_ours();
                    if (count == 1) {
                        baseline = own_sample;
                    }
                    const auto* comparison = &baseline;
#endif
                    const double traffic =
                        static_cast<double>(n) *
                        (mensurflow::dtype_size(source) + mensurflow::dtype_size(destination));
                    table.row(std::to_string(n), "mensurflow", std::to_string(count), own_sample,
                              traffic, comparison);
#ifdef MENSURFLOW_TORCH_REFERENCE
                    table.row(std::to_string(n), "pytorch", std::to_string(count), torch_sample,
                              traffic, comparison);
#endif
                }
                table.blank();
            }
        }
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "cast_benchmark: " << e.what() << '\n';
        return 1;
    }
}
