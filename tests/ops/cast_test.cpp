#include "backends/cpu/ops/elementwise/cast.hpp"
#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"
#include "backends/cpu/threads.hpp"
#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/cast.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;
namespace ref = reference::casts;

namespace {

const DType types[] = {DType::F32, DType::I32, DType::F16, DType::BF16};
using Bytes = std::vector<unsigned char>;

Bytes bytes(const Tensor& input) {
    Bytes result(input.nbytes());
    input.copy_to_host(result.data());
    return result;
}

Bytes make_input(DType type, std::size_t n) {
    Bytes result(n * dtype_size(type));
    const std::uint32_t float_edges[] = {
        0, 0x80000000U, 0x3f800000U, 0xbf800000U, 0x7f800000U, 0xff800000U,
        0x7fc12345U, 0x7f800001U, 0xff800001U, 1, 0x80000001U, 0x007fffffU,
        0x00800000U, 0x477fe000U, 0x477ff000U, 0x33000000U, 0x33800000U,
        0x4f000000U, 0xcf000000U, 0x4effffffU, 0xcf000001U, 0x7f7fffffU,
        0x3f808000U, 0x3f818000U, 0x3fc00000U, 0xbfc00000U};
    std::uint32_t state = 17;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525U + 1013904223U;
        auto raw = state;
        if (type == DType::F32 && i < sizeof(float_edges) / sizeof(*float_edges)) {
            raw = float_edges[i];
        } else if (type == DType::I32 && i < 4) {
            const std::uint32_t edges[] = {0, 0x7fffffffU, 0x80000000U, 0xffffffffU};
            raw = edges[i];
        } else if (dtype_size(type) == 2) {
            raw = static_cast<std::uint32_t>(i) & 0xffffU;
        }
        ref::store(result.data(), type, i, raw);
    }
    return result;
}

void check_conversion(const Tensor& input, DType destination, Execution execution = Execution::CPU) {
    const auto before = bytes(input);
    Bytes expected(static_cast<std::size_t>(input.numel()) * dtype_size(destination));
    ref::cast(before.data(), input.dtype(), expected.data(), destination,
              static_cast<std::size_t>(input.numel()));
    const auto result = cast(input, destination, execution);
    const auto actual = bytes(result);
    const std::string label = std::string(dtype_name(input.dtype())) + " -> " + dtype_name(destination);
    test::require(result.shape() == input.shape() && result.dtype() == destination &&
                      result.device() == input.device(), label + " metadata");
    test::require(result.is_contiguous() && !result.shares_storage(input), label + " fresh output");
    test::require(!result.requires_grad() && !result.has_grad(), label + " forward-only result");
    test::require(ref::equal(expected.data(), actual.data(), destination,
                             static_cast<std::size_t>(input.numel()), input.dtype() == destination),
                  label + " independent mathematical reference");
    test::require(bytes(input) == before, label + " preserves input bits");
}

void check_all_pairs() {
    const int original_threads = cpu::max_threads();
    const auto cutoff = cpu::elementwise_parallel_cutoff(cpu::ElementwiseOp::cast);
    for (DType source : types) {
        // Exhaust every binary16/bfloat16 encoding, including subnormals and signaling NaNs.
        const std::size_t n = dtype_size(source) == 2 ? 65536 : 8192;
        const auto input = make_input(source, n);
        const auto tensor = Tensor::from_host(input.data(), {static_cast<std::int64_t>(n)}, source);
        for (int threads : {1, 2, 3}) {
            cpu::set_max_threads(threads);
            for (DType destination : types) {
                check_conversion(tensor, destination);
            }
        }
    }
    // Exercise all pairs on each side of a custom cutoff without relying on an untuned default.
    cpu::set_elementwise_parallel_cutoff(cpu::ElementwiseOp::cast, 17);
    for (int n : {0, 1, 16, 17, 18, 67}) {
        for (DType source : types) {
            const auto input = make_input(source, static_cast<std::size_t>(n));
            const auto tensor = Tensor::from_host(input.data(), {n}, source);
            for (DType destination : types) {
                check_conversion(tensor, destination);
            }
        }
    }
    cpu::set_elementwise_parallel_cutoff(cpu::ElementwiseOp::cast, cutoff);
    for (auto n : {cutoff - 1, cutoff, cutoff + 1}) {
        for (DType source : types) {
            const auto input = make_input(source, static_cast<std::size_t>(n));
            const auto tensor = Tensor::from_host(input.data(), {n}, source);
            for (DType destination : types) {
                check_conversion(tensor, destination);
            }
        }
    }
    cpu::set_max_threads(original_threads);
}

void check_rounding() {
    for (DType target : {DType::F16, DType::BF16}) {
        const std::uint32_t largest = target == DType::F16 ? 0x7bffU : 0x7f7fU;
        std::vector<float> values;
        for (std::uint32_t raw = 0; raw < largest; ++raw) {
            const auto midpoint = static_cast<float>((ref::decode(raw, target) +
                                                      ref::decode(raw + 1, target)) / 2);
            for (float value : {std::nextafter(midpoint, 0.0f), midpoint,
                                std::nextafter(midpoint, std::numeric_limits<float>::infinity())}) {
                values.push_back(value);
                values.push_back(-value);
            }
        }
        check_conversion(Tensor::from_vector(values, {static_cast<std::int64_t>(values.size())}),
                         target);
    }
    // Values just around BF16 midpoints above 2^24 detect double rounding via F32.
    std::vector<std::int32_t> integers;
    for (int exponent = 24; exponent <= 30; ++exponent) {
        const std::int64_t step = std::int64_t{1} << (exponent - 7);
        for (int significand = 128; significand < 256; ++significand) {
            const auto midpoint = significand * step + step / 2;
            for (int delta : {-1, 0, 1}) {
                integers.push_back(static_cast<std::int32_t>(midpoint + delta));
                integers.push_back(static_cast<std::int32_t>(-midpoint - delta));
            }
        }
    }
    const auto input = Tensor::from_vector(integers, {static_cast<std::int64_t>(integers.size())});
    for (DType destination : types) {
        check_conversion(input, destination);
    }
}

void check_views_and_errors() {
    for (DType source : types) {
        const auto data = make_input(source, 24);
        Tensor input = Tensor::from_host(data.data(), {4, 6}, source);
        if (source == DType::F32) {
            input.set_requires_grad();
        }
        for (DType destination : types) {
            check_conversion(input, destination, Execution::Auto);
            check_conversion(input.transpose(0, 1), destination);
            check_conversion(input.slice(0, 1, 4).slice(1, 1, 6, 2), destination);
            check_conversion(input.slice(0, 1, 2), destination); // contiguous, nonzero offset
            check_conversion(input.slice(0, 1, 2).expand({3, 6}), destination);
            check_conversion(input.slice(0, 0, 1).slice(1, 0, 1).view({}), destination);
            check_conversion(input.slice(0, 1, 1), destination);
        }
        test::require(detail::Registry::instance().dispatcher().find(
                          {detail::OpId::Cast, DeviceType::CPU, source}) != nullptr,
                      "source dtype is registered");
        for (auto bad : {DType::I64, DType::U8, DType::Bool}) {
            for (const auto& tensor : {input, input.slice(0, 0, 0)}) {
                test::require_throws<detail::NotImplemented>([&] { cast(tensor, bad); },
                                                            "unsupported target including empty");
            }
            for (Shape shape : {Shape{2}, Shape{0}}) {
                const auto unsupported = Tensor::empty(shape, bad);
                test::require_throws<detail::NotImplemented>([&] { cast(unsupported, source); },
                                                            "unsupported source including empty");
            }
        }
        test::require_throws<std::invalid_argument>([&] { cast(input, static_cast<DType>(-1)); },
                                                    "invalid dtype enum");
    }
    test::require_throws<std::invalid_argument>([] { cast(Tensor{}, DType::F32); }, "undefined input");
    test::require(std::string(detail::op_name(detail::OpId::Cast)) == "cast", "dispatcher name");
    for (DType source : types) {
        for (DType destination : types) {
            cpu::cast_buffer(nullptr, source, nullptr, destination, 0);
        }
    }
    test::require_throws<std::invalid_argument>(
        [] { cpu::cast_buffer(nullptr, DType::U8, nullptr, DType::F32, 0); }, "raw invalid source");
    test::require_throws<std::invalid_argument>(
        [] { cpu::cast_buffer(nullptr, DType::F32, nullptr, DType::Bool, 0); }, "raw invalid target");
    std::vector<float> in_place{0.0f, -0.0f, 1.0f};
    cpu::cast_buffer(in_place.data(), DType::F32, in_place.data(), DType::F32, 3);
    test::require(in_place[2] == 1.0f && std::signbit(in_place[1]), "raw same-dtype alias");
    const auto input = Tensor::from_vector(std::vector<float>{1}, {});
    if (list_devices().size() == 1) {
        test::require_throws<std::runtime_error>([&] { cast(input, DType::F16, Execution::GPU); },
                                                 "GPU request without a GPU");
    } else {
        for (auto device : list_devices()) {
            if (device.type == DeviceType::CPU) {
                continue;
            }
            for (DType source : types) {
                const auto data = make_input(source, 24);
                const auto gpu = Tensor::from_host(data.data(), {4, 6}, source, device);
                for (DType destination : types) {
                    check_conversion(gpu, destination, Execution::CPU);
                    check_conversion(gpu, destination, Execution::Auto);
                    test::require_throws<detail::NotImplemented>(
                        [&] { cast(gpu, destination, Execution::GPU); }, "no GPU cast kernel");
                }
            }
        }
    }
}

} // namespace

int main() {
    return test::run("cast_test", [] {
        check_all_pairs();
        check_rounding();
        check_views_and_errors();
    });
}
