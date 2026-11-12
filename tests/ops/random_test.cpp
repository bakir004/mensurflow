#include "mensurflow/mensurflow.hpp"
#include "mensurflow/ops/random.hpp"

#include "backends/cpu/ops/random/philox.hpp"
#include "backends/cpu/ops/random/random.hpp"
#include "backends/cpu/threads.hpp"
#include "core/dispatcher.hpp"
#include "reference/random.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

using namespace mensurflow;

namespace {

std::vector<std::uint8_t> mask_values(const Tensor& mask) {
    std::vector<std::uint8_t> values(static_cast<std::size_t>(mask.numel()));
    mask.copy_to_host(values.data());
    return values;
}

void require_values(const std::vector<float>& actual, const std::vector<float>& expected,
                    const std::string& name, float tolerance = 0.0f) {
    test::require(test::matches_f32(actual, expected, tolerance), name);
}

void known_answers() {
    // Published independent Random123 Philox4x32-10 known-answer vectors:
    // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    using cpu::PhiloxCounter;
    test::require(cpu::philox4x32_10({0, 0, 0, 0}, {0, 0}) ==
                      PhiloxCounter{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u},
                  "Philox all-zero known answer");
    test::require(cpu::philox4x32_10({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
                                    {0xffffffffu, 0xffffffffu}) ==
                      PhiloxCounter{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu},
                  "Philox all-one known answer");
    test::require(cpu::philox4x32_10({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
                                    {0xa4093822u, 0x299f31d0u}) ==
                      PhiloxCounter{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u},
                  "Philox mixed known answer");
}

void kernel_references() {
    const int maximum = cpu::max_threads();
    const std::set<int> threads{1, 2, 3, maximum};
    for (std::int64_t n : {0, 1, 3, 4, 5, 32767, 32768, 32769, 65539}) {
        for (std::uint64_t offset : {std::uint64_t{0}, std::uint64_t{3},
                                     (std::uint64_t{1} << 34) - 1}) {
            const std::uint64_t seed = 0xfedcba9876543210ull;
            const auto uref = reference::uniform(n, -2.5f, 4.75f, seed, offset);
            const auto nref = reference::normal(n, -1.25f, 2.5f, seed, offset);
            const auto bref = reference::bernoulli(n, 0.3f, seed, offset);
            std::vector<float> output(static_cast<std::size_t>(n));
            std::vector<float> normal_sequential;
            for (int count : threads) {
                cpu::set_max_threads(count);
                cpu::uniform_f32(output.data(), n, -2.5f, 4.75f, seed, offset);
                require_values(output, uref, "uniform reference and thread counts");
                cpu::normal_f32(output.data(), n, -1.25f, 2.5f, seed, offset);
                require_values(output, nref, "normal scalar reference", 2e-6f);
                if (count == 1) {
                    normal_sequential = output;
                } else {
                    require_values(output, normal_sequential, "normal bit-identical threads");
                }
                cpu::bernoulli_f32(output.data(), n, 0.3f, seed, offset);
                require_values(output, bref, "bernoulli reference and thread counts");
            }
        }
    }
    cpu::set_max_threads(maximum);
}

void factories_and_streams() {
    test::require(uniform({}).shape().empty(), "uniform scalar shape");
    test::require(normal({}).dtype() == DType::F32, "normal F32 result");
    test::require(bernoulli({}).device() == Device{}, "bernoulli CPU result");
    for (const auto& value : {uniform({2, 0, 3}), normal({2, 0, 3}), bernoulli({2, 0, 3})}) {
        test::require(value.numel() == 0 && value.shape() == Shape{2, 0, 3}, "RNG empty shape");
    }

    const std::uint64_t seed = 123;
    for (std::uint64_t start : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2},
                                std::uint64_t{3}, (std::uint64_t{1} << 34) - 3}) {
        const auto check_chunks = [start](auto generate, const char* name) {
            const auto whole = generate(37, start).template to_vector<float>();
            for (std::uint64_t split : {1u, 2u, 3u, 4u, 17u, 36u}) {
                auto chunks = generate(static_cast<std::int64_t>(split), start)
                                  .template to_vector<float>();
                const auto tail = generate(37 - static_cast<std::int64_t>(split), start + split)
                                      .template to_vector<float>();
                chunks.insert(chunks.end(), tail.begin(), tail.end());
                require_values(chunks, whole, std::string(name) + " stream chunking");
            }
        };
        check_chunks([=](std::int64_t n, std::uint64_t offset) {
            return uniform({n}, -3.0f, 4.0f, seed, offset, Execution::CPU);
        }, "uniform");
        check_chunks([=](std::int64_t n, std::uint64_t offset) {
            return normal({n}, 2.0f, 3.0f, seed, offset, Execution::CPU);
        }, "normal");
        check_chunks([=](std::int64_t n, std::uint64_t offset) {
            return bernoulli({n}, 0.6f, seed, offset, Execution::CPU);
        }, "bernoulli");
    }
    test::require(uniform({32}, 0, 1, 123).to_vector<float>() !=
                      uniform({32}, 0, 1, 124).to_vector<float>(), "different seeds");
    test::require(uniform({32}, 0, 1, 123).to_vector<float>() !=
                      uniform({32}, 0, 1, (std::uint64_t{1} << 32) + 123).to_vector<float>(),
                  "seed high word used");
    const auto last = std::numeric_limits<std::uint64_t>::max();
    require_values(uniform({1}, 0, 1, 0, last).to_vector<float>(),
                   reference::uniform(1, 0, 1, 0, last), "last uint64 sample");
    require_values(normal({1}, 0, 1, 0, last).to_vector<float>(),
                   reference::normal(1, 0, 1, 0, last), "last uint64 normal sample", 2e-6f);

    require_values(uniform({9}, 4, 4).to_vector<float>(), std::vector<float>(9, 4),
                   "equal uniform bounds");
    require_values(normal({9}, 4, 0).to_vector<float>(), std::vector<float>(9, 4),
                   "zero normal stddev");
    require_values(normal({9}, -0.0f, 0).to_vector<float>(), std::vector<float>(9, -0.0f),
                   "zero normal stddev preserves signed mean");
    require_values(uniform({9}, -0.0f, 0.0f).to_vector<float>(), std::vector<float>(9, -0.0f),
                   "equal uniform bounds preserve signed low");
    require_values(bernoulli({9}, 0).to_vector<float>(), std::vector<float>(9, 0), "bernoulli p0");
    require_values(bernoulli({9}, 1).to_vector<float>(), std::vector<float>(9, 1), "bernoulli p1");
    const float largest = std::numeric_limits<float>::max();
    for (float value : uniform({1000}, -largest, largest).to_vector<float>()) {
        test::require(std::isfinite(value) && value >= -largest && value < largest,
                      "wide uniform finite interval");
    }
    const float low = 1.0f, high = std::nextafter(low, 2.0f);
    require_values(uniform({1000}, low, high).to_vector<float>(), std::vector<float>(1000, low),
                   "uniform upper bound excluded after rounding");
}

void distribution_moments() {
    constexpr std::int64_t n = 200000;
    const auto u = uniform({n}, 0, 1, 2026).to_vector<float>();
    const auto z = normal({n}, 0, 1, 2026).to_vector<float>();
    const auto b = bernoulli({n}, 0.25f, 2026).to_vector<float>();
    double usum = 0, zsum = 0, zsquare = 0, bsum = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        test::require(u[i] >= 0 && u[i] < 1, "uniform bounds");
        test::require(std::isfinite(z[i]), "normal finite samples");
        test::require(b[i] == 0 || b[i] == 1, "bernoulli binary samples");
        usum += u[i];
        zsum += z[i];
        zsquare += static_cast<double>(z[i]) * z[i];
        bsum += b[i];
    }
    test::require(std::abs(usum / n - 0.5) < 0.005, "uniform mean");
    test::require(std::abs(zsum / n) < 0.015 && std::abs(zsquare / n - 1) < 0.025,
                  "normal mean and variance");
    test::require(std::abs(bsum / n - 0.25) < 0.005, "bernoulli frequency");
}

void dropout_references_and_threads() {
    const int maximum = cpu::max_threads();
    for (std::int64_t n : {0, 1, 32767, 32768, 32769, 262143, 262144, 262145}) {
        std::vector<float> input(static_cast<std::size_t>(n)), gradient(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 4;
            gradient[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8;
        }
        const auto expected_mask = reference::dropout_mask(n, 0.3f, true, 8, 3);
        const auto expected_output = reference::dropout(input, expected_mask, 0.3f, true);
        const auto expected_gradient = reference::dropout(gradient, expected_mask, 0.3f, true);
        for (int count : std::set<int>{1, 2, 3, maximum}) {
            cpu::set_max_threads(count);
            std::vector<float> output(input.size()), backward(input.size());
            std::vector<std::uint8_t> mask(input.size());
            cpu::dropout_f32(input.data(), output.data(), mask.data(), n, 0.3f, true, 8, 3);
            test::require(mask == expected_mask, "dropout mask thread-independent reference");
            require_values(output, expected_output, "dropout forward reference");
            cpu::dropout_backward_f32(gradient.data(), mask.data(), backward.data(), n, 0.3f, true);
            require_values(backward, expected_gradient, "dropout backward reference");
        }
    }
    cpu::set_max_threads(maximum);
}

void dropout_api_and_autograd() {
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    const Tensor unusual = Tensor::from_vector(std::vector<float>{nan, inf, -inf, -0.0f}, {4});
    for (bool training : {false, true}) {
        const auto result = dropout(unusual, 0.0f, training);
        require_values(result.output.to_vector<float>(), unusual.to_vector<float>(),
                       "dropout p0 copies nonfinite/signed zero");
        test::require(mask_values(result.mask) == std::vector<std::uint8_t>(4, 1),
                      "dropout p0 mask");
    }
    const auto eval = dropout(unusual, 1.0f, false);
    require_values(eval.output.to_vector<float>(), unusual.to_vector<float>(),
                   "dropout eval copies");
    require_values(dropout_backward(unusual, eval.mask, 1.0f, false).to_vector<float>(),
                   unusual.to_vector<float>(), "dropout eval backward copies");
    const auto all_dropped = dropout(unusual, 1.0f, true);
    require_values(all_dropped.output.to_vector<float>(), std::vector<float>(4, 0),
                   "dropout p1 masks nonfinite values");
    require_values(dropout_backward(unusual, all_dropped.mask, 1.0f).to_vector<float>(),
                   std::vector<float>(4, 0), "dropout p1 backward masks nonfinite values");

    const Tensor base = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor view = base.transpose(0, 1);
    const auto result = dropout(view, 0.25f, true, 7, 1);
    const auto expected_mask = reference::dropout_mask(6, 0.25f, true, 7, 1);
    test::require(result.output.shape() == view.shape() && result.output.is_contiguous(),
                  "dropout view output");
    test::require(result.mask.dtype() == DType::Bool && mask_values(result.mask) == expected_mask,
                  "dropout public Bool mask");
    require_values(result.output.to_vector<float>(),
                   reference::dropout(view.to_vector<float>(), expected_mask, 0.25f, true),
                   "dropout strided input");
    const Tensor broadcast = base.slice(0, 0, 1).expand({5, 3});
    const auto broadcast_result = dropout(broadcast, 0.5f, true, 2);
    require_values(broadcast_result.output.to_vector<float>(),
                   reference::dropout(broadcast.to_vector<float>(),
                                      reference::dropout_mask(15, 0.5f, true, 2, 0), 0.5f, true),
                   "dropout broadcast view");
    const Tensor gradient = Tensor::full(view.shape(), 2.0f);
    const Tensor strided_mask = result.mask.transpose(0, 1);
    require_values(dropout_backward(gradient.transpose(0, 1), strided_mask, 0.25f)
                       .to_vector<float>(),
                   reference::dropout(gradient.transpose(0, 1).to_vector<float>(),
                                      mask_values(strided_mask), 0.25f, true),
                   "dropout backward strided inputs");
    const auto empty = dropout(Tensor::empty({2, 0, 3}));
    test::require(empty.output.numel() == 0 && empty.mask.shape() == Shape{2, 0, 3},
                  "dropout empty shape");
    test::require(dropout_backward(empty.output, empty.mask).numel() == 0,
                  "dropout backward empty shape");

    for (float probability : {0.0f, 0.25f, 1.0f}) {
        for (bool training : {false, true}) {
            Tensor x = Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
            x.set_requires_grad();
            auto differentiable = dropout(x, probability, training, 7, 1);
            const auto original_mask = mask_values(differentiable.mask);
            // Returned masks are mutable through copy_from; autograd needs private saved data.
            differentiable.mask.copy_from(Tensor::zeros(x.shape(), DType::Bool));
            const Tensor upstream =
                Tensor::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
            sum(mul(differentiable.output, upstream)).backward();
            require_values(x.grad().to_vector<float>(),
                           reference::dropout(upstream.to_vector<float>(), original_mask,
                                              probability, training),
                           "dropout autograd saved mask");
        }
    }
}

void dropout_view_autograd() {
    Tensor base = Tensor::ones({2, 3});
    base.set_requires_grad();
    const Tensor view = base.transpose(0, 1);
    const std::vector<float> coefficients{1, 2, 3, 4, 5, 6};
    const auto result = dropout(view, 0.25f, true, 7, 1);
    sum(mul(result.output, Tensor::from_vector(coefficients, {3, 2}))).backward();
    const auto view_gradient = reference::dropout(
        coefficients, reference::dropout_mask(6, 0.25f, true, 7, 1), 0.25f, true);
    test::require(base.grad().shape() == base.shape(), "dropout view gradient base shape");
    require_values(base.grad().to_vector<float>(),
                   {view_gradient[0], view_gradient[2], view_gradient[4],
                    view_gradient[1], view_gradient[3], view_gradient[5]},
                   "dropout backward through transpose");

    Tensor broadcast_base = Tensor::ones({1, 3});
    broadcast_base.set_requires_grad();
    const auto broadcast = dropout(broadcast_base.expand({4, 3}), 0.5f, true, 3);
    const std::vector<float> broadcast_coefficients{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    sum(mul(broadcast.output, Tensor::from_vector(broadcast_coefficients, {4, 3}))).backward();
    const auto broadcast_gradient = reference::dropout(
        broadcast_coefficients, reference::dropout_mask(12, 0.5f, true, 3, 0), 0.5f, true);
    std::vector<float> expected_broadcast(3, 0.0f);
    for (std::size_t i = 0; i < broadcast_gradient.size(); ++i) {
        expected_broadcast[i % 3] += broadcast_gradient[i];
    }
    test::require(broadcast_base.grad().shape() == broadcast_base.shape(),
                  "dropout broadcast gradient base shape");
    require_values(broadcast_base.grad().to_vector<float>(), expected_broadcast,
                   "dropout backward sums expanded elements");

    Tensor sliced_base = Tensor::ones({2, 5});
    sliced_base.set_requires_grad();
    const auto sliced = dropout(sliced_base.slice(1, 1, 5, 2), 0.5f, true, 11);
    const std::vector<float> sliced_coefficients{1, 2, 3, 4};
    sum(mul(sliced.output, Tensor::from_vector(sliced_coefficients, {2, 2}))).backward();
    const auto sliced_gradient = reference::dropout(
        sliced_coefficients, reference::dropout_mask(4, 0.5f, true, 11, 0), 0.5f, true);
    require_values(sliced_base.grad().to_vector<float>(),
                   {0, sliced_gradient[0], 0, sliced_gradient[1], 0,
                    0, sliced_gradient[2], 0, sliced_gradient[3], 0},
                   "dropout backward through stepped slice");
}

void validation() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float value : {nan, inf, -inf}) {
        test::require_throws<std::invalid_argument>([&] { uniform({0}, value, 1); },
                                                    "uniform finite low");
        test::require_throws<std::invalid_argument>([&] { uniform({0}, 0, value); },
                                                    "uniform finite high");
        test::require_throws<std::invalid_argument>([&] { normal({0}, value, 1); },
                                                    "normal finite mean");
        test::require_throws<std::invalid_argument>([&] { normal({0}, 0, value); },
                                                    "normal finite stddev");
    }
    test::require_throws<std::invalid_argument>([] { uniform({0}, 2, 1); }, "uniform bounds");
    test::require_throws<std::invalid_argument>([] { normal({0}, 0, -1); }, "negative stddev");
    test::require_throws<std::invalid_argument>([] { uniform({-1}); }, "negative shape");
    test::require_throws<std::invalid_argument>([] { normal({-1}); }, "normal negative shape");
    test::require_throws<std::invalid_argument>([] { bernoulli({-1}); },
                                                "bernoulli negative shape");
    const auto last = std::numeric_limits<std::uint64_t>::max();
    test::require_throws<std::invalid_argument>([&] { uniform({2}, 0, 1, 0, last); },
                                                "uniform offset overflow");
    test::require_throws<std::invalid_argument>([&] { normal({2}, 0, 1, 0, last); },
                                                "normal offset overflow");
    test::require_throws<std::invalid_argument>([&] { bernoulli({2}, 0.5f, 0, last); },
                                                "bernoulli offset overflow");
    const Tensor x = Tensor::ones({4});
    const Tensor mask = Tensor::ones({4}, DType::Bool);
    for (float probability : {-0.1f, 1.1f, nan, inf}) {
        test::require_throws<std::invalid_argument>([&] { bernoulli({0}, probability); },
                                                    "bernoulli invalid p");
        test::require_throws<std::invalid_argument>([&] { dropout(x, probability); },
                                                    "dropout invalid p");
        test::require_throws<std::invalid_argument>([&] { dropout_backward(x, mask, probability); },
                                                    "dropout backward invalid p");
    }
    test::require_throws<std::invalid_argument>([&] { dropout(Tensor{}); }, "dropout undefined");
    test::require_throws<std::invalid_argument>([&] { dropout(x, 0.5f, true, 0, last); },
                                                "dropout offset overflow");
    test::require_throws<std::invalid_argument>([&] { dropout_backward(Tensor{}, mask); },
                                                "dropout backward undefined gradient");
    test::require_throws<std::invalid_argument>([&] { dropout_backward(x, Tensor{}); },
                                                "dropout backward undefined mask");
    test::require_throws<std::invalid_argument>([&] { dropout_backward(x, x); },
                                                "dropout backward mask dtype");
    test::require_throws<std::invalid_argument>([&] {
        dropout_backward(x, Tensor::ones({2, 2}, DType::Bool));
    }, "dropout backward mask shape");
    for (std::int64_t n : {0, 4}) {
        const Tensor unsupported = Tensor::zeros({n}, DType::I32);
        test::require_throws<detail::NotImplemented>([&] { dropout(unsupported); },
                                                    "dropout unsupported dtype incl empty");
        test::require_throws<detail::NotImplemented>([&] {
            dropout_backward(unsupported, Tensor::zeros({n}, DType::Bool));
        }, "dropout backward unsupported dtype incl empty");
        test::require_throws<std::runtime_error>([&] {
            uniform({n}, 0, 1, 0, 0, Execution::GPU);
        }, "uniform GPU unsupported incl empty");
        test::require_throws<std::runtime_error>([&] {
            normal({n}, 0, 1, 0, 0, Execution::GPU);
        }, "normal GPU unsupported incl empty");
        test::require_throws<std::runtime_error>([&] {
            bernoulli({n}, 0.5f, 0, 0, Execution::GPU);
        }, "bernoulli GPU unsupported incl empty");
        test::require_throws<std::runtime_error>([&] {
            dropout(Tensor::ones({n}), 0.5f, true, 0, 0, Execution::GPU);
        }, "dropout GPU unsupported incl empty");
    }
}

} // namespace

int main() {
    return test::run("random_test", [] {
        known_answers();
        kernel_references();
        factories_and_streams();
        distribution_moments();
        dropout_references_and_threads();
        dropout_api_and_autograd();
        dropout_view_autograd();
        validation();
    });
}
