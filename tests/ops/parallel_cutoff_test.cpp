#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"
#include "backends/cpu/threads.hpp"
#include "reference/elementwise.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

// Restore process-wide configuration even when a check throws.
struct RestorePolicy {
    const int threads = cpu::max_threads();
    std::vector<std::int64_t> cutoffs;

    RestorePolicy() {
        for (const auto& policy : cpu::elementwise_parallel_policies) {
            cutoffs.push_back(cpu::elementwise_parallel_cutoff(policy.op));
        }
    }

    ~RestorePolicy() {
        for (std::size_t i = 0; i < cutoffs.size(); ++i) {
            cpu::set_elementwise_parallel_cutoff(cpu::elementwise_parallel_policies[i].op,
                                                cutoffs[i]);
        }
        cpu::set_max_threads(threads);
    }
};

void check_configuration() {
    const RestorePolicy restore;
    for (const auto& changed : cpu::elementwise_parallel_policies) {
        const auto before = cpu::elementwise_parallel_cutoff(changed.op);
        cpu::set_elementwise_parallel_cutoff(changed.op, 17);
        for (std::size_t i = 0; i < cpu::elementwise_op_count; ++i) {
            const auto op = cpu::elementwise_parallel_policies[i].op;
            test::require(cpu::elementwise_parallel_cutoff(op) ==
                              (op == changed.op ? 17 : restore.cutoffs[i]),
                          "a cutoff override affects only its operation");
        }
        test::require_throws<std::invalid_argument>(
            [&] { cpu::set_elementwise_parallel_cutoff(changed.op, -1); },
            "negative cutoff is rejected");
        test::require(cpu::elementwise_parallel_cutoff(changed.op) == 17,
                      "rejected cutoff leaves configuration intact");
        cpu::set_elementwise_parallel_cutoff(changed.op, before);
    }
    for (auto op : {cpu::ElementwiseOp::count, static_cast<cpu::ElementwiseOp>(-1)}) {
        test::require_throws<std::invalid_argument>(
            [&] { cpu::elementwise_parallel_cutoff(op); }, "invalid cutoff lookup");
        test::require_throws<std::invalid_argument>(
            [&] { cpu::set_elementwise_parallel_cutoff(op, 0); }, "invalid cutoff override");
    }
}

void check_kernel_paths() {
    const RestorePolicy restore;
    const auto serial = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t n : {0, 1, 16, 17, 18, 67}) {
        const auto len = static_cast<std::size_t>(n);
        std::vector<float> x(len), y(len), expected(len), actual(len);
        std::vector<std::uint8_t> mask(len), flags(len);
        for (std::size_t i = 0; i < len; ++i) {
            x[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.25f;
            y[i] = static_cast<float>(i % 7) * 0.5f;
            mask[i] = i % 3 == 0 ? 0 : 7;
        }
        for (int threads : {1, 2, 3}) {
            cpu::set_max_threads(threads);
            for (std::int64_t cutoff : {std::int64_t{0}, std::int64_t{17}, serial}) {
                for (auto op : {cpu::ElementwiseOp::add, cpu::ElementwiseOp::leaky_relu,
                                cpu::ElementwiseOp::gelu, cpu::ElementwiseOp::gelu_tanh,
                                cpu::ElementwiseOp::eq, cpu::ElementwiseOp::where}) {
                    cpu::set_elementwise_parallel_cutoff(op, cutoff);
                }
                reference::add(x.data(), y.data(), expected.data(), n);
                cpu::add_f32(x.data(), y.data(), actual.data(), n);
                test::require(test::matches_f32(actual, expected), "add across cutoff paths");
                reference::leaky_relu(x.data(), expected.data(), n);
                cpu::leaky_relu_f32(x.data(), actual.data(), n, 0.01f);
                test::require(test::matches_f32(actual, expected), "shared activation cutoff");
                for (bool approximate : {false, true}) {
                    if (approximate) {
                        reference::gelu_tanh(x.data(), expected.data(), n);
                    } else {
                        reference::gelu(x.data(), expected.data(), n);
                    }
                    cpu::gelu_f32(x.data(), actual.data(), n, approximate);
                    test::require(test::matches_f32(actual, expected, 3e-6f),
                                  "both GELU cutoff paths match the independent reference");
                }
                cpu::eq_f32(x.data(), y.data(), flags.data(), n);
                cpu::where_f32(mask.data(), x.data(), y.data(), actual.data(), n);
                for (std::size_t i = 0; i < len; ++i) {
                    test::require(flags[i] == static_cast<std::uint8_t>(x[i] == y[i]),
                                  "shared comparison cutoff");
                    const float& selected = mask[i] ? x[i] : y[i];
                    test::require(std::memcmp(&actual[i], &selected, sizeof(float)) == 0,
                                  "where selects the same bits across cutoff paths");
                }
            }
        }
    }
}

} // namespace

int main() {
    return test::run("parallel_cutoff_test", [] {
        check_configuration();
        check_kernel_paths();
    });
}
