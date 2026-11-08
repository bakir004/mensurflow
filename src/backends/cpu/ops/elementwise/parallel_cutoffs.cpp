#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"

#include <array>
#include <atomic>
#include <stdexcept>

namespace mensurflow::cpu {
namespace {

struct Cutoffs {
    std::array<std::atomic<std::int64_t>, elementwise_op_count> values;

    Cutoffs() {
        for (const auto& policy : elementwise_parallel_policies) {
            auto& value = values[static_cast<std::size_t>(policy.op)];
            std::atomic_init(&value, policy.default_cutoff);
        }
    }
};

Cutoffs& cutoffs() {
    static Cutoffs state;
    return state;
}

std::size_t index(ElementwiseOp op) {
    const auto value = static_cast<std::size_t>(op);
    if (value >= elementwise_op_count) {
        throw std::invalid_argument("invalid elementwise operation for parallel cutoff");
    }
    return value;
}

} // namespace

std::int64_t elementwise_parallel_cutoff(ElementwiseOp op) {
    const auto i = index(op);
    return cutoffs().values[i].load(std::memory_order_relaxed);
}

void set_elementwise_parallel_cutoff(ElementwiseOp op, std::int64_t cutoff) {
    const auto i = index(op);
    if (cutoff < 0) {
        throw std::invalid_argument("elementwise parallel cutoff must be nonnegative");
    }
    cutoffs().values[i].store(cutoff, std::memory_order_relaxed);
}

} // namespace mensurflow::cpu
