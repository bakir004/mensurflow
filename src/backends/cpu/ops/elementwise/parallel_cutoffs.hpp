// Internal CPU elementwise scheduling policy. This does not change Tensor dispatch.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mensurflow::cpu {

enum class ElementwiseOp {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    where,
    add,
    sub,
    mul,
    div,
    pow,
    minimum,
    maximum,
    axpy,
    bias_add,
    neg,
    abs,
    square,
    relu,
    leaky_relu,
    gelu,
    gelu_tanh,
    silu,
    elu,
    softplus,
    sqrt,
    exp,
    sigmoid,
    log,
    rsqrt,
    tanh,
    sin,
    cos,
    reciprocal,
    clamp,
    cast,
    count
};

struct ElementwiseParallelPolicy {
    ElementwiseOp op;
    const char* name;
    std::int64_t default_cutoff;
};

// Independent defaults, initially preserving the existing behavior. Update individual rows only
// after repeated crossover measurements; these values are not claimed to be tuned.
inline constexpr ElementwiseParallelPolicy elementwise_parallel_policies[] = {
    {ElementwiseOp::eq, "eq", 32768},
    {ElementwiseOp::ne, "ne", 32768},
    {ElementwiseOp::lt, "lt", 32768},
    {ElementwiseOp::le, "le", 32768},
    {ElementwiseOp::gt, "gt", 32768},
    {ElementwiseOp::ge, "ge", 32768},
    {ElementwiseOp::where, "where", 32768},
    {ElementwiseOp::add, "add", 32768},
    {ElementwiseOp::sub, "sub", 32768},
    {ElementwiseOp::mul, "mul", 32768},
    {ElementwiseOp::div, "div", 32768},
    {ElementwiseOp::pow, "pow", 32768},
    {ElementwiseOp::minimum, "minimum", 32768},
    {ElementwiseOp::maximum, "maximum", 32768},
    {ElementwiseOp::axpy, "axpy", 32768},
    {ElementwiseOp::bias_add, "bias_add", 32768},
    {ElementwiseOp::neg, "neg", 32768},
    {ElementwiseOp::abs, "abs", 32768},
    {ElementwiseOp::square, "square", 32768},
    {ElementwiseOp::relu, "relu", 32768},
    {ElementwiseOp::leaky_relu, "leaky_relu", 32768},
    {ElementwiseOp::gelu, "gelu", 32768},
    {ElementwiseOp::gelu_tanh, "gelu_tanh", 32768},
    {ElementwiseOp::silu, "silu", 32768},
    {ElementwiseOp::elu, "elu", 32768},
    {ElementwiseOp::softplus, "softplus", 32768},
    {ElementwiseOp::sqrt, "sqrt", 32768},
    {ElementwiseOp::exp, "exp", 32768},
    {ElementwiseOp::sigmoid, "sigmoid", 32768},
    {ElementwiseOp::log, "log", 32768},
    {ElementwiseOp::rsqrt, "rsqrt", 32768},
    {ElementwiseOp::tanh, "tanh", 32768},
    {ElementwiseOp::sin, "sin", 32768},
    {ElementwiseOp::cos, "cos", 32768},
    {ElementwiseOp::reciprocal, "reciprocal", 32768},
    {ElementwiseOp::clamp, "clamp", 32768},
    {ElementwiseOp::cast, "cast", 32768}};

inline constexpr std::size_t elementwise_op_count = static_cast<std::size_t>(ElementwiseOp::count);
static_assert(sizeof(elementwise_parallel_policies) / sizeof(ElementwiseParallelPolicy) ==
              elementwise_op_count);
static_assert([] {
    for (std::size_t i = 0; i < elementwise_op_count; ++i) {
        if (static_cast<std::size_t>(elementwise_parallel_policies[i].op) != i ||
            elementwise_parallel_policies[i].default_cutoff < 0) {
            return false;
        }
    }
    return true;
}(), "cutoff table must have one nonnegative entry for every operation, in enum order");

// Parallel iff n > cutoff. Zero enables the parallel branch for every nonempty input;
// INT64_MAX keeps every input on the serial branch. Without OpenMP all kernels remain serial.
std::int64_t elementwise_parallel_cutoff(ElementwiseOp op);

// Internal measurement hook, shared by the library and benchmarks. Overrides affect only this
// operation in the current process. Negative cutoffs and invalid operation IDs throw.
// Each kernel snapshots its cutoff once before entering its OpenMP region. Atomic updates avoid
// races, but configure before timing so concurrent changes do not mix scheduling policies.
void set_elementwise_parallel_cutoff(ElementwiseOp op, std::int64_t cutoff);

} // namespace mensurflow::cpu
