// Dispatcher: maps (op, device type, dtype) to the kernel that runs it. Each backend registers
// the kernels it has; ops never mention a backend. A missing kernel is an error, never a silent
// switch to another device, because the data lives in that device's memory.
#pragma once

#include "backends/backend.hpp"
#include "mensurflow/dtype.hpp"
#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace mensurflow::detail {

enum class OpId {
    Dot,
    Scal,
    Asum,
    Nrm2,
    Copy,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Matvec,
    Gemv,
    Matmul,
    Neg,
    Abs,
    Square,
    Relu,
    LeakyRelu,
    Gelu,
    Silu,
    Elu,
    Softplus,
    ReluBackward,
    SigmoidBackward,
    TanhBackward,
    GeluBackward,
    SiluBackward,
    Sqrt,
    Exp,
    Sigmoid,
    Log,
    Rsqrt,
    Tanh,
    Sin,
    Cos,
    Reciprocal,
    Axpy,
    Minimum,
    Maximum,
    Clamp,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Where,
    Cast,
    BiasAdd,
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    AxisSum,
    AxisMean,
    AxisMax,
    AxisMin,
    AxisProd,
    Argmax,
    GemmBatched,
    Ger,
    Concat,
    Pad,
    Gather,
    IndexSelect,
    Uniform,
    Normal,
    Bernoulli,
    Dropout,
    DropoutBackward,
    OneHot,
    Embedding,
    EmbeddingBackward
};

const char* op_name(OpId op);

struct KernelKey {
    OpId op;
    DeviceType device;
    DType dtype;
};

inline bool operator<(const KernelKey& a, const KernelKey& b) {
    return std::tie(a.op, a.device, a.dtype) < std::tie(b.op, b.device, b.dtype);
}

// Everything a kernel needs. Ops validate shapes and dtypes before building this, so kernels
// may assume the inputs agree. All tensors are on the same device.
struct OpArgs {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    // Op-specific scalar: multiplier, clamp lower bound, activation slope/alpha, or softplus beta.
    float alpha = 1.0f;
    float beta = 1.0f;  // gemv beta, clamp upper bound, or softplus threshold.
    int axis = -1;      // Reduction axis; unused by other operations.
    bool keepdim = false;
    bool transpose = false;
    bool approximate = false; // GELU: tanh approximation instead of the Gaussian CDF.
    // pad: elements added before and after each dimension, as (before, after) pairs in dimension
    // order. Unused by other operations.
    std::vector<std::int64_t> padding{};
    // Counter-based random generation: offset counts output elements, not threads.
    std::uint64_t seed = 0;
    std::uint64_t offset = 0;
    bool training = true; // Dropout: eval mode is the identity.
};

using KernelFn = void (*)(const OpArgs& args, Stream& stream);

// Thrown when no backend registered a kernel for the requested op, device, and dtype.
class NotImplemented : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class Dispatcher {
  public:
    void register_kernel(KernelKey key, KernelFn kernel);

    // nullptr when nothing is registered for the key.
    KernelFn find(KernelKey key) const;

    // Like find, but throws NotImplemented naming the op, device type, and dtype.
    KernelFn lookup(KernelKey key) const;

  private:
    std::map<KernelKey, KernelFn> kernels_;
};

// The device an op runs on, given the caller's choice and the device its inputs live on:
// CPU -> cpu:0; GPU -> data_device if it is a GPU, else the first available CUDA or Metal
// device (runtime_error if there is none); Auto -> cpu:0 for now. This is the one place a future
// auto-tuner would decide.
Device execution_device(Execution execution, Device data_device);

// Runs `op` where `execution` says (see execution_device). The inputs must all be on one device;
// the outputs must be allocated on that same device. If the op runs elsewhere, the inputs are
// copied there, the kernel writes temporary outputs, and those are copied back into `outputs`.
// The kernel is looked up for the execution device and the first input's dtype (the first
// output's dtype for random factories, which have no inputs). On a GPU it may
// return before the device finishes; reading an output through Tensor blocks until it is ready.
// Throws invalid_argument if the tensors are on different devices, runtime_error if the
// execution device is unavailable, and NotImplemented if it has no kernel for the op and dtype.
void dispatch(OpId op, const OpArgs& args, Execution execution);

} // namespace mensurflow::detail
