// Dispatcher: maps (op, device type, dtype) to the kernel that runs it. Each backend registers
// the kernels it has; ops never mention a backend. A missing kernel is an error, never a silent
// switch to another device, because the data lives in that device's memory.
#pragma once

#include "backends/backend.hpp"
#include "mensurflow/dtype.hpp"
#include "mensurflow/execution.hpp"
#include "mensurflow/tensor.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
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
    EmbeddingBackward,
    Conv2d,
    Conv2dBackwardData,
    Conv2dBackwardWeight,
    Conv2dBackwardBias,
    MaxPool2d,
    MaxPool2dBackward,
    AvgPool2d,
    AvgPool2dBackward,
    Argmin,
    Var,
    VarBackward,
    Std,
    StdBackward,
    Softmax,
    LogSoftmax,
    SoftmaxBackward,
    LogSoftmaxBackward,
    CrossEntropy,
    CrossEntropyBackward,
    MseLoss,
    MseLossBackward,
    BceWithLogits,
    BceWithLogitsBackward,
    Sgd,
    Adam,
    AdamW,
    LayerNorm,
    LayerNormBackward,
    RmsNorm,
    RmsNormBackward,
    BatchNorm,
    BatchNormBackward
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

// The settings the auto-tuner chose for one call of a tunable kernel (see core/tuning.hpp): the
// values of the parameters the kernel registered, in that order. size == 0 means none were chosen
// and the kernel uses its own defaults.
struct TuningConfig {
    static constexpr int capacity = 8;
    std::array<std::int64_t, capacity> values{};
    int size = 0;
};

// Everything a kernel needs. Ops validate shapes and dtypes before building this, so kernels
// may assume the inputs agree. All tensors are on the same device.
struct OpArgs {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    // Op-specific scalar: multiplier, clamp lower bound, activation slope/alpha, or softplus beta.
    float alpha = 1.0f;
    float beta = 1.0f; // gemv beta, clamp upper bound, or softplus threshold.
    // Optimizers (alpha is the learning rate) and normalization (epsilon, momentum).
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float momentum = 0.0f; // SGD momentum, or the batch-norm running-statistics momentum.
    float epsilon = 1e-5f;
    float weight_decay = 0.0f;
    std::int64_t step = 1; // Adam / AdamW: the 1-based step count, for bias correction.
    int axis = -1;         // Reduction axis; unused by other operations.
    bool keepdim = false;
    bool transpose = false;
    bool approximate = false; // GELU: tanh approximation instead of the Gaussian CDF.
    // pad: elements added before and after each dimension, as (before, after) pairs in dimension
    // order. conv2d and pooling: the zero padding {pad_h, pad_w} on each side. Unused by other
    // operations.
    std::vector<std::int64_t> padding{};
    // conv2d and pooling: the window step {stride_h, stride_w}.
    std::vector<std::int64_t> stride{};
    // Pooling: the window size {kernel_h, kernel_w} (conv2d takes it from the weight's shape).
    std::vector<std::int64_t> kernel_size{};
    // var and std: the divisor is (count - correction); 0 is the population variance.
    std::int64_t correction = 1;
    // Counter-based random generation: offset counts output elements, not threads.
    std::uint64_t seed = 0;
    std::uint64_t offset = 0;
    bool training = true;  // Dropout: eval mode is the identity.
    int reduction = 0;     // Losses: 0 none, 1 sum, 2 mean (the order of mensurflow::Reduction).
    TuningConfig tuning{}; // Set by dispatch() from the active tuner; only tunable kernels read it.
};

using KernelFn = void (*)(const OpArgs& args, Stream& stream);

// One argument of the function the auto-tuner minimizes: a parameter of the kernel and the values
// it may take, in order along the axis (the binary search assumes the time falls and then rises
// along it, so neighbours must be neighbours in effect, not just in number).
struct SearchAxis {
    int param = 0; // index into TunableKernel::params
    std::vector<std::int64_t> values;
};

// A region of a kernel's settings where the time is a function of a few ordered arguments. The
// parameters not on an axis keep their value from `start`. A kernel with several variants has one
// region per variant (the Metal matmul: tiled, simdgroup, MPS), since a variant is a category, not a
// point on an axis.
struct SearchRegion {
    std::string name; // e.g. "simdgroup"
    TuningConfig
        start; // where a local search starts; also the value of every parameter not on an axis
    std::vector<SearchAxis> axes;
};

// What a backend registers for a kernel whose parameters (tile sizes and the like) the auto-tuner
// may choose. A kernel without a TunableKernel always runs with its own settings.
//
// For one call, tuning is the minimization of a function of discrete arguments:
//   time(arguments; key)   with the key (m, n, k, ...) fixed by the call,
// over the points of `space(args)` that `allowed` accepts.
struct TunableKernel {
    std::string name; // the name in the results file, e.g. "matmul"
    int version = 1;  // raise it when the kernel changes, so results measured before are ignored
    std::vector<std::string> key_names; // what a call is keyed on, e.g. {"m", "n", "k"}
    std::vector<std::int64_t> (*key)(const OpArgs& args) = nullptr; // values of key_names
    std::vector<std::string> params;      // the names of TuningConfig's values, in order
    TuningConfig (*defaults)() = nullptr; // what the kernel uses when no tuning is given
    // The search space for this call; empty when there is nothing to tune for it (for example a
    // kernel variant the call will not use).
    std::vector<SearchRegion> (*space)(const OpArgs& args) = nullptr;
    // Whether a point of the space is worth running for this call: valid for the kernel, and not
    // known to be useless (for example a tile far larger than the matrix).
    bool (*allowed)(const OpArgs& args, const TuningConfig& config) = nullptr;
};

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

    // Declares the kernel registered under `key` tunable. Throws logic_error if it already is, or
    // invalid_argument if a function is missing or it has more parameters than TuningConfig holds.
    void register_tunable(KernelKey key, TunableKernel tunable);

    // nullptr when the kernel under `key` is not tunable.
    const TunableKernel* tunable(KernelKey key) const;

    const std::map<KernelKey, TunableKernel>& tunables() const { return tunables_; }

  private:
    std::map<KernelKey, KernelFn> kernels_;
    std::map<KernelKey, TunableKernel> tunables_;
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
