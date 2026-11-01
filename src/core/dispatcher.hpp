// Dispatcher: maps (op, device type, dtype) to the kernel that runs it. Each backend registers
// the kernels it has; ops never mention a backend. A missing kernel is an error, never a silent
// switch to another device, because the data lives in that device's memory.
#pragma once

#include "backends/backend.hpp"
#include "mensurflow/dtype.hpp"
#include "mensurflow/tensor.hpp"

#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace mensurflow::detail {

enum class OpId {
    Dot,
    Add,
    Sub,
    Mul,
    Div,
    Matvec,
    Matmul,
    Neg,
    Abs,
    Square,
    Relu,
    Sqrt,
    Exp,
    Sigmoid,
    Axpy,
    BiasAdd,
    Sum,
    Mean,
    Max,
    Softmax
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
    float alpha = 1.0f; // Scalar multiplier for axpy; unused by other operations.
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

// Runs `op` on the device its inputs live on: picks that device's backend, looks up the kernel for
// the first input's dtype, and calls it on the device's stream. The kernel may return before the
// device finishes; reading an output through Tensor blocks until it is ready.
// Throws invalid_argument if the inputs are on different devices, runtime_error if the device is
// unavailable, and NotImplemented if no kernel exists.
void dispatch(OpId op, const OpArgs& args);

} // namespace mensurflow::detail
