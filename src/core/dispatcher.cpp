// Kernel table and the entry point every op uses to reach a backend.
#include "core/dispatcher.hpp"

#include "core/registry.hpp"

namespace mensurflow::detail {

const char* op_name(OpId op) {
    switch (op) {
    case OpId::Dot:
        return "dot";
    case OpId::Scal:
        return "scal";
    case OpId::Asum:
        return "asum";
    case OpId::Nrm2:
        return "nrm2";
    case OpId::Copy:
        return "copy";
    case OpId::Add:
        return "add";
    case OpId::Sub:
        return "sub";
    case OpId::Mul:
        return "mul";
    case OpId::Div:
        return "div";
    case OpId::Pow:
        return "pow";
    case OpId::Matvec:
        return "matvec";
    case OpId::Gemv:
        return "gemv";
    case OpId::Matmul:
        return "matmul";
    case OpId::Neg:
        return "neg";
    case OpId::Abs:
        return "abs";
    case OpId::Square:
        return "square";
    case OpId::Relu:
        return "relu";
    case OpId::Sqrt:
        return "sqrt";
    case OpId::Exp:
        return "exp";
    case OpId::Sigmoid:
        return "sigmoid";
    case OpId::Log:
        return "log";
    case OpId::Rsqrt:
        return "rsqrt";
    case OpId::Tanh:
        return "tanh";
    case OpId::Sin:
        return "sin";
    case OpId::Cos:
        return "cos";
    case OpId::Reciprocal:
        return "reciprocal";
    case OpId::Axpy:
        return "axpy";
    case OpId::Minimum:
        return "minimum";
    case OpId::Maximum:
        return "maximum";
    case OpId::Clamp:
        return "clamp";
    case OpId::BiasAdd:
        return "bias_add";
    case OpId::Sum:
        return "sum";
    case OpId::Mean:
        return "mean";
    case OpId::Max:
        return "max";
    case OpId::Min:
        return "min";
    case OpId::Prod:
        return "prod";
    case OpId::AxisSum:
        return "sum_axis";
    case OpId::AxisMean:
        return "mean_axis";
    case OpId::AxisMax:
        return "max_axis";
    case OpId::AxisMin:
        return "min_axis";
    case OpId::AxisProd:
        return "prod_axis";
    case OpId::Argmax:
        return "argmax";
    }
    throw std::invalid_argument("invalid OpId");
}

void Dispatcher::register_kernel(KernelKey key, KernelFn kernel) {
    if (kernel == nullptr) {
        throw std::invalid_argument("register_kernel: null kernel");
    }
    if (!kernels_.emplace(key, kernel).second) {
        throw std::logic_error(std::string("kernel registered twice: ") + op_name(key.op) + " on " +
                               device_type_name(key.device) + " for " + dtype_name(key.dtype));
    }
}

KernelFn Dispatcher::find(KernelKey key) const {
    const auto it = kernels_.find(key);
    return it == kernels_.end() ? nullptr : it->second;
}

KernelFn Dispatcher::lookup(KernelKey key) const {
    if (KernelFn kernel = find(key)) {
        return kernel;
    }
    throw NotImplemented(std::string(op_name(key.op)) + " is not implemented on " +
                         device_type_name(key.device) + " for " + dtype_name(key.dtype));
}

Device execution_device(Execution execution, Device data_device) {
    switch (execution) {
    case Execution::Auto:
        // Not tuned yet: everything runs on the CPU.
        return {DeviceType::CPU, 0};
    case Execution::CPU:
        return {DeviceType::CPU, 0};
    case Execution::GPU:
        if (data_device.type != DeviceType::CPU) {
            return data_device;
        }
        for (DeviceType type : {DeviceType::CUDA, DeviceType::ROCm, DeviceType::Metal}) {
            if (Registry::instance().find(type) != nullptr) {
                return {type, 0};
            }
        }
        throw std::runtime_error("Execution::GPU requested, but no GPU is available");
    }
    throw std::invalid_argument("invalid Execution value");
}

void dispatch(OpId op, const OpArgs& args, Execution execution) {
    if (args.inputs.empty()) {
        throw std::invalid_argument(std::string(op_name(op)) + ": no input tensors");
    }
    const Device data_device = args.inputs[0].device();
    for (const auto* group : {&args.inputs, &args.outputs}) {
        for (const Tensor& t : *group) {
            if (t.device() != data_device) {
                throw std::invalid_argument(std::string(op_name(op)) + ": tensors are on " +
                                            to_string(data_device) + " and " +
                                            to_string(t.device()) + "; move one with Tensor::to()");
            }
        }
    }

    const Device target = execution_device(execution, data_device);
    Registry& registry = Registry::instance();
    Backend& backend = registry.get(target);
    const KernelFn kernel = registry.dispatcher().lookup({op, target.type, args.inputs[0].dtype()});
    if (target == data_device) {
        kernel(args, backend.stream(target.index));
        return;
    }

    // Run on another device: copy the inputs over, compute into fresh outputs there, copy back.
    OpArgs moved = args;
    for (Tensor& input : moved.inputs) {
        input = input.to(target);
    }
    for (Tensor& output : moved.outputs) {
        output = Tensor::empty(output.shape(), output.dtype(), target);
    }
    kernel(moved, backend.stream(target.index));
    for (std::size_t i = 0; i < args.outputs.size(); ++i) {
        Tensor destination = args.outputs[i]; // shares storage with the caller's output
        destination.copy_from(moved.outputs[i].to(data_device));
    }
}

} // namespace mensurflow::detail
