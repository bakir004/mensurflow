// Kernel table and the entry point every op uses to reach a backend.
#include "core/dispatcher.hpp"

#include "core/registry.hpp"

namespace mensurflow::detail {

const char* op_name(OpId op) {
    switch (op) {
    case OpId::Dot:
        return "dot";
    case OpId::Add:
        return "add";
    case OpId::Sub:
        return "sub";
    case OpId::Mul:
        return "mul";
    case OpId::Div:
        return "div";
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

void dispatch(OpId op, const OpArgs& args) {
    if (args.inputs.empty()) {
        throw std::invalid_argument(std::string(op_name(op)) + ": no input tensors");
    }
    const Device device = args.inputs[0].device();
    for (const auto* group : {&args.inputs, &args.outputs}) {
        for (const Tensor& t : *group) {
            if (t.device() != device) {
                throw std::invalid_argument(std::string(op_name(op)) + ": tensors are on " +
                                            to_string(device) + " and " + to_string(t.device()) +
                                            "; move one with Tensor::to()");
            }
        }
    }
    Registry& registry = Registry::instance();
    Backend& backend = registry.get(device);
    const KernelFn kernel = registry.dispatcher().lookup({op, device.type, args.inputs[0].dtype()});
    kernel(args, backend.stream(device.index));
}

} // namespace mensurflow::detail
