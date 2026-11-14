// Validation and dispatch for convolution. Kernels receive contiguous NCHW tensors and the stride
// and padding in OpArgs; the conventions are in include/mensurflow/ops/conv.hpp.
#include "mensurflow/ops/conv.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow {

namespace {

std::string pair_string(Size2d pair) {
    return "{" + std::to_string(pair[0]) + ", " + std::to_string(pair[1]) + "}";
}

void require_tensor(const Tensor& t, const std::string& op, const char* name, int rank) {
    if (!t.defined()) {
        throw std::invalid_argument(op + ": " + name + " is an undefined tensor");
    }
    if (t.dim() != rank) {
        throw std::invalid_argument(op + ": " + name + " must be " + std::to_string(rank) +
                                    "-D, got shape " + to_string(t.shape()));
    }
}

void require_same(const Tensor& a, const Tensor& b, const std::string& op) {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument(op + ": dtypes differ (" + dtype_name(a.dtype()) + " and " +
                                    dtype_name(b.dtype()) + ")");
    }
    if (a.device() != b.device()) {
        throw std::invalid_argument(op + ": tensors are on " + to_string(a.device()) + " and " +
                                    to_string(b.device()) + "; move one with Tensor::to()");
    }
}

void require_shape(const Shape& shape, const std::string& op, const char* name) {
    if (shape.size() != 4) {
        throw std::invalid_argument(op + ": " + name + " must have 4 dimensions, got " +
                                    to_string(shape));
    }
    for (std::int64_t size : shape) {
        if (size < 0) {
            throw std::invalid_argument(op + ": " + name + " has a negative size " +
                                        to_string(shape));
        }
    }
}

void require_window(Size2d stride, Size2d padding, const std::string& op) {
    if (stride[0] < 1 || stride[1] < 1) {
        throw std::invalid_argument(op + ": stride must be at least 1, got " + pair_string(stride));
    }
    if (padding[0] < 0 || padding[1] < 0) {
        throw std::invalid_argument(op + ": padding must be non-negative, got " +
                                    pair_string(padding));
    }
}

// floor((size + 2 * pad - kernel) / stride) + 1, or invalid_argument if that is below 1.
std::int64_t output_size(std::int64_t size, std::int64_t kernel, std::int64_t stride,
                         std::int64_t pad, const std::string& op, const char* dimension) {
    if (kernel < 1) {
        throw std::invalid_argument(op + ": kernel " + dimension + " must be at least 1");
    }
    const std::int64_t span = size + 2 * pad - kernel;
    if (span < 0) {
        throw std::invalid_argument(op + ": kernel " + dimension + " " + std::to_string(kernel) +
                                    " is larger than the padded input (" +
                                    std::to_string(size) + " + 2 * " + std::to_string(pad) + ")");
    }
    return span / stride + 1;
}

// The output shape [N, C_out, H_out, W_out] of a convolution of an input_shape input with a
// weight_shape weight; checks that the channels agree.
Shape conv_output_shape(const Shape& input_shape, const Shape& weight_shape, Size2d stride,
                        Size2d padding, const std::string& op) {
    require_window(stride, padding, op);
    if (weight_shape[1] != input_shape[1]) {
        throw std::invalid_argument(op + ": input has " + std::to_string(input_shape[1]) +
                                    " channels, the weight expects " +
                                    std::to_string(weight_shape[1]) + " (groups must be 1)");
    }
    return {input_shape[0], weight_shape[0],
            output_size(input_shape[2], weight_shape[2], stride[0], padding[0], op, "height"),
            output_size(input_shape[3], weight_shape[3], stride[1], padding[1], op, "width")};
}

void require_grad_shape(const Tensor& grad_output, const Shape& expected, const std::string& op) {
    if (grad_output.shape() != expected) {
        throw std::invalid_argument(op + ": grad_output has shape " +
                                    to_string(grad_output.shape()) + ", expected " +
                                    to_string(expected));
    }
}

detail::OpArgs window_args(std::vector<Tensor> inputs, Tensor out, Size2d stride,
                           Size2d padding) {
    detail::OpArgs args{std::move(inputs), {std::move(out)}};
    args.stride = {stride[0], stride[1]};
    args.padding = {padding[0], padding[1]};
    return args;
}

// A pooling window with the default stride resolved.
struct PoolWindow {
    Size2d kernel, stride, padding;
};

PoolWindow pool_window(Size2d kernel, Size2d stride, Size2d padding, const std::string& op) {
    if (kernel[0] < 1 || kernel[1] < 1) {
        throw std::invalid_argument(op + ": kernel_size must be at least 1, got " +
                                    pair_string(kernel));
    }
    if (stride[0] == 0 && stride[1] == 0) {
        stride = kernel;
    }
    require_window(stride, padding, op);
    if (padding[0] > kernel[0] / 2 || padding[1] > kernel[1] / 2) {
        throw std::invalid_argument(op + ": padding " + pair_string(padding) +
                                    " must be at most half of kernel_size " + pair_string(kernel));
    }
    return {kernel, stride, padding};
}

// [N, C, H_out, W_out] for an input of shape x [N, C, H, W]; H and W must be at least 1.
Shape pool_output_shape(const Shape& x, const PoolWindow& window, const std::string& op) {
    if (x[2] < 1 || x[3] < 1) {
        throw std::invalid_argument(op + ": input height and width must be at least 1, got " +
                                    to_string(x));
    }
    return {x[0], x[1],
            output_size(x[2], window.kernel[0], window.stride[0], window.padding[0], op, "height"),
            output_size(x[3], window.kernel[1], window.stride[1], window.padding[1], op, "width")};
}

detail::OpArgs pool_args(std::vector<Tensor> inputs, std::vector<Tensor> outputs,
                         const PoolWindow& window) {
    detail::OpArgs args{std::move(inputs), std::move(outputs)};
    args.kernel_size = {window.kernel[0], window.kernel[1]};
    args.stride = {window.stride[0], window.stride[1]};
    args.padding = {window.padding[0], window.padding[1]};
    return args;
}

} // namespace

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias, Size2d stride,
              Size2d padding, Size2d dilation, std::int64_t groups, Execution execution) {
    const std::string op = "conv2d";
    require_tensor(x, op, "x", 4);
    require_tensor(weight, op, "weight", 4);
    require_same(x, weight, op);
    if (bias != nullptr) {
        require_tensor(*bias, op, "bias", 1);
        require_same(x, *bias, op);
        if (bias->size(0) != weight.size(0)) {
            throw std::invalid_argument("conv2d: bias has " + std::to_string(bias->size(0)) +
                                        " elements, expected C_out = " +
                                        std::to_string(weight.size(0)));
        }
    }
    if (dilation[0] != 1 || dilation[1] != 1) {
        throw std::invalid_argument("conv2d: only dilation {1, 1} is supported, got " +
                                    pair_string(dilation));
    }
    if (groups != 1) {
        throw std::invalid_argument("conv2d: only groups = 1 is supported, got " +
                                    std::to_string(groups));
    }
    const Shape out_shape = conv_output_shape(x.shape(), weight.shape(), stride, padding, op);
    Tensor out = Tensor::empty(out_shape, x.dtype(), x.device());
    std::vector<Tensor> inputs{x.contiguous(), weight.contiguous()};
    if (bias != nullptr) {
        inputs.push_back(bias->contiguous());
    }
    detail::dispatch(detail::OpId::Conv2d, window_args(std::move(inputs), out, stride, padding),
                     execution);

    const bool bias_grad = bias != nullptr && bias->requires_grad();
    if (x.requires_grad() || weight.requires_grad() || bias_grad) {
        std::vector<Tensor> parents{x, weight};
        const Tensor saved_bias = bias != nullptr ? *bias : Tensor();
        if (bias != nullptr) {
            parents.push_back(*bias);
        }
        detail::AutogradAccess::record(
            out, std::move(parents),
            [x, weight, saved_bias, stride, padding, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                if (x.requires_grad()) {
                    detail::autograd::propagate(
                        x, conv2d_backward_data(gradient, weight, x.shape(), stride, padding,
                                                execution));
                }
                if (weight.requires_grad()) {
                    detail::autograd::propagate(
                        weight, conv2d_backward_weight(gradient, x, weight.shape(), stride,
                                                       padding, execution));
                }
                if (saved_bias.defined() && saved_bias.requires_grad()) {
                    detail::autograd::propagate(saved_bias,
                                                conv2d_backward_bias(gradient, execution));
                }
            });
    }
    return out;
}

Tensor conv2d_backward_data(const Tensor& grad_output, const Tensor& weight,
                            const Shape& input_shape, Size2d stride, Size2d padding,
                            Execution execution) {
    const std::string op = "conv2d_backward_data";
    require_tensor(grad_output, op, "grad_output", 4);
    require_tensor(weight, op, "weight", 4);
    require_same(grad_output, weight, op);
    require_shape(input_shape, op, "input_shape");
    require_grad_shape(grad_output,
                       conv_output_shape(input_shape, weight.shape(), stride, padding, op), op);
    Tensor out = Tensor::empty(input_shape, grad_output.dtype(), grad_output.device());
    detail::dispatch(detail::OpId::Conv2dBackwardData,
                     window_args({grad_output.contiguous(), weight.contiguous()}, out, stride,
                                 padding),
                     execution);
    return out;
}

Tensor conv2d_backward_weight(const Tensor& grad_output, const Tensor& x,
                              const Shape& weight_shape, Size2d stride, Size2d padding,
                              Execution execution) {
    const std::string op = "conv2d_backward_weight";
    require_tensor(grad_output, op, "grad_output", 4);
    require_tensor(x, op, "x", 4);
    require_same(grad_output, x, op);
    require_shape(weight_shape, op, "weight_shape");
    require_grad_shape(grad_output, conv_output_shape(x.shape(), weight_shape, stride, padding, op),
                       op);
    Tensor out = Tensor::empty(weight_shape, grad_output.dtype(), grad_output.device());
    detail::dispatch(detail::OpId::Conv2dBackwardWeight,
                     window_args({grad_output.contiguous(), x.contiguous()}, out, stride, padding),
                     execution);
    return out;
}

Tensor conv2d_backward_bias(const Tensor& grad_output, Execution execution) {
    require_tensor(grad_output, "conv2d_backward_bias", "grad_output", 4);
    Tensor out = Tensor::empty({grad_output.size(1)}, grad_output.dtype(), grad_output.device());
    detail::dispatch(detail::OpId::Conv2dBackwardBias, {{grad_output.contiguous()}, {out}},
                     execution);
    return out;
}

MaxPool2dResult max_pool2d(const Tensor& x, Size2d kernel_size, Size2d stride, Size2d padding,
                           Execution execution) {
    const std::string op = "max_pool2d";
    require_tensor(x, op, "x", 4);
    const PoolWindow window = pool_window(kernel_size, stride, padding, op);
    const Shape out_shape = pool_output_shape(x.shape(), window, op);
    Tensor out = Tensor::empty(out_shape, x.dtype(), x.device());
    Tensor indices = Tensor::empty(out_shape, DType::I64, x.device());
    detail::dispatch(detail::OpId::MaxPool2d, pool_args({x.contiguous()}, {out, indices}, window),
                     execution);
    if (x.requires_grad() && detail::autograd::enabled()) {
        // A private copy: the caller may change the returned indices before backward runs.
        const Tensor saved = Tensor::from_vector(indices.to_vector<std::int64_t>(), out_shape,
                                                 indices.device());
        detail::AutogradAccess::record(out, {x}, [x, saved, execution](const Tensor& gradient) {
            detail::autograd::NoGradGuard guard;
            detail::autograd::propagate(x,
                                        max_pool2d_backward(gradient, saved, x.shape(), execution));
        });
    }
    return {out, indices};
}

Tensor max_pool2d_backward(const Tensor& grad_output, const Tensor& indices,
                           const Shape& input_shape, Execution execution) {
    const std::string op = "max_pool2d_backward";
    require_tensor(grad_output, op, "grad_output", 4);
    require_tensor(indices, op, "indices", 4);
    if (indices.dtype() != DType::I64) {
        throw std::invalid_argument(op + ": indices must be I64, got " +
                                    dtype_name(indices.dtype()));
    }
    if (indices.device() != grad_output.device()) {
        throw std::invalid_argument(op + ": grad_output and indices are on different devices");
    }
    if (indices.shape() != grad_output.shape()) {
        throw std::invalid_argument(op + ": indices have shape " + to_string(indices.shape()) +
                                    ", grad_output " + to_string(grad_output.shape()));
    }
    require_shape(input_shape, op, "input_shape");
    if (input_shape[0] != grad_output.size(0) || input_shape[1] != grad_output.size(1)) {
        throw std::invalid_argument(op + ": input_shape " + to_string(input_shape) +
                                    " does not match grad_output " +
                                    to_string(grad_output.shape()) + " in N and C");
    }
    // Checked here: the kernel writes where the indices point.
    const std::int64_t plane = input_shape[2] * input_shape[3];
    for (const std::int64_t index : indices.to_vector<std::int64_t>()) {
        if (index < 0 || index >= plane) {
            throw std::invalid_argument(op + ": index " + std::to_string(index) +
                                        " is outside the [H, W] plane of " +
                                        std::to_string(plane) + " cells");
        }
    }
    Tensor out = Tensor::empty(input_shape, grad_output.dtype(), grad_output.device());
    detail::dispatch(detail::OpId::MaxPool2dBackward,
                     {{grad_output.contiguous(), indices.contiguous()}, {out}}, execution);
    return out;
}

} // namespace mensurflow
