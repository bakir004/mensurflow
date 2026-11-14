#include "mensurflow/ops/normalization.hpp"

#include "core/dispatcher.hpp"

#include <cmath>
#include <stdexcept>

namespace mensurflow {
namespace {
void validate(const Tensor& x, const Shape& normalized, const char* op) {
    if (!x.defined() || x.dtype() != DType::F32 || x.device().type != DeviceType::CPU ||
        normalized.empty() || normalized.size() > x.shape().size())
        throw std::invalid_argument(std::string(op) + ": invalid input or normalized shape");
    for (std::size_t i = 0; i < normalized.size(); ++i)
        if (normalized[i] != x.shape()[x.shape().size() - normalized.size() + i])
            throw std::invalid_argument(std::string(op) + ": normalized shape does not match input");
}
Tensor default_weight(const Shape& s, Device d) { return Tensor::ones(s, DType::F32, d); }
Tensor default_bias(const Shape& s, Device d) { return Tensor::zeros(s, DType::F32, d); }
void check_param(const Tensor* t, const Shape& s, const char* op) {
    if (t != nullptr && (!t->defined() || t->dtype() != DType::F32 || t->shape() != s ||
                         t->device().type != DeviceType::CPU))
        throw std::invalid_argument(std::string(op) + ": parameter has wrong shape or dtype");
}
std::int64_t product(const Shape& s) { std::int64_t n = 1; for (auto v : s) n *= v; return n; }
}

Tensor layer_norm(const Tensor& x, const Shape& shape, const Tensor* weight, const Tensor* bias,
                  float eps, Execution execution) {
    validate(x, shape, "layer_norm"); check_param(weight, shape, "layer_norm"); check_param(bias, shape, "layer_norm");
    if (!(eps > 0 && std::isfinite(eps))) throw std::invalid_argument("layer_norm: invalid epsilon");
    Tensor w = weight ? weight->contiguous() : default_weight(shape, x.device());
    Tensor b = bias ? bias->contiguous() : default_bias(shape, x.device());
    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    detail::OpArgs args{{x.contiguous(), w, b}, {out}};
    args.axis = static_cast<int>(shape.size()); args.epsilon = eps;
    detail::dispatch(detail::OpId::LayerNorm, args, execution);
    return out;
}

NormalizationBackward layer_norm_backward(const Tensor& x, const Tensor& grad, const Shape& shape,
                                          const Tensor* weight, float eps, Execution execution) {
    validate(x, shape, "layer_norm_backward"); check_param(weight, shape, "layer_norm_backward");
    if (grad.shape() != x.shape() || grad.dtype() != DType::F32) throw std::invalid_argument("layer_norm_backward: gradient mismatch");
    Tensor w = weight ? weight->contiguous() : default_weight(shape, x.device());
    NormalizationBackward r{Tensor::empty(x.shape()), Tensor::empty(shape), Tensor::empty(shape)};
    detail::OpArgs args{{x.contiguous(), grad.contiguous(), w}, {r.input, r.weight, r.bias}};
    args.axis = static_cast<int>(shape.size()); args.epsilon = eps;
    detail::dispatch(detail::OpId::LayerNormBackward, args, execution); return r;
}

Tensor rms_norm(const Tensor& x, const Shape& shape, const Tensor* weight, float eps, Execution execution) {
    validate(x, shape, "rms_norm"); check_param(weight, shape, "rms_norm");
    Tensor w = weight ? weight->contiguous() : default_weight(shape, x.device());
    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    detail::OpArgs args{{x.contiguous(), w}, {out}}; args.axis = static_cast<int>(shape.size()); args.epsilon = eps;
    detail::dispatch(detail::OpId::RmsNorm, args, execution); return out;
}

NormalizationBackward rms_norm_backward(const Tensor& x, const Tensor& grad, const Shape& shape,
                                        const Tensor* weight, float eps, Execution execution) {
    validate(x, shape, "rms_norm_backward"); check_param(weight, shape, "rms_norm_backward");
    if (grad.shape() != x.shape() || grad.dtype() != DType::F32) throw std::invalid_argument("rms_norm_backward: gradient mismatch");
    Tensor w = weight ? weight->contiguous() : default_weight(shape, x.device());
    NormalizationBackward r{Tensor::empty(x.shape()), Tensor::empty(shape), Tensor()};
    detail::OpArgs args{{x.contiguous(), grad.contiguous(), w}, {r.input, r.weight}};
    args.axis = static_cast<int>(shape.size()); args.epsilon = eps;
    detail::dispatch(detail::OpId::RmsNormBackward, args, execution); return r;
}

Tensor batch_norm(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor* weight,
                  const Tensor* bias, bool training, float momentum, float eps, Execution execution) {
    if (!x.defined() || x.dtype() != DType::F32 || x.dim() < 2 || mean.shape() != Shape{x.size(1)} ||
        variance.shape() != mean.shape() || mean.dtype() != DType::F32 || variance.dtype() != DType::F32)
        throw std::invalid_argument("batch_norm: expected F32 [N,C,...] and C running statistics");
    Shape cs{ x.size(1) }; check_param(weight, cs, "batch_norm"); check_param(bias, cs, "batch_norm");
    Tensor w = weight ? weight->contiguous() : default_weight(cs, x.device());
    Tensor b = bias ? bias->contiguous() : default_bias(cs, x.device());
    Tensor out = Tensor::empty(x.shape()); Tensor stats = Tensor::empty({2 * x.size(1)});
    detail::OpArgs args{{x.contiguous(), w, b, mean, variance}, {out, stats}};
    args.training = training; args.gamma = momentum; args.epsilon = eps;
    detail::dispatch(detail::OpId::BatchNorm, args, execution); return out;
}

NormalizationBackward batch_norm_backward(const Tensor& x, const Tensor& grad, const Tensor& mean,
                                          const Tensor& variance, const Tensor* weight, bool training,
                                          float momentum, float eps, Execution execution) {
    if (x.shape() != grad.shape() || x.dim() < 2 || mean.shape() != Shape{x.size(1)} ||
        variance.shape() != mean.shape()) throw std::invalid_argument("batch_norm_backward: shape mismatch");
    Shape cs{x.size(1)}; check_param(weight, cs, "batch_norm_backward");
    Tensor w = weight ? weight->contiguous() : default_weight(cs, x.device());
    Tensor stats = Tensor::empty({2 * x.size(1)});
    std::vector<float> sv(static_cast<std::size_t>(2 * x.size(1)));
    if (training) {
        const auto xv = x.to_vector<float>();
        const std::int64_t n = x.size(0), c = x.size(1), spatial = x.numel() / (n * c);
        for (std::int64_t ch = 0; ch < c; ++ch) {
            double m = 0.0, v = 0.0;
            for (std::int64_t bn = 0; bn < n; ++bn)
                for (std::int64_t s = 0; s < spatial; ++s) m += xv[static_cast<std::size_t>((bn * c + ch) * spatial + s)];
            m /= n * spatial;
            for (std::int64_t bn = 0; bn < n; ++bn)
                for (std::int64_t s = 0; s < spatial; ++s) {
                    const double d = xv[static_cast<std::size_t>((bn * c + ch) * spatial + s)] - m;
                    v += d * d;
                }
            sv[static_cast<std::size_t>(ch)] = static_cast<float>(m);
            sv[static_cast<std::size_t>(c + ch)] = static_cast<float>(v / (n * spatial));
        }
    } else {
        auto mv = mean.to_vector<float>(), vv = variance.to_vector<float>();
        for (std::int64_t i = 0; i < x.size(1); ++i) {
            sv[static_cast<std::size_t>(i)] = mv[static_cast<std::size_t>(i)];
            sv[static_cast<std::size_t>(x.size(1)+i)] = vv[static_cast<std::size_t>(i)];
        }
    }
    stats.copy_from(Tensor::from_vector(sv, stats.shape(), x.device()));
    NormalizationBackward r{Tensor::empty(x.shape()), Tensor::empty(cs), Tensor::empty(cs)};
    detail::OpArgs args{{x.contiguous(), grad.contiguous(), w, stats}, {r.input, r.weight, r.bias}};
    args.training = training; args.gamma = momentum; args.epsilon = eps;
    detail::dispatch(detail::OpId::BatchNormBackward, args, execution); return r;
}
} // namespace mensurflow
