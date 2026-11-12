// Validation, dispatch and gradient recording for seeded RNG and dropout.
#include "mensurflow/ops/random.hpp"

#include "core/autograd.hpp"
#include "core/dispatcher.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mensurflow {

namespace {

void require_probability(float probability, const char* op) {
    if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f) {
        throw std::invalid_argument(std::string(op) + ": probability must be finite and in [0, 1]");
    }
}

void require_stream(std::int64_t count, std::uint64_t offset, const char* op) {
    if (count > 0 && static_cast<std::uint64_t>(count - 1) >
                         std::numeric_limits<std::uint64_t>::max() - offset) {
        throw std::invalid_argument(std::string(op) + ": sample offset overflows uint64_t");
    }
}

Tensor random_op(detail::OpId op, Shape shape, float alpha, float beta, std::uint64_t seed,
                 std::uint64_t offset, Execution execution) {
    validate_shape(shape);
    require_stream(numel(shape), offset, detail::op_name(op));
    Tensor out = Tensor::empty(std::move(shape), DType::F32);
    detail::OpArgs args{{}, {out}};
    args.alpha = alpha;
    args.beta = beta;
    args.seed = seed;
    args.offset = offset;
    detail::dispatch(op, args, execution);
    return out;
}

} // namespace

Tensor uniform(Shape shape, float low, float high, std::uint64_t seed, std::uint64_t offset,
               Execution execution) {
    if (!std::isfinite(low) || !std::isfinite(high) || low > high) {
        throw std::invalid_argument("uniform: bounds must be finite and low <= high");
    }
    return random_op(detail::OpId::Uniform, std::move(shape), low, high, seed, offset, execution);
}

Tensor normal(Shape shape, float mean, float stddev, std::uint64_t seed, std::uint64_t offset,
              Execution execution) {
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev < 0.0f) {
        throw std::invalid_argument("normal: mean and stddev must be finite, stddev >= 0");
    }
    return random_op(detail::OpId::Normal, std::move(shape), mean, stddev, seed, offset, execution);
}

Tensor bernoulli(Shape shape, float probability, std::uint64_t seed, std::uint64_t offset,
                 Execution execution) {
    require_probability(probability, "bernoulli");
    return random_op(detail::OpId::Bernoulli, std::move(shape), probability, 0.0f, seed, offset,
                     execution);
}

DropoutResult dropout(const Tensor& x, float probability, bool training, std::uint64_t seed,
                      std::uint64_t offset, Execution execution) {
    if (!x.defined()) {
        throw std::invalid_argument("dropout: undefined tensor");
    }
    require_probability(probability, "dropout");
    require_stream(x.numel(), offset, "dropout");
    Tensor out = Tensor::empty(x.shape(), x.dtype(), x.device());
    const Tensor mask = Tensor::empty(x.shape(), DType::Bool, x.device());
    detail::OpArgs args{{x.contiguous()}, {out, mask}};
    args.alpha = probability;
    args.training = training;
    args.seed = seed;
    args.offset = offset;
    detail::dispatch(detail::OpId::Dropout, args, execution);
    if (x.requires_grad() && detail::autograd::enabled()) {
        Tensor saved_mask = Tensor::empty(mask.shape(), mask.dtype(), mask.device());
        saved_mask.copy_from(mask);
        detail::AutogradAccess::record(
            out, {x}, [x, saved_mask, probability, training, execution](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                detail::autograd::propagate(
                    x, dropout_backward(gradient, saved_mask, probability, training, execution));
            });
    }
    return {out, mask};
}

Tensor dropout_backward(const Tensor& gradient, const Tensor& mask, float probability,
                        bool training, Execution execution) {
    if (!gradient.defined() || !mask.defined()) {
        throw std::invalid_argument("dropout_backward: undefined tensor");
    }
    require_probability(probability, "dropout_backward");
    if (mask.dtype() != DType::Bool || gradient.shape() != mask.shape() ||
        gradient.device() != mask.device()) {
        throw std::invalid_argument(
            "dropout_backward: mask must be Bool with gradient shape/device");
    }
    Tensor out = Tensor::empty(gradient.shape(), gradient.dtype(), gradient.device());
    detail::OpArgs args{{gradient.contiguous(), mask.contiguous()}, {out}};
    args.alpha = probability;
    args.training = training;
    detail::dispatch(detail::OpId::DropoutBackward, args, execution);
    return out;
}

} // namespace mensurflow
