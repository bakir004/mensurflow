#include "mensurflow/ops/loss.hpp"

#include "core/autograd.hpp"
#include "mensurflow/ops/elementwise.hpp"
#include "mensurflow/ops/reduction.hpp"
#include "mensurflow/ops/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mensurflow {

Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    if (!prediction.defined() || !target.defined()) {
        throw std::invalid_argument("mse_loss: undefined tensor");
    }
    if (prediction.shape() != target.shape() || prediction.dtype() != DType::F32 ||
        target.dtype() != DType::F32 || prediction.device() != target.device()) {
        throw std::invalid_argument("mse_loss: inputs must match as CPU f32 tensors");
    }
    const Tensor difference = sub(prediction, target);
    const Tensor loss = mean(square(difference));
    return loss;
}

Tensor log_softmax(const Tensor& x, int axis) {
    if (!x.defined() || x.dtype() != DType::F32 || x.device().type != DeviceType::CPU) {
        throw std::invalid_argument("log_softmax: expected a CPU f32 tensor");
    }
    const int normalized = axis < 0 ? axis + x.dim() : axis;
    if (x.dim() == 0 || normalized < 0 || normalized >= x.dim()) {
        throw std::invalid_argument("log_softmax: axis out of range");
    }
    const Tensor packed = x.contiguous();
    Tensor out = Tensor::empty(x.shape(), DType::F32, x.device());
    const auto values = packed.to_vector<float>();
    auto result = std::vector<float>(values.size());
    const std::int64_t width = x.size(normalized);
    const std::int64_t rows = x.numel() / width;
    if (normalized != x.dim() - 1) {
        return log_softmax(x.transpose(normalized, x.dim() - 1), -1)
            .transpose(normalized, x.dim() - 1)
            .contiguous();
    }
    for (std::int64_t row = 0; row < rows; ++row) {
        const auto begin = values.begin() + row * width;
        const auto end = begin + width;
        const float maximum = *std::max_element(begin, end);
        double denominator = 0.0;
        for (auto it = begin; it != end; ++it) {
            denominator += std::exp(static_cast<double>(*it - maximum));
        }
        const float log_denominator = maximum + static_cast<float>(std::log(denominator));
        for (std::int64_t i = 0; i < width; ++i) {
            result[static_cast<std::size_t>(row * width + i)] = values[static_cast<std::size_t>(
                row * width + i)] - log_denominator;
        }
    }
    out = Tensor::from_vector(result, x.shape(), x.device());
    if (x.requires_grad()) {
        detail::AutogradAccess::record(
            out, {x}, [x, out, normalized](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const Tensor probabilities = exp(out);
                const Tensor summed = sum(gradient, normalized, true);
                detail::autograd::propagate(x, sub(gradient, mul(probabilities, summed)));
            });
    }
    return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets, int axis) {
    if (logits.dim() != 2 || targets.dim() != 1 || targets.size(0) != logits.size(0) ||
        targets.dtype() != DType::I64) {
        throw std::invalid_argument("cross_entropy: expected logits [batch, classes] and i64 targets");
    }
    const Tensor logs = log_softmax(logits, axis);
    const auto values = logs.to_vector<float>();
    const auto labels = targets.to_vector<std::int64_t>();
    double total = 0.0;
    for (std::size_t row = 0; row < labels.size(); ++row) {
        if (labels[row] < 0 || labels[row] >= logits.size(1)) {
            throw std::invalid_argument("cross_entropy: target index out of range");
        }
        total -= values[row * static_cast<std::size_t>(logits.size(1)) +
                       static_cast<std::size_t>(labels[row])];
    }
    Tensor out = Tensor::from_vector(
        std::vector<float>{static_cast<float>(total / static_cast<double>(labels.size()))}, {});
    if (logits.requires_grad()) {
        detail::AutogradAccess::record(
            out, {logits}, [logits, targets, labels](const Tensor& gradient) {
                detail::autograd::NoGradGuard guard;
                const Tensor probabilities = softmax(logits, -1);
                auto result = probabilities.to_vector<float>();
                const float scale = gradient.item<float>() /
                                    static_cast<float>(labels.size());
                for (std::size_t row = 0; row < labels.size(); ++row) {
                    result[row * static_cast<std::size_t>(logits.size(1)) +
                           static_cast<std::size_t>(labels[row])] -= 1.0f;
                    for (std::int64_t c = 0; c < logits.size(1); ++c) {
                        result[row * static_cast<std::size_t>(logits.size(1)) +
                               static_cast<std::size_t>(c)] *= scale;
                    }
                }
                detail::autograd::propagate(
                    logits, Tensor::from_vector(result, logits.shape(), logits.device()));
            });
    }
    return out;
}

} // namespace mensurflow
