// Scalar references for softmax, the losses and attention: the textbook formulas in double
// precision, one line or element at a time, with no parallelism and no shared code with the kernels.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace reference {

// Input seen as [outer, length, inner]; each line of `length` elements is `inner` apart.
inline std::vector<float> softmax(const std::vector<float>& x, std::int64_t outer,
                                  std::int64_t length, std::int64_t inner, bool log) {
    std::vector<float> out(x.size());
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
            auto at = [&](std::int64_t j) {
                return static_cast<std::size_t>((o * length + j) * inner + i);
            };
            double maximum = -std::numeric_limits<double>::infinity();
            bool nan = false;
            for (std::int64_t j = 0; j < length; ++j) {
                nan = nan || std::isnan(x[at(j)]);
                maximum = std::max(maximum, static_cast<double>(x[at(j)]));
            }
            double sum = 0.0;
            for (std::int64_t j = 0; j < length; ++j) {
                sum += std::exp(static_cast<double>(x[at(j)]) - maximum);
            }
            for (std::int64_t j = 0; j < length; ++j) {
                double value;
                if (nan) {
                    value = std::numeric_limits<double>::quiet_NaN();
                } else if (maximum == -std::numeric_limits<double>::infinity()) {
                    value = log ? -std::numeric_limits<double>::infinity() : 0.0;
                } else if (log) {
                    value = static_cast<double>(x[at(j)]) - maximum - std::log(sum);
                } else {
                    value = std::exp(static_cast<double>(x[at(j)]) - maximum) / sum;
                }
                out[at(j)] = static_cast<float>(value);
            }
        }
    }
    return out;
}

// Backward from the forward output y and upstream gradient g.
inline std::vector<float> softmax_backward(const std::vector<float>& y, const std::vector<float>& g,
                                           std::int64_t outer, std::int64_t length,
                                           std::int64_t inner, bool log) {
    std::vector<float> out(y.size());
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i = 0; i < inner; ++i) {
            auto at = [&](std::int64_t j) {
                return static_cast<std::size_t>((o * length + j) * inner + i);
            };
            double s = 0.0;
            for (std::int64_t j = 0; j < length; ++j) {
                s += log ? static_cast<double>(g[at(j)])
                         : static_cast<double>(g[at(j)]) * static_cast<double>(y[at(j)]);
            }
            for (std::int64_t j = 0; j < length; ++j) {
                const double yj = y[at(j)], gj = g[at(j)];
                out[at(j)] = static_cast<float>(log ? gj - std::exp(yj) * s : yj * (gj - s));
            }
        }
    }
    return out;
}

// Per-row cross-entropy of [rows, classes] logits: -log(softmax(row)[label]).
inline std::vector<double> cross_entropy(const std::vector<float>& logits,
                                         const std::vector<std::int64_t>& labels,
                                         std::int64_t classes) {
    std::vector<double> loss(labels.size());
    for (std::size_t r = 0; r < labels.size(); ++r) {
        const float* x = logits.data() + r * static_cast<std::size_t>(classes);
        double maximum = x[0];
        for (std::int64_t j = 1; j < classes; ++j) {
            maximum = std::max(maximum, static_cast<double>(x[j]));
        }
        double sum = 0.0;
        for (std::int64_t j = 0; j < classes; ++j) {
            sum += std::exp(static_cast<double>(x[j]) - maximum);
        }
        loss[r] = maximum + std::log(sum) - static_cast<double>(x[labels[r]]);
    }
    return loss;
}

// d/d logits of the loss, with upstream[r] the gradient reaching row r.
inline std::vector<float> cross_entropy_backward(const std::vector<float>& logits,
                                                 const std::vector<std::int64_t>& labels,
                                                 std::int64_t classes,
                                                 const std::vector<double>& upstream) {
    std::vector<float> out(logits.size());
    const auto softmax_rows =
        softmax(logits, static_cast<std::int64_t>(labels.size()), classes, 1, false);
    for (std::size_t r = 0; r < labels.size(); ++r) {
        for (std::int64_t j = 0; j < classes; ++j) {
            const std::size_t at =
                r * static_cast<std::size_t>(classes) + static_cast<std::size_t>(j);
            const double hot = j == labels[r] ? 1.0 : 0.0;
            out[at] = static_cast<float>((softmax_rows[at] - hot) * upstream[r]);
        }
    }
    return out;
}

inline std::vector<double> mse(const std::vector<float>& x, const std::vector<float>& t) {
    std::vector<double> loss(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = static_cast<double>(x[i]) - t[i];
        loss[i] = d * d;
    }
    return loss;
}

// -(t log s + (1 - t) log(1 - s)) with s = sigmoid(x), written with log1p for accuracy.
inline std::vector<double> bce_with_logits(const std::vector<float>& x,
                                           const std::vector<float>& t) {
    std::vector<double> loss(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double v = x[i];
        // log(sigmoid(v)) = -log1p(exp(-v)); log(1 - sigmoid(v)) = -log1p(exp(v)).
        const double log_s = v >= 0 ? -std::log1p(std::exp(-v)) : v - std::log1p(std::exp(v));
        const double log_1ms = v >= 0 ? -v - std::log1p(std::exp(-v)) : -std::log1p(std::exp(v));
        loss[i] = -(t[i] * log_s + (1.0 - t[i]) * log_1ms);
    }
    return loss;
}

inline double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

inline double total(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum;
}

// Attention for one batch of contiguous q [lq, d], k [lk, d], v [lk, dv]. keep[i * lk + j] says
// whether query i may see key j (empty: all). A query that sees no key gets zeros.
inline std::vector<float> attention(const std::vector<float>& q, const std::vector<float>& k,
                                    const std::vector<float>& v, std::int64_t lq, std::int64_t lk,
                                    std::int64_t d, std::int64_t dv, double scale,
                                    const std::vector<bool>& keep) {
    std::vector<float> out(static_cast<std::size_t>(lq * dv), 0.0f);
    for (std::int64_t i = 0; i < lq; ++i) {
        std::vector<double> score(static_cast<std::size_t>(lk));
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int64_t j = 0; j < lk; ++j) {
            double s = 0.0;
            for (std::int64_t c = 0; c < d; ++c) {
                s += static_cast<double>(q[i * d + c]) * k[j * d + c];
            }
            const bool visible = keep.empty() || keep[static_cast<std::size_t>(i * lk + j)];
            score[j] = visible ? s * scale : -std::numeric_limits<double>::infinity();
            maximum = std::max(maximum, score[j]);
        }
        if (maximum == -std::numeric_limits<double>::infinity()) {
            continue; // every key removed
        }
        double sum = 0.0;
        for (double& s : score) {
            s = std::exp(s - maximum);
            sum += s;
        }
        for (std::int64_t c = 0; c < dv; ++c) {
            double acc = 0.0;
            for (std::int64_t j = 0; j < lk; ++j) {
                acc += score[j] / sum * v[j * dv + c];
            }
            out[static_cast<std::size_t>(i * dv + c)] = static_cast<float>(acc);
        }
    }
    return out;
}

} // namespace reference
