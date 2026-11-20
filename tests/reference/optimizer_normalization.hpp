// Scalar references for the optimizers and the normalization layers: the textbook formulas in double
// precision, one element at a time, with no parallelism and no shared code with the kernels.
//
// The normalization backward passes are written as the full Jacobian-vector product
//   dx_k = sum_i g_i * d xhat_i / d x_k,  d xhat_i / d x_k = inv * (delta_ik - 1/N - xhat_i xhat_k / N)
// (and its RMS counterpart), which costs N^2 per line but shares nothing with the kernels' closed
// form. The tests also check these formulas against finite differences of the forward references.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reference {

using Vec = std::vector<double>;

inline Vec to_double(const std::vector<float>& v) { return Vec(v.begin(), v.end()); }

// ------------------------------------------------------------------------------- optimizers

// One PyTorch-style SGD step: returns the new parameter, and updates the velocity in place.
inline Vec sgd_step(const Vec& p, const Vec& g, Vec& velocity, double lr, double momentum,
                    double weight_decay) {
    Vec out(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double grad = g[i] + weight_decay * p[i];
        if (momentum > 0.0) {
            velocity[i] = momentum * velocity[i] + grad;
            out[i] = p[i] - lr * velocity[i];
        } else {
            out[i] = p[i] - lr * grad;
        }
    }
    return out;
}

// One Adam (decoupled == false) or AdamW (decoupled == true) step, with m and v updated in place.
inline Vec adam_step(const Vec& p, const Vec& g, Vec& m, Vec& v, double lr, long step, double beta1,
                     double beta2, double eps, double weight_decay, bool decoupled) {
    Vec out(p.size());
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double grad = decoupled ? g[i] : g[i] + weight_decay * p[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
        const double base = decoupled ? p[i] * (1.0 - lr * weight_decay) : p[i];
        out[i] = base - (lr / bc1) * m[i] / (std::sqrt(v[i]) / std::sqrt(bc2) + eps);
    }
    return out;
}

// ------------------------------------------------------------------------ layer and RMS norm

struct Grads {
    Vec dx, dw, db;
};

// Layer norm over `rows` lines of `width`; w and b have `width` elements.
inline Vec layer_norm(const Vec& x, const Vec& w, const Vec& b, std::int64_t rows,
                      std::int64_t width, double eps) {
    Vec y(x.size());
    for (std::int64_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            mean += x[static_cast<std::size_t>(r * width + j)];
        }
        mean /= static_cast<double>(width);
        double var = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            const double d = x[static_cast<std::size_t>(r * width + j)] - mean;
            var += d * d;
        }
        var /= static_cast<double>(width);
        for (std::int64_t j = 0; j < width; ++j) {
            const auto k = static_cast<std::size_t>(r * width + j);
            y[k] = (x[k] - mean) / std::sqrt(var + eps) * w[static_cast<std::size_t>(j)] +
                   b[static_cast<std::size_t>(j)];
        }
    }
    return y;
}

inline Grads layer_norm_backward(const Vec& x, const Vec& dy, const Vec& w, std::int64_t rows,
                                 std::int64_t width, double eps) {
    Grads g{Vec(x.size()), Vec(static_cast<std::size_t>(width), 0.0),
            Vec(static_cast<std::size_t>(width), 0.0)};
    const double n = static_cast<double>(width);
    for (std::int64_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            mean += x[static_cast<std::size_t>(r * width + j)];
        }
        mean /= n;
        double var = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            const double d = x[static_cast<std::size_t>(r * width + j)] - mean;
            var += d * d;
        }
        var /= n;
        const double inv = 1.0 / std::sqrt(var + eps);
        Vec xhat(static_cast<std::size_t>(width));
        for (std::int64_t j = 0; j < width; ++j) {
            xhat[static_cast<std::size_t>(j)] =
                (x[static_cast<std::size_t>(r * width + j)] - mean) * inv;
        }
        for (std::int64_t j = 0; j < width; ++j) {
            const auto jj = static_cast<std::size_t>(j);
            const double grad = dy[static_cast<std::size_t>(r * width + j)];
            g.dw[jj] += grad * xhat[jj];
            g.db[jj] += grad;
        }
        for (std::int64_t k = 0; k < width; ++k) {
            double sum = 0.0;
            for (std::int64_t i = 0; i < width; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                const double jac = inv * ((i == k ? 1.0 : 0.0) - 1.0 / n -
                                          xhat[ii] * xhat[static_cast<std::size_t>(k)] / n);
                sum += dy[static_cast<std::size_t>(r * width + i)] * w[ii] * jac;
            }
            g.dx[static_cast<std::size_t>(r * width + k)] = sum;
        }
    }
    return g;
}

inline Vec rms_norm(const Vec& x, const Vec& w, std::int64_t rows, std::int64_t width, double eps) {
    Vec y(x.size());
    for (std::int64_t r = 0; r < rows; ++r) {
        double ms = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            const double v = x[static_cast<std::size_t>(r * width + j)];
            ms += v * v;
        }
        ms /= static_cast<double>(width);
        for (std::int64_t j = 0; j < width; ++j) {
            const auto k = static_cast<std::size_t>(r * width + j);
            y[k] = x[k] / std::sqrt(ms + eps) * w[static_cast<std::size_t>(j)];
        }
    }
    return y;
}

inline Grads rms_norm_backward(const Vec& x, const Vec& dy, const Vec& w, std::int64_t rows,
                               std::int64_t width, double eps) {
    Grads g{Vec(x.size()), Vec(static_cast<std::size_t>(width), 0.0), Vec()};
    const double n = static_cast<double>(width);
    for (std::int64_t r = 0; r < rows; ++r) {
        double ms = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            const double v = x[static_cast<std::size_t>(r * width + j)];
            ms += v * v;
        }
        ms /= n;
        const double inv = 1.0 / std::sqrt(ms + eps);
        for (std::int64_t j = 0; j < width; ++j) {
            const auto k = static_cast<std::size_t>(r * width + j);
            g.dw[static_cast<std::size_t>(j)] += dy[k] * x[k] * inv;
        }
        for (std::int64_t k = 0; k < width; ++k) {
            double sum = 0.0;
            for (std::int64_t i = 0; i < width; ++i) {
                const auto ii = static_cast<std::size_t>(r * width + i);
                // d (x_i * inv) / d x_k, with d inv / d x_k = -inv^3 x_k / N.
                const double jac =
                    (i == k ? inv : 0.0) -
                    x[ii] * inv * inv * inv * x[static_cast<std::size_t>(r * width + k)] / n;
                sum += dy[ii] * w[static_cast<std::size_t>(i)] * jac;
            }
            g.dx[static_cast<std::size_t>(r * width + k)] = sum;
        }
    }
    return g;
}

// ------------------------------------------------------------------------------- batch norm

struct BatchResult {
    Vec y, running_mean, running_var;
};

// x = [n, c, s]. Training uses batch statistics (biased variance) and updates the running ones
// with the unbiased variance; evaluation uses the running ones.
inline BatchResult batch_norm(const Vec& x, const Vec& w, const Vec& b, Vec running_mean,
                              Vec running_var, std::int64_t n, std::int64_t c, std::int64_t s,
                              bool training, double momentum, double eps) {
    BatchResult r{Vec(x.size()), running_mean, running_var};
    const double count = static_cast<double>(n * s);
    for (std::int64_t ch = 0; ch < c; ++ch) {
        auto at = [&](std::int64_t i, std::int64_t j) {
            return static_cast<std::size_t>((i * c + ch) * s + j);
        };
        double mean = running_mean[static_cast<std::size_t>(ch)];
        double var = running_var[static_cast<std::size_t>(ch)];
        if (training) {
            mean = 0.0;
            for (std::int64_t i = 0; i < n; ++i)
                for (std::int64_t j = 0; j < s; ++j)
                    mean += x[at(i, j)];
            mean /= count;
            var = 0.0;
            for (std::int64_t i = 0; i < n; ++i)
                for (std::int64_t j = 0; j < s; ++j)
                    var += (x[at(i, j)] - mean) * (x[at(i, j)] - mean);
            var /= count;
            const auto k = static_cast<std::size_t>(ch);
            r.running_mean[k] = (1.0 - momentum) * running_mean[k] + momentum * mean;
            r.running_var[k] =
                (1.0 - momentum) * running_var[k] + momentum * var * count / (count - 1.0);
        }
        for (std::int64_t i = 0; i < n; ++i)
            for (std::int64_t j = 0; j < s; ++j)
                r.y[at(i, j)] =
                    (x[at(i, j)] - mean) / std::sqrt(var + eps) * w[static_cast<std::size_t>(ch)] +
                    b[static_cast<std::size_t>(ch)];
    }
    return r;
}

inline Grads batch_norm_backward(const Vec& x, const Vec& dy, const Vec& w, const Vec& running_mean,
                                 const Vec& running_var, std::int64_t n, std::int64_t c,
                                 std::int64_t s, bool training, double eps) {
    Grads g{Vec(x.size()), Vec(static_cast<std::size_t>(c), 0.0),
            Vec(static_cast<std::size_t>(c), 0.0)};
    const double count = static_cast<double>(n * s);
    for (std::int64_t ch = 0; ch < c; ++ch) {
        auto at = [&](std::int64_t i, std::int64_t j) {
            return static_cast<std::size_t>((i * c + ch) * s + j);
        };
        double mean = running_mean[static_cast<std::size_t>(ch)];
        double var = running_var[static_cast<std::size_t>(ch)];
        if (training) {
            mean = 0.0;
            for (std::int64_t i = 0; i < n; ++i)
                for (std::int64_t j = 0; j < s; ++j)
                    mean += x[at(i, j)];
            mean /= count;
            var = 0.0;
            for (std::int64_t i = 0; i < n; ++i)
                for (std::int64_t j = 0; j < s; ++j)
                    var += (x[at(i, j)] - mean) * (x[at(i, j)] - mean);
            var /= count;
        }
        const double inv = 1.0 / std::sqrt(var + eps);
        const auto k = static_cast<std::size_t>(ch);
        for (std::int64_t i = 0; i < n; ++i)
            for (std::int64_t j = 0; j < s; ++j) {
                g.dw[k] += dy[at(i, j)] * (x[at(i, j)] - mean) * inv;
                g.db[k] += dy[at(i, j)];
            }
        // Each output depends on every element of the channel in training (through mean and var),
        // and only on itself in evaluation.
        for (std::int64_t i2 = 0; i2 < n; ++i2)
            for (std::int64_t j2 = 0; j2 < s; ++j2) {
                double sum = 0.0;
                for (std::int64_t i = 0; i < n; ++i)
                    for (std::int64_t j = 0; j < s; ++j) {
                        const bool same = i == i2 && j == j2;
                        double jac = same ? inv : 0.0;
                        if (training) {
                            const double xh_i = (x[at(i, j)] - mean) * inv;
                            const double xh_k = (x[at(i2, j2)] - mean) * inv;
                            jac = inv * ((same ? 1.0 : 0.0) - 1.0 / count - xh_i * xh_k / count);
                        }
                        sum += dy[at(i, j)] * w[k] * jac;
                    }
                g.dx[at(i2, j2)] = sum;
            }
    }
    return g;
}

} // namespace reference
