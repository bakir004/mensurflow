#include "backends/cpu/ops/normalization.hpp"
#include "backends/cpu/threads.hpp"

#include <cmath>

namespace mensurflow::cpu {

void layer_norm_f32(const float* x, const float* w, const float* b, float* y, std::int64_t rows,
                    std::int64_t width, float eps) {
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* in = x + r * width;
        float* out = y + r * width;
        double mean = 0.0;
        for (std::int64_t j = 0; j < width; ++j) mean += in[j];
        mean /= width;
        double variance = 0.0;
        for (std::int64_t j = 0; j < width; ++j) {
            const double d = in[j] - mean;
            variance += d * d;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(variance / width) + eps);
        for (std::int64_t j = 0; j < width; ++j) out[j] = (in[j] - static_cast<float>(mean)) * inv * w[j] + b[j];
    }
}

void layer_norm_backward_f32(const float* x, const float* dy, const float* w, float* dx, float* dw,
                             float* db, std::int64_t rows, std::int64_t width, float eps) {
    for (std::int64_t j = 0; j < width; ++j) dw[j] = db[j] = 0.0f;
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* in = x + r * width;
        const float* grad = dy + r * width;
        double mean = 0.0, sum1 = 0.0, sum2 = 0.0;
        for (std::int64_t j = 0; j < width; ++j) mean += in[j];
        mean /= width;
        double variance = 0.0;
        for (std::int64_t j = 0; j < width; ++j) { const double d = in[j] - mean; variance += d * d; }
        const float inv = 1.0f / std::sqrt(static_cast<float>(variance / width) + eps);
        for (std::int64_t j = 0; j < width; ++j) {
            const float xn = (in[j] - static_cast<float>(mean)) * inv;
            dw[j] += grad[j] * xn; db[j] += grad[j];
            sum1 += static_cast<double>(grad[j] * w[j]);
            sum2 += static_cast<double>(grad[j] * w[j] * xn);
        }
        for (std::int64_t j = 0; j < width; ++j) {
            const float xn = (in[j] - static_cast<float>(mean)) * inv;
            dx[r * width + j] = inv * w[j] *
                (grad[j] - static_cast<float>(sum1 / width) - xn * static_cast<float>(sum2 / width));
        }
    }
}

void rms_norm_f32(const float* x, const float* w, float* y, std::int64_t rows, std::int64_t width,
                  float eps) {
    for (std::int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < width; ++j) sum += static_cast<double>(x[r * width + j]) * x[r * width + j];
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
        for (std::int64_t j = 0; j < width; ++j) y[r * width + j] = x[r * width + j] * inv * w[j];
    }
}

void rms_norm_backward_f32(const float* x, const float* dy, const float* w, float* dx, float* dw,
                           std::int64_t rows, std::int64_t width, float eps) {
    for (std::int64_t j = 0; j < width; ++j) dw[j] = 0.0f;
    for (std::int64_t r = 0; r < rows; ++r) {
        double sum = 0.0, dot = 0.0;
        for (std::int64_t j = 0; j < width; ++j) sum += static_cast<double>(x[r * width + j]) * x[r * width + j];
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
        for (std::int64_t j = 0; j < width; ++j) {
            const float xn = x[r * width + j] * inv;
            dw[j] += dy[r * width + j] * xn;
            dot += static_cast<double>(dy[r * width + j] * w[j] * x[r * width + j]);
        }
        const float correction = inv * inv * inv * static_cast<float>(dot / width);
        for (std::int64_t j = 0; j < width; ++j)
            dx[r * width + j] = dy[r * width + j] * w[j] * inv - correction * x[r * width + j];
    }
}

void batch_norm_f32(const float* x, const float* w, const float* b, float* running_mean,
                    float* running_var, float* y, float* saved_mean, std::int64_t n,
                    std::int64_t c, std::int64_t spatial, bool training, float momentum, float eps) {
    const std::int64_t count = n * spatial;
    for (std::int64_t ch = 0; ch < c; ++ch) {
        double mean = 0.0;
        for (std::int64_t bn = 0; bn < n; ++bn) for (std::int64_t s = 0; s < spatial; ++s)
            mean += x[(bn * c + ch) * spatial + s];
        mean = training ? mean / count : running_mean[ch];
        double var = 0.0;
        if (training) for (std::int64_t bn = 0; bn < n; ++bn) for (std::int64_t s = 0; s < spatial; ++s) {
            const double d = x[(bn * c + ch) * spatial + s] - mean; var += d * d;
        }
        var = training ? var / count : running_var[ch];
        saved_mean[ch] = static_cast<float>(mean);
        if (training) {
            saved_mean[c + ch] = static_cast<float>(var);
            running_mean[ch] = (1.0f - momentum) * running_mean[ch] + momentum * static_cast<float>(mean);
            running_var[ch] = (1.0f - momentum) * running_var[ch] + momentum * static_cast<float>(var);
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        for (std::int64_t bn = 0; bn < n; ++bn) for (std::int64_t s = 0; s < spatial; ++s) {
            const auto idx = (bn * c + ch) * spatial + s;
            y[idx] = (x[idx] - static_cast<float>(mean)) * inv * w[ch] + b[ch];
        }
    }
}

void batch_norm_backward_f32(const float* x, const float* dy, const float* w, const float* mean,
                             float* dx, float* dw, float* db, std::int64_t n, std::int64_t c,
                             std::int64_t spatial, bool training, float, float eps) {
    const std::int64_t count = n * spatial;
    for (std::int64_t ch = 0; ch < c; ++ch) {
        double sum = 0.0, dot = 0.0;
        const float variance_mean = mean[c + ch];
        const float inv = 1.0f / std::sqrt(variance_mean + eps);
        for (std::int64_t bn = 0; bn < n; ++bn) for (std::int64_t s = 0; s < spatial; ++s) {
            const auto idx = (bn * c + ch) * spatial + s;
            const float xn = (x[idx] - mean[ch]) * inv;
            sum += dy[idx]; dot += static_cast<double>(dy[idx] * xn);
        }
        dw[ch] = static_cast<float>(dot); db[ch] = static_cast<float>(sum);
        for (std::int64_t bn = 0; bn < n; ++bn) for (std::int64_t s = 0; s < spatial; ++s) {
            const auto idx = (bn * c + ch) * spatial + s;
            if (!training) dx[idx] = dy[idx] * w[ch] * inv;
            else {
                const float xn = (x[idx] - mean[ch]) * inv;
                dx[idx] = w[ch] * inv * (dy[idx] - static_cast<float>(sum / count) -
                                         xn * static_cast<float>(dot / count));
            }
        }
    }
}
} // namespace mensurflow::cpu
