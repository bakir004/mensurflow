// CPU 2-D convolution kernels, float32, NCHW layout, im2col + GEMM. These are the sequential
// versions: they run on the calling thread (the products use the single-threaded packed GEMM), and
// they are the baseline the parallel versions are tested and benchmarked against.
//
// Callers (src/ops) have validated the shapes, so kernels do no checking of their own. All
// buffers are contiguous row-major. Only zero padding, dilation 1 and groups 1.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// The sizes of one convolution: input x[n, c_in, h, w], weight[c_out, c_in, kh, kw] and output
// y[n, c_out, h_out, w_out], where h_out = (h + 2 * pad_h - kh) / stride_h + 1 (rounded down) and
// w_out likewise. Input row ih = oh * stride_h - pad_h + i is read for output row oh and kernel
// row i; rows outside [0, h) are zero padding.
struct Conv2dShape {
    std::int64_t n = 0, c_in = 0, h = 0, w = 0;
    std::int64_t c_out = 0, kh = 0, kw = 0;
    std::int64_t stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0;
    std::int64_t h_out = 0, w_out = 0;
};

// One image x[c_in, h, w] to the matrix cols[c_in * kh * kw, h_out * w_out]: row
// (c * kh + i) * kw + j, column oh * w_out + ow holds
// x[c, oh * stride_h - pad_h + i, ow * stride_w - pad_w + j], or 0 where that cell is padding.
void im2col_f32(const float* image, float* cols, const Conv2dShape& s);

// The adjoint of im2col_f32: adds every entry of cols into the image cell it was read from and
// drops the padding entries. Accumulates, so the caller initializes image[c_in, h, w].
void col2im_f32(const float* cols, float* image, const Conv2dShape& s);

// y = conv2d(x, weight) + bias. For each image,
// y[c_out, h_out * w_out] = weight[c_out, K] * cols[K, h_out * w_out] with K = c_in * kh * kw.
// bias[c_out] may be nullptr (no bias). c_in == 0 gives the bias (or zeros).
void conv2d_f32_sequential(const float* x, const float* weight, const float* bias, float* y,
                           const Conv2dShape& s);

// grad_x[n, c_in, h, w] from grad_y[n, c_out, h_out, w_out]: per image, cols = weight^T * grad_y
// (weight read transposed in place), then col2im. Overwrites grad_x.
void conv2d_backward_data_f32_sequential(const float* grad_y, const float* weight, float* grad_x,
                                         const Conv2dShape& s);

// grad_weight[c_out, c_in, kh, kw] = sum over the batch of grad_y[c_out, h_out * w_out] * cols^T
// (cols read transposed in place). Overwrites grad_weight; an empty batch gives zeros.
void conv2d_backward_weight_f32_sequential(const float* grad_y, const float* x, float* grad_weight,
                                           const Conv2dShape& s);

// grad_bias[c_out] = sum of grad_y over the batch and both spatial dimensions, accumulated in
// double. Overwrites grad_bias.
void conv2d_backward_bias_f32_sequential(const float* grad_y, float* grad_bias,
                                         const Conv2dShape& s);

// ---- Parallel versions ---------------------------------------------------------------------
//
// Same results as the sequential versions up to float rounding, and the same bits at every thread
// count. Below conv2d_parallel_cutoff multiply-adds they run the per-image code on the calling
// thread (no threads, but the reused buffers and the faster im2col). A 1x1
// kernel with stride 1 and no padding skips im2col: the image is the column matrix. Temporary
// buffers are kept per thread between calls (they only grow; at most about 3 x 32 MB).

// How the forward pass spreads its work over threads.
enum class Conv2dStrategy {
    // Chooses one of the two below from the shape (see conv2d.cpp).
    Auto,
    // A chunk of the batch at a time (bounded by conv2d_column_budget floats): im2col of all its
    // images into one matrix [K, chunk * h_out * w_out] in parallel, then one large product on
    // all threads, then the result copied into NCHW order.
    Folded,
    // Whole images per thread: each thread runs im2col and the single-threaded product for its
    // own images, with its own column buffer. No synchronization inside an image.
    ImagePerThread,
};

// Multiply-adds below which the parallel versions stay on the calling thread.
constexpr std::int64_t conv2d_parallel_cutoff = std::int64_t{1} << 20;

// The largest column matrix the folded strategy builds, in floats (32 MB): the batch is processed
// in chunks of as many images as fit (at least one).
constexpr std::int64_t conv2d_column_budget = std::int64_t{1} << 23;

void conv2d_f32_parallel(const float* x, const float* weight, const float* bias, float* y,
                         const Conv2dShape& s, Conv2dStrategy strategy = Conv2dStrategy::Auto);

// With at least one image per thread, whole images per task (product on one thread, then
// col2im); otherwise per chunk of the batch, weight^T * grad_y as one product on all threads,
// then col2im with one task per (image, input channel) plane. Either way no two threads write the
// same cell of grad_x, and both give the same bits.
void conv2d_backward_data_f32_parallel(const float* grad_y, const float* weight, float* grad_x,
                                       const Conv2dShape& s);

// A small gradient (c_out * K below 2^18 floats): fixed groups of images, each group summed on one
// thread into its own partial matrix, the partials added in group order. Otherwise per chunk of
// the batch, one product of depth chunk * h_out * w_out (the batch folded into the depth) on all
// threads, added chunk after chunk. The choice and the groups depend only on the shape; no
// atomics anywhere.
void conv2d_backward_weight_f32_parallel(const float* grad_y, const float* x, float* grad_weight,
                                         const Conv2dShape& s);

// One task per output channel.
void conv2d_backward_bias_f32_parallel(const float* grad_y, float* grad_bias,
                                       const Conv2dShape& s);

} // namespace mensurflow::cpu
