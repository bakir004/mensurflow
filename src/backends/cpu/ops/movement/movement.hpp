// CPU data-movement kernels over contiguous float32 buffers: concat and constant pad. Both only copy
// and fill, so they are memory-bound; the copies are memcpy and the fills plain loops the compiler
// vectorizes. Each is one loop marked with MENSURFLOW_PARALLEL_FOR (threads.hpp), which splits it
// across threads in an OpenMP build. split needs no kernel: it returns views (see ops/movement.hpp).
//
// Callers (src/ops) have already validated the shapes, so kernels do no checking of their own.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

// Copies a strided float32 view into contiguous row-major storage.
void strided_copy_f32(const float* x, float* out, const std::int64_t* shape,
                      const std::int64_t* strides, int ndim);

// Transposes a contiguous row-major [rows, cols] matrix into [cols, rows].
void transpose_f32(const float* x, float* out, std::int64_t rows, std::int64_t cols);

// Joins `count` contiguous inputs along one axis. Seen around that axis, input t is an
// [outer, widths[t]] matrix (widths[t] = its size along the axis times the product of the sizes
// after it), and the output is the [outer, sum of widths] matrix whose rows are the input rows side
// by side. Long rows are copied in pieces so that even one row (concat along axis 0) is shared
// across threads.
void concat_f32(const float* const* inputs, const std::int64_t* widths, std::int64_t count,
                float* out, std::int64_t outer);

// Constant padding: out has shape[d] + before[d] + after[d] elements along each of the ndim
// dimensions (ndim >= 1); the input sits at offset before[d] and everything else is `value`. x is the
// contiguous input of shape `shape`; it is not read when any of its dimensions is 0.
void pad_f32(const float* x, float* out, const std::int64_t* shape, const std::int64_t* before,
             const std::int64_t* after, int ndim, float value);

} // namespace mensurflow::cpu
