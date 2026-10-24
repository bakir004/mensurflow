#pragma once

#include <vector>

namespace prs {

enum class Backend { CPU, CUDA, Metal };

// True when CUDA support was built and a CUDA device is available.
bool cuda_available();

// True when Metal support was built and a Metal device is available.
bool metal_available();

// Compute y[i] = alpha * x[i] + y[i]. Both vectors live in host memory.
// Sizes must match. Passing the same vector as x and y is supported.
// GPU backends include allocation, input/output copies, and execution.
// Throws std::invalid_argument for size mismatch, std::runtime_error on GPU errors.
void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y,
          Backend backend = Backend::CPU);

} // namespace prs
