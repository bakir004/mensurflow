#pragma once

#include <vector>

namespace prs {

enum class Backend { CPU, CUDA };

// True when CUDA support was built and a CUDA device is available.
bool cuda_available();

// Compute y[i] = alpha * x[i] + y[i]. Both vectors live in host memory.
// Sizes must match. Passing the same vector as x and y is supported.
// CUDA includes allocation, host/device transfers, execution, and copy-back.
// Throws std::invalid_argument for size mismatch, std::runtime_error on CUDA errors.
void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y,
          Backend backend = Backend::CPU);

} // namespace prs
