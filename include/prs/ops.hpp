#pragma once

#include "prs/backend.hpp"
#include <cstddef>
#include <vector>

namespace prs {
// Synchronous host-buffer operations. GPU calls include allocations and copies.
// y[i] = alpha*x[i] + y[i]. Sizes must match; x and y may be the same vector.
// CPU remains the default for compatibility with the original AXPY API.
void axpy(float alpha, const std::vector<float>& x, std::vector<float>& y,
          Backend backend = Backend::CPU);

// Row-major C[M,N] = A[M,K] * B[K,N]. All vectors must already have exactly
// M*K, K*N, and M*N elements respectively. C may alias A or B (via a temporary).
// K=0 fills C with zeros; M=0 or N=0 produces an empty C.
// Invalid sizes throw invalid_argument; dimension/byte overflow throws
// overflow_error; unavailable backends and GPU errors throw runtime_error.
void matmul(const std::vector<float>& a, const std::vector<float>& b,
            std::vector<float>& c, std::size_t m, std::size_t n, std::size_t k,
            Backend backend = Backend::Auto);
} // namespace prs
