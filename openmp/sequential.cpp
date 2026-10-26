#include <cstddef>

// Compiled separately with auto-vectorization disabled: a scalar, single-thread
// baseline rather than an accidentally vectorized "sequential" implementation.
double dot_sequential(const double* a, const double* b, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
