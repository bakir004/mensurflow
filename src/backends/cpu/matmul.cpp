#include <algorithm>
#include <cstddef>
namespace prs::detail {
void cpu_matmul(const float* a, const float* b, float* c,
                std::size_t m, std::size_t n, std::size_t k) {
    std::fill(c, c + m*n, 0.0f);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const float value = a[i*k + p];
            for (std::size_t j = 0; j < n; ++j) c[i*n + j] += value * b[p*n + j];
        }
    }
}
}
