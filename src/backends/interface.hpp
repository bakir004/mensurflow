#pragma once
#include <cstddef>

namespace prs::detail {
class BackendInterface {
public:
    virtual ~BackendInterface() = default;
    virtual bool available() const = 0;
    virtual void axpy(float alpha, const float* x, float* y, std::size_t count) const = 0;
    virtual void matmul(const float* a, const float* b, float* c,
                        std::size_t m, std::size_t n, std::size_t k) const = 0;
};
const BackendInterface& cpu_backend();
const BackendInterface& cuda_backend();
const BackendInterface& metal_backend();
} // namespace prs::detail
