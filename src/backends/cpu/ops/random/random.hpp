// CPU F32 RNG and dropout kernels. Callers validate parameters and pack strided inputs.
// One loop supplies sequential and OpenMP execution, following the existing CPU convention.
#pragma once

#include <cstdint>

namespace mensurflow::cpu {

void uniform_f32(float* out, std::int64_t n, float low, float high, std::uint64_t seed,
                 std::uint64_t offset);
void normal_f32(float* out, std::int64_t n, float mean, float stddev, std::uint64_t seed,
                std::uint64_t offset);
void bernoulli_f32(float* out, std::int64_t n, float probability, std::uint64_t seed,
                   std::uint64_t offset);
void dropout_f32(const float* x, float* out, std::uint8_t* mask, std::int64_t n,
                 float probability, bool training, std::uint64_t seed, std::uint64_t offset);
void dropout_backward_f32(const float* gradient, const std::uint8_t* mask, float* out,
                          std::int64_t n, float probability, bool training);

} // namespace mensurflow::cpu
