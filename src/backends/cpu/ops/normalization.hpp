#pragma once

#include <cstdint>

namespace mensurflow::cpu {
void layer_norm_f32(const float*, const float*, const float*, float*, std::int64_t, std::int64_t,
                    float);
void layer_norm_backward_f32(const float*, const float*, const float*, float*, float*, float*,
                            std::int64_t, std::int64_t, float);
void rms_norm_f32(const float*, const float*, float*, std::int64_t, std::int64_t, float);
void rms_norm_backward_f32(const float*, const float*, const float*, float*, float*, std::int64_t,
                           std::int64_t, float);
void batch_norm_f32(const float*, const float*, const float*, float*, float*, float*,
                    float*, std::int64_t, std::int64_t, std::int64_t, bool, float, float);
void batch_norm_backward_f32(const float*, const float*, const float*, const float*, float*, float*,
                             float*, std::int64_t, std::int64_t, std::int64_t, bool, float, float);
} // namespace mensurflow::cpu
