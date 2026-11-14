#pragma once

#include "core/dispatcher.hpp"

namespace mensurflow::cpu {
void sgd_f32(const float*, const float*, float*, float*, std::int64_t, float, float, float);
void adam_f32(const float*, const float*, float*, float*, float*, std::int64_t, float, float,
              float, float, float, std::int64_t, bool);
} // namespace mensurflow::cpu
