// Minimal test helpers: each test is a plain executable that returns non-zero on failure.
#pragma once

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

// Zero tolerance means exact bits. Otherwise compare finite values with relative error plus
// one float32 subnormal step; NaNs must agree in kind, infinities and signed zeros in sign.
inline bool close_f32(float actual, float expected, float relative_tolerance = 0.0f) {
    if (relative_tolerance == 0.0f) {
        return std::memcmp(&actual, &expected, sizeof(float)) == 0;
    }
    if (std::isnan(actual) || std::isnan(expected)) {
        return std::isnan(actual) && std::isnan(expected);
    }
    if (std::isinf(actual) || std::isinf(expected)) {
        return actual == expected;
    }
    if (actual == 0.0f && expected == 0.0f) {
        return std::signbit(actual) == std::signbit(expected);
    }
    const double difference = std::abs(static_cast<double>(actual) - expected);
    const double tolerance = relative_tolerance * std::abs(static_cast<double>(expected)) +
                             std::numeric_limits<float>::denorm_min();
    return difference <= tolerance;
}

inline bool matches_f32(const std::vector<float>& actual, const std::vector<float>& expected,
                        float relative_tolerance = 0.0f) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!close_f32(actual[i], expected[i], relative_tolerance)) {
            return false;
        }
    }
    return true;
}

// Report a failed condition through run()'s common exception handler.
inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Passes only if fn throws an exception of type E.
template <typename E, typename Fn> void require_throws(Fn&& fn, const std::string& message) {
    try {
        fn();
    } catch (const E&) {
        return;
    }
    throw std::runtime_error("expected exception: " + message);
}

template <typename Fn> int run(const char* name, Fn&& body) {
    try {
        body();
    } catch (const std::exception& e) {
        std::cerr << name << " FAILED: " << e.what() << '\n';
        return 1;
    }
    std::cout << name << " passed\n";
    return 0;
}

} // namespace test
