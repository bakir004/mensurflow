// Minimal test helpers: each test is a plain executable that returns non-zero on failure.
#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace test {

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
