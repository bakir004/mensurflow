// Tests element counts, stride math, contiguity checks, and broadcasting rules.
#include "mensurflow/shape.hpp"
#include "test_util.hpp"

#include <limits>
#include <stdexcept>

using namespace mensurflow;

namespace {

void test_numel_and_validation() {
    test::require(numel({}) == 1, "scalar has one element");
    test::require(numel({2, 3, 4}) == 24, "numel");
    test::require(numel({5, 0, 3}) == 0, "zero dimension");
    validate_shape({2, 0, 7});
    test::require_throws<std::invalid_argument>([] { validate_shape({2, -1}); }, "negative dim");
    const std::int64_t big = std::numeric_limits<std::int64_t>::max() / 2;
    test::require_throws<std::invalid_argument>([big] { validate_shape({big, 3}); }, "overflow");
}

void test_strides() {
    test::require(contiguous_strides({2, 3, 4}) == Strides{12, 4, 1}, "row-major strides");
    test::require(contiguous_strides({}).empty(), "scalar strides");
    test::require(is_contiguous({2, 3}, {3, 1}), "contiguous");
    test::require(!is_contiguous({2, 3}, {1, 2}), "transposed is not contiguous");
    test::require(is_contiguous({2, 1, 3}, {3, 99, 1}), "size-1 dim stride is ignored");
    test::require(!is_contiguous({2, 3}, {6, 1}), "gap between rows");
    test::require(is_contiguous({0, 3}, {5, 7}), "empty tensor");
}

void test_broadcasting() {
    test::require(broadcast_shapes({2, 3}, {3}) == Shape{2, 3}, "row vector");
    test::require(broadcast_shapes({4, 1}, {1, 5}) == Shape{4, 5}, "outer product");
    test::require(broadcast_shapes({}, {2, 2}) == Shape{2, 2}, "scalar");
    test::require(broadcast_shapes({8, 1, 6, 1}, {7, 1, 5}) == Shape{8, 7, 6, 5}, "numpy docs");
    test::require(broadcast_shapes({0}, {1}) == Shape{0}, "empty broadcasts");
    test::require_throws<std::invalid_argument>([] { broadcast_shapes({2, 3}, {4}); },
                                                "incompatible");

    // [3] viewed as [2, 3]: new leading dim gets stride 0.
    test::require(broadcast_strides({3}, {1}, {2, 3}) == Strides{0, 1}, "new leading dim");
    // [4, 1] viewed as [4, 5]: the size-1 dim gets stride 0.
    test::require(broadcast_strides({4, 1}, {1, 1}, {4, 5}) == Strides{1, 0}, "size-1 dim");
    test::require_throws<std::invalid_argument>([] { broadcast_strides({3}, {1}, {2, 4}); },
                                                "bad target");
}

void test_dims_and_names() {
    test::require(normalize_dim(-1, 3) == 2, "-1 is last");
    test::require(normalize_dim(0, 3) == 0, "0 is first");
    test::require_throws<std::out_of_range>([] { normalize_dim(3, 3); }, "too large");
    test::require_throws<std::out_of_range>([] { normalize_dim(-4, 3); }, "too negative");
    test::require(to_string(Shape{2, 3}) == "[2, 3]", "to_string");
    test::require(to_string(Shape{}) == "[]", "scalar to_string");
}

} // namespace

int main() {
    return test::run("shape_test", [] {
        test_numel_and_validation();
        test_strides();
        test_broadcasting();
        test_dims_and_names();
    });
}
