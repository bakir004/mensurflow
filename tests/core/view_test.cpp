// Tests view ops (transpose, slice, expand, unsqueeze, view) and strided copy_to_host.
#include "mensurflow/tensor.hpp"
#include "test_util.hpp"

#include <stdexcept>
#include <vector>

using namespace mensurflow;
using Values = std::vector<float>;

namespace {

// [[0, 1, 2],
//  [3, 4, 5]]
Tensor matrix() { return Tensor::from_vector(Values{0, 1, 2, 3, 4, 5}, {2, 3}); }

void test_transpose() {
    Tensor m = matrix();
    Tensor t = m.transpose(0, 1);
    test::require(t.shape() == Shape{3, 2} && t.strides() == Strides{1, 3}, "transposed meta");
    test::require(!t.is_contiguous(), "transpose is not contiguous");
    test::require(t.shares_storage(m), "transpose shares storage");
    test::require(t.to_vector<float>() == Values{0, 3, 1, 4, 2, 5}, "transposed values");
    test::require(t.transpose(-1, -2).to_vector<float>() == m.to_vector<float>(), "double");
}

void test_slice() {
    Tensor m = matrix();
    Tensor row = m.slice(0, 1, 2);
    test::require(row.shape() == Shape{1, 3} && row.offset() == 3, "row slice meta");
    test::require(row.is_contiguous(), "row slice is contiguous");
    test::require(row.to_vector<float>() == Values{3, 4, 5}, "row slice");

    Tensor col = m.slice(1, 1, 2);
    test::require(col.offset() == 1 && !col.is_contiguous(), "column slice meta");
    test::require(col.to_vector<float>() == Values{1, 4}, "column slice");

    test::require(m.slice(1, 0, 3, 2).to_vector<float>() == Values{0, 2, 3, 5}, "step 2");
    test::require(m.slice(1, -2, 100).to_vector<float>() == Values{1, 2, 4, 5}, "negative+clamp");
    test::require(m.slice(1, 2, 1).numel() == 0, "end before start is empty");
    test::require_throws<std::invalid_argument>([&] { m.slice(1, 0, 3, 0); }, "step 0");
}

void test_expand_and_unsqueeze() {
    Tensor bias = Tensor::from_vector(Values{10, 20, 30}, {3});
    Tensor rows = bias.expand({2, 3});
    test::require(rows.strides() == Strides{0, 1}, "expand strides");
    test::require(rows.to_vector<float>() == Values{10, 20, 30, 10, 20, 30}, "expand rows");

    Tensor column = bias.unsqueeze(1);
    test::require(column.shape() == Shape{3, 1}, "unsqueeze shape");
    test::require(column.expand({3, 2}).to_vector<float>() == Values{10, 10, 20, 20, 30, 30},
                  "expand columns");
    test::require(bias.unsqueeze(0).shape() == Shape{1, 3}, "unsqueeze front");
    test::require(bias.unsqueeze(-1).shape() == Shape{3, 1}, "unsqueeze -1 appends");
    test::require_throws<std::invalid_argument>([&] { bias.expand({4}); }, "bad expand");
}

void test_view() {
    Tensor m = matrix();
    test::require(m.view({3, 2}).to_vector<float>() == m.to_vector<float>(), "same order");
    test::require(m.view({-1}).shape() == Shape{6}, "infer -1");
    test::require(m.view({3, -1}).shape() == Shape{3, 2}, "infer inner -1");
    // A row slice is contiguous and starts at an offset; view must keep that offset.
    test::require(m.slice(0, 1, 2).view({3}).to_vector<float>() == Values{3, 4, 5}, "offset");
    test::require_throws<std::invalid_argument>([&] { m.view({4, 2}); }, "wrong count");
    test::require_throws<std::invalid_argument>([&] { m.view({-1, -1}); }, "two -1");
    test::require_throws<std::invalid_argument>([&] { m.transpose(0, 1).view({6}); },
                                                "non-contiguous view");
}

void test_composed_views() {
    // 2x3x4 tensor holding 0..23; views compose because each only rewrites shape/strides/offset.
    Values values(24);
    for (int i = 0; i < 24; ++i) {
        values[static_cast<std::size_t>(i)] = static_cast<float>(i);
    }
    Tensor t = Tensor::from_vector(values, {2, 3, 4});
    // Swap the outer two dims, then take the last column of every row: element (b, r, 3).
    Tensor v = t.transpose(0, 1).slice(2, 3, 4);
    test::require(v.shape() == Shape{3, 2, 1}, "composed shape");
    test::require(v.to_vector<float>() == Values{3, 15, 7, 19, 11, 23}, "composed values");
}

} // namespace

int main() {
    return test::run("view_test", [] {
        test_transpose();
        test_slice();
        test_expand_and_unsqueeze();
        test_view();
        test_composed_views();
    });
}
