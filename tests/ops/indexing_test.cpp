// CPU kernels against independent scalar references, then public validation, views and autograd.
#include "backends/cpu/ops/indexing/indexing.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "mensurflow/ops/indexing.hpp"
#include "reference/indexing.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

std::vector<float> values(std::int64_t n) {
    std::vector<float> out(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<float>(i % 29 - 14) / 8.0f;
    }
    return out;
}

std::vector<std::int64_t> indices(std::int64_t count, std::int64_t rows) {
    std::vector<std::int64_t> out(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        out[static_cast<std::size_t>(i)] = (i * 13 + 2) % rows;
    }
    return out;
}

void test_kernels() {
    const int original_threads = cpu::max_threads();
    for (const std::int64_t count : {0, 1, 8191, 8192, 8193, 40000}) {
        const auto ids = indices(count, 4);
        const auto expected = reference::one_hot(ids, 4);
        for (const int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<std::int64_t> out(expected.size(), -1);
            cpu::one_hot_i64(ids.data(), out.data(), count, 4);
            test::require(out == expected, "one_hot reference threads=" + std::to_string(threads));
        }
    }
    for (const std::int64_t classes : {1023, 1024, 1025, 40000}) {
        // Selected classes at both sides of a chunk boundary, including very wide rows.
        const std::vector<std::int64_t> ids{0, classes - 1, classes / 2};
        const auto expected = reference::one_hot(ids, classes);
        for (const int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<std::int64_t> out(expected.size(), -1);
            cpu::one_hot_i64(ids.data(), out.data(), static_cast<std::int64_t>(ids.size()),
                              classes);
            test::require(out == expected, "one_hot class chunks");
        }
    }

    // Below, at and above the flat-element cutoff; widths crossing the memcpy/block boundaries.
    const std::int64_t cases[][3] = {{0, 7, 4},     {1, 1, 1},      {7, 13, 0},
                                    {8191, 7, 4},  {8192, 7, 4},   {8193, 7, 4},
                                    {512, 19, 65}, {3, 7, 1025},   {1, 3, 40000},
                                    {40000, 1, 3}, {33, 1000, 17}};
    for (const auto& shape : cases) {
        const auto count = shape[0], rows = shape[1], width = shape[2];
        const auto ids = indices(count, rows);
        const auto weight = values(rows * width);
        const auto gradient = values(count * width);
        const auto expected = reference::embedding(weight, ids, width);
        const auto expected_backward = reference::embedding_backward(gradient, ids, rows, width);
        for (const int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<float> out(expected.size(), 123.0f);
            std::vector<float> backward(expected_backward.size(), 123.0f);
            cpu::embedding_f32(weight.data(), ids.data(), out.data(), count, width);
            cpu::embedding_backward_f32(gradient.data(), ids.data(), backward.data(), count, rows,
                                         width);
            test::require(test::matches_f32(out, expected), "embedding kernel reference");
            test::require(test::matches_f32(backward, expected_backward),
                          "embedding backward reference threads=" + std::to_string(threads));
        }
    }
    // Cancellation makes input order observable even when all additions are race-free. Force
    // the parallel branch, give every row repeated IDs, and split feature tiles across threads.
    constexpr std::int64_t cancellation_count = 8192, cancellation_rows = 7,
                           cancellation_width = 129;
    auto repeated = indices(cancellation_count, cancellation_rows);
    std::vector<float> cancellation(
        static_cast<std::size_t>(cancellation_count * cancellation_width));
    const float pattern[]{1e20f, 1.0f, -1e20f, 7.0f};
    for (std::int64_t i = 0; i < cancellation_count; ++i) {
        for (std::int64_t column = 0; column < cancellation_width; ++column) {
            cancellation[static_cast<std::size_t>(i * cancellation_width + column)] =
                pattern[(i / cancellation_rows + column) % 4];
        }
    }
    const auto expected = reference::embedding_backward(cancellation, repeated,
                                                         cancellation_rows, cancellation_width);
    for (const int threads : {1, 2, 3, original_threads}) {
        cpu::set_max_threads(threads);
        std::vector<float> out(expected.size(), 123.0f);
        cpu::embedding_backward_f32(cancellation.data(), repeated.data(), out.data(),
                                     cancellation_count, cancellation_rows, cancellation_width);
        test::require(test::matches_f32(out, expected),
                      "embedding backward stable cancellation order threads=" +
                          std::to_string(threads));
    }
    cpu::set_max_threads(original_threads);
}

void test_api() {
    const auto raw_indices = indices(6, 4);
    const Tensor ids = Tensor::from_vector(raw_indices, {2, 3});
    const Tensor labels = one_hot(ids, 4);
    test::require(labels.shape() == Shape{2, 3, 4} && labels.dtype() == DType::I64,
                  "one_hot shape and dtype");
    test::require(labels.to_vector<std::int64_t>() == reference::one_hot(raw_indices, 4),
                  "one_hot API values");
    const Tensor scalar_id = Tensor::from_vector(std::vector<std::int64_t>{2}, {});
    test::require(one_hot(scalar_id, 4).shape() == Shape{4}, "one_hot scalar shape");
    test::require(one_hot(scalar_id, 4).to_vector<std::int64_t>() ==
                      std::vector<std::int64_t>{0, 0, 1, 0},
                  "one_hot scalar values");
    const Tensor empty_ids = Tensor::empty({2, 0}, DType::I64);
    test::require(one_hot(empty_ids, 4).shape() == Shape{2, 0, 4}, "one_hot empty shape");
    const Tensor single_class = Tensor::zeros({3}, DType::I64);
    test::require(one_hot(single_class, 1).to_vector<std::int64_t>() ==
                      std::vector<std::int64_t>{1, 1, 1},
                  "one_hot one class");

    const Tensor weight = Tensor::from_vector(values(20), {4, 5});
    const Tensor output = embedding(weight, ids);
    test::require(output.shape() == Shape{2, 3, 5}, "embedding shape");
    test::require(test::matches_f32(output.to_vector<float>(),
                                     reference::embedding(values(20), raw_indices, 5)),
                  "embedding API values");
    test::require(embedding(weight, scalar_id).shape() == Shape{5}, "embedding scalar shape");
    test::require(embedding(weight, empty_ids).shape() == Shape{2, 0, 5}, "embedding empty");
    test::require(embedding(Tensor::empty({4, 0}), ids).shape() == Shape{2, 3, 0},
                  "embedding zero width");
    test::require(embedding(Tensor::empty({0, 5}), empty_ids).shape() == Shape{2, 0, 5},
                  "embedding empty table and indices");
    const Tensor gradient = Tensor::from_vector(values(30), {2, 3, 5});
    test::require(test::matches_f32(embedding_backward(gradient, ids, 4).to_vector<float>(),
                                     reference::embedding_backward(values(30), raw_indices, 4, 5)),
                  "embedding backward API");
    test::require(embedding_backward(Tensor::empty({2, 0, 5}), empty_ids, 4).to_vector<float>() ==
                      std::vector<float>(20, 0.0f),
                  "embedding backward empty indices zeros every row");
    test::require(embedding_backward(Tensor::empty({2, 0, 5}), empty_ids, 0).shape() == Shape{0, 5},
                  "embedding backward zero rows");

    // Public adapters pack noncontiguous and expanded views; references use logical element order.
    const Tensor strided_ids = ids.transpose(0, 1);
    const auto logical_ids = strided_ids.to_vector<std::int64_t>();
    test::require(one_hot(strided_ids, 4).to_vector<std::int64_t>() ==
                      reference::one_hot(logical_ids, 4),
                  "one_hot transposed indices");
    const Tensor strided_weight = Tensor::from_vector(values(20), {5, 4}).transpose(0, 1);
    test::require(test::matches_f32(embedding(strided_weight, strided_ids).to_vector<float>(),
                                     reference::embedding(strided_weight.to_vector<float>(),
                                                          logical_ids, 5)),
                  "embedding transposed weight and indices");
    const Tensor expanded_ids = scalar_id.expand({3, 2});
    test::require(one_hot(expanded_ids, 4).to_vector<std::int64_t>() ==
                      reference::one_hot(std::vector<std::int64_t>(6, 2), 4),
                  "one_hot expanded indices");
    const Tensor expanded_weight = Tensor::from_vector(values(5), {1, 5}).expand({4, 5});
    const auto expanded_expected = reference::embedding(expanded_weight.to_vector<float>(),
                                                        expanded_ids.to_vector<std::int64_t>(), 5);
    test::require(test::matches_f32(embedding(expanded_weight, expanded_ids).to_vector<float>(),
                                     expanded_expected),
                  "embedding expanded weight and indices");
    const Tensor sliced_ids = ids.view({6}).slice(0, 1, 6, 2);
    test::require(one_hot(sliced_ids, 4).to_vector<std::int64_t>() ==
                      reference::one_hot(sliced_ids.to_vector<std::int64_t>(), 4),
                  "one_hot sliced indices with offset");
    const Tensor strided_gradient = Tensor::from_vector(values(30), {2, 5, 3}).transpose(1, 2);
    test::require(test::matches_f32(
                      embedding_backward(strided_gradient, ids, 4).to_vector<float>(),
                      reference::embedding_backward(strided_gradient.to_vector<float>(),
                                                     raw_indices, 4, 5)),
                  "embedding backward strided gradient");
}

void test_autograd() {
    Tensor weight = Tensor::from_vector(values(12), {4, 3});
    weight.set_requires_grad();
    Tensor ids = Tensor::from_vector(std::vector<std::int64_t>{2, 0, 2}, {3});
    const auto upstream_values = values(9);
    const Tensor upstream = Tensor::from_vector(upstream_values, {3, 3});
    const Tensor output = embedding(weight, ids);
    test::require(output.requires_grad(), "embedding records autograd");
    // Mutating the caller's index tensor after forward must not move saved gradients.
    ids.copy_from(Tensor::from_vector(std::vector<std::int64_t>{1, 1, 1}, {3}));
    sum(mul(output, upstream)).backward();
    const auto expected = reference::embedding_backward(upstream_values, {2, 0, 2}, 4, 3);
    test::require(test::matches_f32(weight.grad().to_vector<float>(), expected),
                  "embedding autograd weighted repeated indices and saved IDs");

    Tensor empty_weight = Tensor::ones({4, 3});
    empty_weight.set_requires_grad();
    sum(embedding(empty_weight, Tensor::empty({0}, DType::I64))).backward();
    test::require(empty_weight.grad().to_vector<float>() == std::vector<float>(12, 0.0f),
                  "embedding autograd empty output");

    Tensor scalar_weight = Tensor::ones({4, 3});
    scalar_weight.set_requires_grad();
    const Tensor scalar_id = Tensor::from_vector(std::vector<std::int64_t>{1}, {});
    sum(embedding(scalar_weight, scalar_id)).backward();
    test::require(scalar_weight.grad().to_vector<float>() ==
                      reference::embedding_backward(std::vector<float>(3, 1.0f), {1}, 4, 3),
                  "embedding autograd scalar index");

    const Tensor view_ids = Tensor::from_vector(std::vector<std::int64_t>{2, 0, 2}, {3});
    Tensor transposed_base = Tensor::ones({3, 4});
    transposed_base.set_requires_grad();
    sum(mul(embedding(transposed_base.transpose(0, 1), view_ids), upstream)).backward();
    std::vector<float> transposed_expected(12);
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            transposed_expected[row * 4 + column] = expected[column * 3 + row];
        }
    }
    test::require(transposed_base.grad().shape() == transposed_base.shape() &&
                      test::matches_f32(transposed_base.grad().to_vector<float>(),
                                         transposed_expected),
                  "embedding backward through transposed weight");

    Tensor expanded_base = Tensor::ones({1, 3});
    expanded_base.set_requires_grad();
    sum(mul(embedding(expanded_base.expand({4, 3}), view_ids), upstream)).backward();
    std::vector<float> expanded_expected(3, 0.0f);
    for (std::size_t i = 0; i < upstream_values.size(); ++i) {
        expanded_expected[i % 3] += upstream_values[i];
    }
    test::require(expanded_base.grad().shape() == expanded_base.shape() &&
                      test::matches_f32(expanded_base.grad().to_vector<float>(), expanded_expected),
                  "embedding backward through expanded weight");

    Tensor sliced_base = Tensor::ones({6, 3});
    sliced_base.set_requires_grad();
    sum(mul(embedding(sliced_base.slice(0, 1, 6, 2), view_ids), upstream)).backward();
    test::require(sliced_base.grad().shape() == sliced_base.shape() &&
                      test::matches_f32(
                          sliced_base.grad().to_vector<float>(),
                          reference::embedding_backward(upstream_values, {5, 1, 5}, 6, 3)),
                  "embedding backward through stepped weight slice");
}

void test_errors() {
    using E = std::invalid_argument;
    const Tensor ids = Tensor::from_vector(std::vector<std::int64_t>{0, 2}, {2});
    const Tensor weight = Tensor::ones({4, 3});
    const Tensor gradient = Tensor::ones({2, 3});
    const Tensor negative = Tensor::from_vector(std::vector<std::int64_t>{-1}, {1});
    const Tensor too_large = Tensor::from_vector(std::vector<std::int64_t>{4}, {1});
    test::require_throws<E>([&] { one_hot(Tensor{}, 4); }, "one_hot undefined");
    test::require_throws<E>([&] { one_hot(Tensor::ones({2}), 4); }, "one_hot F32 indices");
    test::require_throws<E>([&] { one_hot(Tensor::ones({2}, DType::I32), 4); },
                            "one_hot I32 indices");
    test::require_throws<E>([&] { one_hot(ids, 0); }, "one_hot zero classes");
    test::require_throws<E>([&] { one_hot(ids, -1); }, "one_hot negative classes");
    test::require_throws<E>([&] { one_hot(negative, 4); }, "one_hot negative index");
    test::require_throws<E>([&] { one_hot(too_large, 4); }, "one_hot upper bound");
    test::require_throws<E>([&] { embedding(Tensor{}, ids); }, "embedding undefined weight");
    test::require_throws<E>([&] { embedding(weight, Tensor{}); }, "embedding undefined indices");
    test::require_throws<E>([&] { embedding(Tensor::ones({4}), ids); }, "embedding weight rank");
    test::require_throws<E>([&] { embedding(Tensor::ones({4, 3}, DType::I64), ids); },
                            "embedding weight dtype");
    test::require_throws<E>([&] { embedding(weight, Tensor::ones({2}, DType::I32)); },
                            "embedding indices dtype");
    test::require_throws<E>([&] { embedding(weight, negative); }, "embedding negative index");
    test::require_throws<E>([&] { embedding(weight, too_large); }, "embedding upper bound");
    test::require_throws<E>([&] { embedding(Tensor::empty({4, 0}), too_large); },
                            "embedding validates indices at zero width");
    test::require_throws<E>([&] { embedding_backward(Tensor{}, ids, 4); },
                            "embedding backward undefined gradient");
    test::require_throws<E>([&] { embedding_backward(gradient, Tensor{}, 4); },
                            "embedding backward undefined indices");
    test::require_throws<E>([&] { embedding_backward(gradient, ids, -1); },
                            "embedding backward negative rows");
    test::require_throws<E>([&] { embedding_backward(gradient, ids, 0); },
                            "embedding backward zero rows with nonempty indices");
    test::require_throws<E>([&] { embedding_backward(Tensor::ones({6}), ids, 4); },
                            "embedding backward rank");
    test::require_throws<E>([&] { embedding_backward(Tensor::ones({3, 3}), ids, 4); },
                            "embedding backward leading shape");
    test::require_throws<E>([&] { embedding_backward(Tensor::ones({2, 3}, DType::I32), ids, 4); },
                            "embedding backward dtype");
    test::require_throws<E>([&] { embedding_backward(gradient, ids, 2); },
                            "embedding backward index bound");
    test::require_throws<std::runtime_error>(
        [&] { one_hot(Tensor::empty({0}, DType::I64), 4, Execution::GPU); },
        "empty one_hot still checks execution support");
    test::require_throws<std::runtime_error>(
        [&] { embedding(weight, Tensor::empty({0}, DType::I64), Execution::GPU); },
        "empty embedding still checks execution support");
}

} // namespace

int main() {
    return test::run("indexing_test", [] {
        test_kernels();
        test_api();
        test_autograd();
        test_errors();
    });
}
