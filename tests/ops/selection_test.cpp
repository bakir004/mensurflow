// F32 comparisons -> Bool and Bool/F32 selection: independent broadcasting, byte/bit contracts,
// strided views, dispatch/errors and invariance across CPU thread counts.
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/threads.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/selection.hpp"
#include "test_util.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

using Byte = std::uint8_t;
using CompareKernel = void (*)(const float*, const float*, Byte*, std::int64_t);

struct Comparison {
    const char* name;
    detail::OpId id;
    Tensor (*apply)(const Tensor&, const Tensor&, Execution);
    CompareKernel kernel;
    CompareKernel reference;
    bool (*predicate)(float, float);
};

const Comparison comparisons[] = {
    {"eq", detail::OpId::Eq, eq, cpu::eq_f32, reference::eq,
     [](float a, float b) { return a == b; }},
    {"ne", detail::OpId::Ne, ne, cpu::ne_f32, reference::ne,
     [](float a, float b) { return a != b; }},
    {"lt", detail::OpId::Lt, lt, cpu::lt_f32, reference::lt,
     [](float a, float b) { return a < b; }},
    {"le", detail::OpId::Le, le, cpu::le_f32, reference::le,
     [](float a, float b) { return a <= b; }},
    {"gt", detail::OpId::Gt, gt, cpu::gt_f32, reference::gt,
     [](float a, float b) { return a > b; }},
    {"ge", detail::OpId::Ge, ge, cpu::ge_f32, reference::ge,
     [](float a, float b) { return a >= b; }}};

float from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

Tensor mask_tensor(const std::vector<Byte>& values, Shape shape, Device device = {}) {
    test::require(values.size() == reference::selection::count(shape), "test mask shape");
    return Tensor::from_host(values.data(), shape, DType::Bool, device);
}

std::vector<Byte> mask_values(const Tensor& tensor) {
    test::require(tensor.dtype() == DType::Bool, "expected Bool tensor");
    std::vector<Byte> values(static_cast<std::size_t>(tensor.numel()));
    tensor.copy_to_host(values.data());
    return values;
}

std::vector<float> make_values(std::size_t n, int shift = 0) {
    const std::array<float, 12> values = {
        -3.0f, -1.0f, -0.0f, 0.0f, 1.0f, 3.0f,
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        from_bits(0x7fc12345), from_bits(0xffc23456),
        std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()};
    std::vector<float> output(n);
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = values[(i + static_cast<std::size_t>(shift)) % values.size()];
    }
    return output;
}

std::vector<Byte> make_mask(std::size_t n) {
    const Byte values[] = {0, 1, 2, 255, 128, 0, 0, 7};
    std::vector<Byte> output(n);
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = values[i % 8];
    }
    return output;
}

void check_comparison(const Comparison& op, const Tensor& a, const Tensor& b,
                      Execution execution = Execution::CPU) {
    const auto a_before = a.to_vector<float>(), b_before = b.to_vector<float>();
    const auto expected = reference::selection::broadcast_compare(a_before, a.shape(), b_before,
                                                                 b.shape(), op.predicate);
    const Tensor output = op.apply(a, b, execution);
    const std::string label = std::string(op.name) + " API";
    test::require(output.shape() == expected.shape && output.dtype() == DType::Bool &&
                      output.device() == a.device(), label + " metadata");
    test::require(output.is_contiguous() && !output.shares_storage(a) && !output.shares_storage(b),
                  label + " fresh contiguous output");
    test::require(mask_values(output) == expected.values, label + " independent broadcast values");
    test::require(!output.requires_grad() && !output.has_grad(), label + " has no autograd");
    test::require(same_bits(a_before, a.to_vector<float>()) &&
                      same_bits(b_before, b.to_vector<float>()), label + " preserves inputs");
}

void check_where(const Tensor& condition, const Tensor& x, const Tensor& y,
                 Execution execution = Execution::CPU) {
    const auto condition_before = mask_values(condition);
    const auto x_before = x.to_vector<float>(), y_before = y.to_vector<float>();
    const auto expected = reference::selection::broadcast_where(condition_before, condition.shape(),
                                                               x_before, x.shape(), y_before,
                                                               y.shape());
    const Tensor output = where(condition, x, y, execution);
    test::require(output.shape() == expected.shape && output.dtype() == DType::F32 &&
                      output.device() == x.device(), "where API metadata");
    test::require(output.is_contiguous() && !output.shares_storage(condition) &&
                      !output.shares_storage(x) && !output.shares_storage(y),
                  "where fresh contiguous output");
    test::require(same_bits(output.to_vector<float>(), expected.values),
                  "where independent bit-exact broadcast values");
    test::require(!output.requires_grad() && !output.has_grad(), "where has no autograd");
    test::require(mask_values(condition) == condition_before &&
                      same_bits(x.to_vector<float>(), x_before) &&
                      same_bits(y.to_vector<float>(), y_before), "where preserves inputs");
}

void test_registration_and_ieee_comparisons() {
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = from_bits(0x7fc12345);
    const Tensor a = Tensor::from_vector<float>({-infinity, -1, -0.0f, 0.0f, 1, infinity, nan, nan}, {8});
    const Tensor b = Tensor::from_vector<float>({-infinity, 0, 0.0f, -0.0f, 0, infinity, 0, nan}, {8});
    const std::vector<Byte> expected[] = {
        {1, 0, 1, 1, 0, 1, 0, 0}, {0, 1, 0, 0, 1, 0, 1, 1},
        {0, 1, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0}, {1, 0, 1, 1, 1, 1, 0, 0}};
    std::size_t index = 0;
    for (const auto& op : comparisons) {
        test::require(detail::Registry::instance().dispatcher().find(
                          {op.id, DeviceType::CPU, DType::F32}) != nullptr,
                      std::string(op.name) + " registered for CPU F32");
        test::require(std::string(detail::op_name(op.id)) == op.name, "comparison operation name");
        test::require(mask_values(op.apply(a, b, Execution::CPU)) == expected[index++],
                      std::string(op.name) + " IEEE comparison truth table");
    }
    // Internal order is {x,y,condition}; dispatch therefore keys where by the F32 branch dtype.
    test::require(detail::Registry::instance().dispatcher().find(
                      {detail::OpId::Where, DeviceType::CPU, DType::F32}) != nullptr,
                  "where registered for CPU F32");
    test::require(std::string(detail::op_name(detail::OpId::Where)) == "where", "where name");
}

void test_comparison_api() {
    const std::pair<Shape, Shape> shapes[] = {
        {{}, {}}, {{}, {2, 3}}, {{2, 3}, {}}, {{2, 3}, {3}}, {{2, 3}, {2, 1}},
        {{2, 1, 3}, {4, 1}}, {{257, 129}, {129}}, {{0}, {1}}, {{2, 0, 3}, {1, 1, 3}}};
    for (const auto& pair : shapes) {
        const Tensor a = Tensor::from_vector(make_values(reference::selection::count(pair.first)),
                                             pair.first);
        const Tensor b = Tensor::from_vector(make_values(reference::selection::count(pair.second), 3),
                                             pair.second);
        for (const auto& op : comparisons) {
            check_comparison(op, a, b);
            check_comparison(op, a, b, Execution::Auto);
        }
    }
    const Tensor matrix = Tensor::from_vector<float>({0, 1, 2, 3, 4, 5}, {2, 3});
    const Tensor transposed = matrix.transpose(0, 1);
    const Tensor expected = Tensor::from_vector<float>({0, 3, 1, 4, 2, 5}, {3, 2});
    test::require(mask_values(eq(transposed, expected)) == std::vector<Byte>(6, 1),
                  "comparison transposed logical order");
    for (const auto& op : comparisons) {
        check_comparison(op, transposed, expected);
        check_comparison(op, matrix.slice(1, 0, 3, 2), Tensor::full({}, 2.0));
        check_comparison(op, matrix.slice(0, 1, 2), Tensor::full({3}, 4.0));
        check_comparison(op, matrix.slice(0, 0, 1).expand({4, 3}), Tensor::full({4, 1}, 1.0));
    }
}

void test_where_api() {
    struct Shapes { Shape condition, x, y; };
    const Shapes cases[] = {
        {{}, {}, {}}, {{}, {2, 3}, {}}, {{2, 1, 3}, {1, 4, 1}, {3}},
        {{3}, {2, 3}, {2, 1}}, {{1}, {2, 0, 3}, {1, 1, 3}}, {{0}, {}, {1}}};
    for (const auto& shapes : cases) {
        const Tensor condition = mask_tensor(make_mask(reference::selection::count(shapes.condition)),
                                              shapes.condition);
        const Tensor x = Tensor::from_vector(make_values(reference::selection::count(shapes.x)), shapes.x);
        const Tensor y = Tensor::from_vector(make_values(reference::selection::count(shapes.y), 3), shapes.y);
        check_where(condition, x, y);
        check_where(condition, x, y, Execution::Auto);
    }
    const Tensor condition = mask_tensor({0, 1, 2, 255, 0, 128}, {2, 3});
    const Tensor x = Tensor::from_vector<float>({0, -0.0f, 2, 3, 4, 5}, {2, 3});
    const Tensor y = Tensor::from_vector<float>({-0.0f, 10, 20, 30, 40, 50}, {2, 3});
    check_where(condition.transpose(0, 1), x.transpose(0, 1), y.transpose(0, 1));
    const auto transposed = where(condition.transpose(0, 1), x.transpose(0, 1), y.transpose(0, 1));
    test::require(same_bits(transposed.to_vector<float>(), {-0.0f, 3, -0.0f, 40, 2, 5}),
                  "where transposed logical order and signed zero");
    check_where(condition.slice(0, 1, 2), x.slice(0, 1, 2), y.slice(0, 1, 2));
    check_where(condition.slice(1, 0, 3, 2), x.slice(1, 0, 3, 2), y.slice(1, 0, 3, 2));
    check_where(condition.slice(0, 0, 1).expand({4, 3}), x.slice(0, 0, 1), Tensor::full({}, -1.0));
    check_where(mask_tensor({255}, {}), x, x);

    const float nan = from_bits(0x7fc12345), negative_nan = from_bits(0xffc23456);
    const Tensor special_x = Tensor::from_vector<float>({nan, 1, -0.0f, 0.0f, 1, nan}, {6});
    const Tensor special_y = Tensor::from_vector<float>({1, negative_nan, 0.0f, -0.0f, negative_nan, 2}, {6});
    const Tensor special_mask = mask_tensor({1, 0, 255, 0, 2, 0}, {6});
    test::require(same_bits(where(special_mask, special_x, special_y).to_vector<float>(),
                            {nan, negative_nan, -0.0f, -0.0f, 1, 2}),
                  "where selects payload bits without evaluating the unused branch");

    const Tensor values = Tensor::from_vector<float>({-2, 0, 3}, {3});
    const Tensor zero = Tensor::full({}, 0.0);
    test::require(same_bits(where(gt(values, zero), values, zero).to_vector<float>(), {0, 0, 3}),
                  "comparison-produced Bool mask feeds where directly");

    Tensor tracked_x = Tensor::full({3}, -1.0), tracked_y = Tensor::full({3}, 1.0);
    tracked_x.set_requires_grad();
    tracked_y.set_requires_grad();
    for (const auto& op : comparisons) {
        check_comparison(op, tracked_x, tracked_y);
    }
    check_where(mask_tensor({0, 1, 255}, {3}), tracked_x, tracked_y);
    test::require(!tracked_x.has_grad() && !tracked_y.has_grad(), "selection does not accumulate gradients");
}

void test_cpu_threads_and_aliasing() {
    const int original_threads = cpu::max_threads();
    const std::int64_t sizes[] = {0, 1, 7, 8, 9, 17, 31, 32, 33, cpu::parallel_threshold - 1,
                                  cpu::parallel_threshold, cpu::parallel_threshold + 1, 100003};
    for (auto n : sizes) {
        const auto count = static_cast<std::size_t>(n);
        const auto x = make_values(count), y = make_values(count, 5);
        const auto condition = make_mask(count);
        for (const auto& op : comparisons) {
            std::vector<Byte> expected(count);
            op.reference(x.data(), y.data(), expected.data(), n);
            op.kernel(nullptr, nullptr, nullptr, 0);
            for (int threads : {1, 2, 3, original_threads}) {
                cpu::set_max_threads(threads);
                std::vector<Byte> actual(count + 2, Byte{0xa5});
                op.kernel(x.data(), y.data(), actual.data() + 1, n);
                test::require(actual.front() == 0xa5 && actual.back() == 0xa5,
                              std::string(op.name) + " output guards");
                test::require(std::vector<Byte>(actual.begin() + 1, actual.end() - 1) == expected,
                              std::string(op.name) + " reference and thread agreement");
            }
        }
        std::vector<float> expected(count);
        reference::where(condition.data(), x.data(), y.data(), expected.data(), n);
        cpu::where_f32(nullptr, nullptr, nullptr, nullptr, 0);
        for (int threads : {1, 2, 3, original_threads}) {
            cpu::set_max_threads(threads);
            std::vector<float> actual(count + 2, 1234.5f);
            cpu::where_f32(condition.data(), x.data(), y.data(), actual.data() + 1, n);
            test::require(actual.front() == 1234.5f && actual.back() == 1234.5f, "where output guards");
            test::require(same_bits(std::vector<float>(actual.begin() + 1, actual.end() - 1), expected),
                          "where bit-exact reference and thread agreement");
            auto alias_x = x, alias_y = y;
            cpu::where_f32(condition.data(), alias_x.data(), y.data(), alias_x.data(), n);
            cpu::where_f32(condition.data(), x.data(), alias_y.data(), alias_y.data(), n);
            test::require(same_bits(alias_x, expected) && same_bits(alias_y, expected),
                          "where supports either exact in-place branch");
            auto both = x;
            cpu::where_f32(condition.data(), both.data(), both.data(), both.data(), n);
            test::require(same_bits(both, x), "where supports identical branches and output");
        }
    }
    cpu::set_max_threads(original_threads);
}

void test_errors() {
    const Tensor x = Tensor::full({2}, 1.0), y = Tensor::full({2}, 2.0);
    const Tensor condition = mask_tensor({0, 1}, {2});
    const Tensor incompatible = Tensor::full({3}, 1.0);
    const Tensor ints = Tensor::from_vector<std::int32_t>({1, 2}, {2});
    for (const auto& op : comparisons) {
        for (bool first : {false, true}) {
            test::require_throws<std::invalid_argument>(
                [&] { op.apply(first ? Tensor{} : x, first ? y : Tensor{}, Execution::CPU); },
                std::string(op.name) + " undefined operand");
        }
        test::require_throws<std::invalid_argument>(
            [&] { op.apply(x, incompatible, Execution::CPU); }, std::string(op.name) + " shapes");
        test::require_throws<std::invalid_argument>(
            [&] { op.apply(x, ints, Execution::CPU); }, std::string(op.name) + " mismatched dtype");
    }
    test::require_throws<std::invalid_argument>([&] { where(Tensor{}, x, y); }, "undefined condition");
    test::require_throws<std::invalid_argument>([&] { where(condition, Tensor{}, y); }, "undefined x");
    test::require_throws<std::invalid_argument>([&] { where(condition, x, Tensor{}); }, "undefined y");
    test::require_throws<std::invalid_argument>([&] { where(x, x, y); }, "condition must be Bool");
    test::require_throws<std::invalid_argument>(
        [&] { where(Tensor::empty({2}, DType::U8), x, y); }, "U8 is not a Bool condition");
    test::require_throws<std::invalid_argument>([&] { where(condition, x, ints); }, "branch dtype mismatch");
    test::require_throws<std::invalid_argument>([&] { where(condition, x, incompatible); }, "branch shapes");
    test::require_throws<std::invalid_argument>(
        [&] { where(mask_tensor({0, 1, 0}, {3}), x, y); }, "condition shape");
    for (const Shape& shape : {Shape{2}, Shape{0}}) {
        const Tensor valid = Tensor::empty(shape, DType::F32);
        const Tensor mask = mask_tensor(make_mask(reference::selection::count(shape)), shape);
        for (auto dtype : {DType::I32, DType::I64, DType::F16, DType::BF16, DType::U8, DType::Bool}) {
            const Tensor unsupported = Tensor::empty(shape, dtype);
            for (const auto& op : comparisons) {
                test::require_throws<detail::NotImplemented>(
                    [&] { op.apply(unsupported, unsupported, Execution::CPU); },
                    std::string(op.name) + " unsupported dtype, including empty");
                test::require_throws<std::invalid_argument>(
                    [&] { op.apply(valid, unsupported, Execution::CPU); },
                    std::string(op.name) + " dtype mismatch, including empty");
            }
            test::require_throws<detail::NotImplemented>(
                [&] { where(mask, unsupported, unsupported, Execution::CPU); },
                "where unsupported branch dtype, including empty");
            test::require_throws<std::invalid_argument>(
                [&] { where(mask, valid, unsupported, Execution::CPU); },
                "where branch dtype mismatch, including empty");
        }
    }
}

void test_optional_gpu_paths() {
    const Tensor x = Tensor::from_vector<float>({-1, -0.0f, 2}, {3});
    const Tensor y = Tensor::from_vector<float>({0, 0, 1}, {3});
    const Tensor condition = mask_tensor({0, 2, 255}, {3});
    for (Device device : list_devices()) {
        if (device.type == DeviceType::CPU) {
            continue;
        }
        const Tensor gx = x.to(device), gy = y.to(device), gc = condition.to(device);
        for (const auto& op : comparisons) {
            test::require_throws<std::invalid_argument>(
                [&] { op.apply(x, gy, Execution::CPU); }, std::string(op.name) + " mixed devices");
            test::require_throws<detail::NotImplemented>(
                [&] { op.apply(gx, gy, Execution::GPU); }, std::string(op.name) + " unsupported GPU");
            check_comparison(op, gx, gy, Execution::CPU);
            check_comparison(op, gx, gy, Execution::Auto);
            const Tensor empty = Tensor::empty({0}, DType::F32, device);
            test::require_throws<detail::NotImplemented>(
                [&] { op.apply(empty, empty, Execution::GPU); }, std::string(op.name) + " empty GPU");
        }
        test::require_throws<std::invalid_argument>([&] { where(gc, x, y); }, "mixed condition device");
        test::require_throws<std::invalid_argument>([&] { where(condition, gx, y); }, "mixed x device");
        test::require_throws<std::invalid_argument>([&] { where(condition, x, gy); }, "mixed y device");
        test::require_throws<detail::NotImplemented>([&] { where(gc, gx, gy, Execution::GPU); },
                                                  "where unsupported GPU");
        check_where(gc, gx, gy, Execution::CPU);
        check_where(gc, gx, gy, Execution::Auto);
        const Tensor empty = Tensor::empty({0}, DType::F32, device);
        const Tensor empty_mask = Tensor::empty({0}, DType::Bool, device);
        test::require_throws<detail::NotImplemented>(
            [&] { where(empty_mask, empty, empty, Execution::GPU); }, "where empty GPU");
    }
}

} // namespace

int main() {
    return test::run("selection_test", [] {
        test_registration_and_ieee_comparisons();
        test_comparison_api();
        test_where_api();
        test_cpu_threads_and_aliasing();
        test_errors();
        test_optional_gpu_paths();
    });
}
