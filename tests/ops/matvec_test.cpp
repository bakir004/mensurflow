#include "backends/cpu/blas.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/sequential.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

std::vector<float> input(std::int64_t n) {
    std::vector<float> values(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        values[static_cast<std::size_t>(i)] = static_cast<float>((i * 7) % 19 - 9) / 4.0f;
    }
    return values;
}

void check_close(const std::vector<float>& got, const std::vector<float>& want,
                 const std::string& label) {
    test::require(got.size() == want.size(), label + ": size");
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(std::abs(got[i] - want[i]) <= 2e-5f, label + ": entry " +
                                                           std::to_string(i));
    }
}

std::vector<float> reference(const std::vector<float>& a, const std::vector<float>& x,
                             std::int64_t m, std::int64_t k) {
    std::vector<float> result(static_cast<std::size_t>(m));
    for (std::int64_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < k; ++j) {
            sum += static_cast<double>(a[static_cast<std::size_t>(i * k + j)]) * x[j];
        }
        result[static_cast<std::size_t>(i)] = static_cast<float>(sum);
    }
    return result;
}

void test_kernels() {
    const int original = cpu::max_threads();
    const std::int64_t m = 37, k = 29;
    const auto a = input(m * k);
    const auto x = input(k);
    const auto want = reference(a, x, m, k);
    std::vector<float> sequential(static_cast<std::size_t>(m), 1.0f);
    cpu::sequential::matvec_f32(a.data(), x.data(), sequential.data(), m, k);
    check_close(sequential, want, "sequential matvec");
    for (int threads : {1, 2, original}) {
        cpu::set_max_threads(threads);
        std::vector<float> parallel(static_cast<std::size_t>(m), 1.0f);
        cpu::matvec_f32(a.data(), x.data(), parallel.data(), m, k);
        check_close(parallel, want, "parallel matvec");
    }
    cpu::set_max_threads(original);
}

void test_kernel_edges() {
    float out[3] = {1, 1, 1};
    cpu::matvec_f32(nullptr, nullptr, out, 3, 0);
    test::require(out[0] == 0 && out[1] == 0 && out[2] == 0, "k == 0 gives zeros");
    cpu::matvec_f32(nullptr, nullptr, nullptr, 0, 3);
    float value = 1;
    test::require_throws<std::invalid_argument>(
        [&] { cpu::matvec_f32(&value, &value, &value, -1, 1); }, "negative m");
    test::require_throws<std::invalid_argument>(
        [&] { cpu::matvec_f32(nullptr, &value, &value, 1, 1); }, "null matrix");
}

bool supported(Device device) {
    return detail::Registry::instance().dispatcher().find(
               {detail::OpId::Matvec, device.type, DType::F32}) != nullptr;
}

void test_public(Device device) {
    const std::int64_t m = 4, k = 3;
    const auto a = input(m * k);
    const auto x = input(k);
    const Tensor result =
        matvec(Tensor::from_vector(a, {m, k}, device), Tensor::from_vector(x, {k}, device));
    check_close(result.to_vector<float>(), reference(a, x, m, k), "public matvec");
    test::require(result.shape() == (Shape{m}) && result.is_contiguous(), "matvec output shape");

    const Tensor stored_a = Tensor::from_vector(input(6 * k), {6, k}, device);
    const Tensor view_a = stored_a.slice(0, 1, 5, 2);
    const Tensor stored_x = Tensor::from_vector(input(2 * k), {2 * k}, device);
    const Tensor view_x = stored_x.slice(0, 0, 2 * k, 2);
    test::require(!view_a.is_contiguous() && !view_x.is_contiguous(), "matvec views");
    check_close(matvec(view_a, view_x).to_vector<float>(),
                reference(view_a.to_vector<float>(), view_x.to_vector<float>(), 2, k),
                "matvec strided inputs");

    test::require_throws<std::invalid_argument>(
        [&] { matvec(Tensor::from_vector(a, {m, k}, device), Tensor::from_vector(x, {k + 1}, device)); },
        "matvec mismatched dimensions");
    test::require_throws<detail::NotImplemented>(
        [&] { matvec(Tensor::empty({m, k}, DType::I32, device), Tensor::empty({k}, DType::I32, device)); },
        "matvec unsupported dtype");
}

} // namespace

int main() {
    try {
        test_kernels();
        test_kernel_edges();
        for (Device device : list_devices()) {
            if (supported(device)) {
                test_public(device);
            } else {
                test::require_throws<detail::NotImplemented>(
                    [&] { matvec(Tensor::empty({2, 2}, DType::F32, device),
                                 Tensor::empty({2}, DType::F32, device)); },
                    "unsupported matvec");
            }
        }
        std::cout << "matvec_test passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "matvec_test FAILED: " << e.what() << '\n';
        return 1;
    }
}
