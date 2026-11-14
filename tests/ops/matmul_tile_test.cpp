// Tests the tile sizes of matmul. CPU: matmul_f32_parallel gives the same bits with any
// MatmulBlocking, and the setter rejects bad ones. Metal (built with the Metal backend): every kernel
// with several tile configurations against a float64 reference, on sizes that are not multiples of
// the tiles and on views that start partway into their buffer, and the validation rules.
#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/blas.hpp"
#include "test_util.hpp"

#ifdef MENSURFLOW_TEST_METAL
#include "backends/metal/gemm.hpp"
#endif

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mensurflow;
namespace cpu = mensurflow::cpu;

namespace {

// Multiples of 0.25 in [-4, 4]: every product and partial sum below is exact in float32, so any
// order of additions gives the reference value.
std::vector<float> make_input(std::int64_t n, std::uint32_t seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t state = seed;
    for (auto& x : v) {
        state = state * 1664525u + 1013904223u;
        x = static_cast<float>(static_cast<int>(state >> 12) % 33 - 16) / 4;
    }
    return v;
}

struct Shape3 {
    std::int64_t m, n, k;
};

std::string name(const Shape3& s) {
    return std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k);
}

// Below, at and above the tile sizes used here (8 to 128), thin, wide, and 1.
const Shape3 shapes[] = {{1, 1, 1},     {1, 33, 7},     {33, 1, 9},     {7, 9, 5},
                         {8, 8, 8},     {16, 16, 16},   {63, 65, 31},   {64, 64, 64},
                         {65, 130, 17}, {129, 63, 200}, {200, 300, 50}, {256, 256, 256},
                         {3, 1000, 2},  {1000, 3, 40}};

void check(const std::vector<float>& got, const std::vector<float>& a, const std::vector<float>& b,
           const Shape3& s, const std::string& what) {
    const auto want = reference::matmul(a.data(), b.data(), s.m, s.n, s.k);
    for (std::size_t i = 0; i < got.size(); ++i) {
        test::require(std::abs(static_cast<double>(got[i]) - want.value[i]) <=
                          reference::matmul_tolerance(want.magnitude[i]),
                      what + ": entry " + std::to_string(i) + " got " + std::to_string(got[i]) +
                          ", want " + std::to_string(want.value[i]));
    }
}

void test_cpu_blocking() {
    const int original_threads = cpu::max_threads();
    const cpu::MatmulBlocking blockings[] = {{0, 128, 256},    {1, 1, 1},   {7, 3, 5},
                                             {64, 1000, 4096}, {2, 64, 16}, {0, 9, 33}};
    for (const Shape3& s : shapes) {
        const auto a = make_input(s.m * s.k, 1);
        const auto b = make_input(s.k * s.n, 2);
        std::vector<float> expected(static_cast<std::size_t>(s.m * s.n));
        cpu::matmul_f32_sequential(a.data(), b.data(), expected.data(), s.m, s.n, s.k);
        for (const cpu::MatmulBlocking& blocking : blockings) {
            for (int threads : {1, 3, original_threads}) {
                cpu::set_max_threads(threads);
                std::vector<float> got(expected.size(), 123.0f);
                cpu::matmul_f32_parallel(a.data(), b.data(), got.data(), s.m, s.n, s.k, blocking);
                test::require(test::matches_f32(got, expected),
                              "cpu blocking " + std::to_string(blocking.rows) + "/" +
                                  std::to_string(blocking.depth) + "/" +
                                  std::to_string(blocking.cols) + " " + name(s) + " threads=" +
                                  std::to_string(threads) + " differs from the sequential bits");
            }
        }
    }
    cpu::set_max_threads(original_threads);

    // The setter changes what the overload without a blocking uses.
    const cpu::MatmulBlocking original = cpu::matmul_blocking();
    cpu::set_matmul_blocking({3, 5, 7});
    const cpu::MatmulBlocking now = cpu::matmul_blocking();
    test::require(now.rows == 3 && now.depth == 5 && now.cols == 7, "set_matmul_blocking");
    const Shape3 s{37, 41, 43};
    const auto a = make_input(s.m * s.k, 3);
    const auto b = make_input(s.k * s.n, 4);
    std::vector<float> got(static_cast<std::size_t>(s.m * s.n));
    cpu::matmul_f32_parallel(a.data(), b.data(), got.data(), s.m, s.n, s.k);
    check(got, a, b, s, "cpu with the set blocking");
    test::require_throws<std::invalid_argument>([] { cpu::set_matmul_blocking({-1, 128, 256}); },
                                                "negative rows");
    test::require_throws<std::invalid_argument>([] { cpu::set_matmul_blocking({0, 0, 256}); },
                                                "zero depth");
    test::require_throws<std::invalid_argument>([] { cpu::set_matmul_blocking({0, 128, 0}); },
                                                "zero cols");
    cpu::set_matmul_blocking(original);
}

#ifdef MENSURFLOW_TEST_METAL

std::string name(const metal::MetalGemmConfig& c) {
    return "kernel " + std::to_string(c.kernel) + " bm=" + std::to_string(c.bm) +
           " bn=" + std::to_string(c.bn) + " bk=" + std::to_string(c.bk) +
           " tm=" + std::to_string(c.tm) + " tn=" + std::to_string(c.tn) +
           " sg=" + std::to_string(c.sg_m) + "x" + std::to_string(c.sg_n);
}

metal::MetalGemmConfig make(std::int64_t kernel, std::int64_t bm, std::int64_t bn, std::int64_t bk,
                            std::int64_t tm, std::int64_t tn, std::int64_t sg_m,
                            std::int64_t sg_n) {
    metal::MetalGemmConfig c;
    c.kernel = kernel;
    c.bm = bm;
    c.bn = bn;
    c.bk = bk;
    c.tm = tm;
    c.tn = tn;
    c.sg_m = sg_m;
    c.sg_n = sg_n;
    return c;
}

void test_metal_configs(Device device) {
    const metal::MetalGemmConfig original = metal::metal_gemm_config();
    const metal::MetalGemmConfig configs[] = {
        metal::MetalGemmConfig{},         // the default
        make(0, 64, 64, 16, 4, 4, 2, 2),  // tiled, 256 threads
        make(0, 32, 32, 8, 2, 2, 2, 2),   // tiled, small tiles
        make(0, 16, 16, 32, 1, 1, 2, 2),  // tiled, one output per thread
        make(0, 128, 64, 8, 8, 4, 2, 2),  // tiled, uneven
        make(1, 32, 32, 8, 4, 4, 1, 1),   // simdgroup, one simdgroup
        make(1, 128, 64, 16, 4, 4, 4, 2), // simdgroup, 8 simdgroups
        make(1, 64, 128, 32, 4, 4, 2, 4), // simdgroup, deep slices
        make(1, 8, 8, 8, 4, 4, 1, 1),     // simdgroup, a single 8x8 block
        make(2, 64, 64, 16, 4, 4, 2, 2),  // MPS
    };
    for (const auto& config : configs) {
        metal::set_metal_gemm_config(config);
        for (const Shape3& s : shapes) {
            const auto a = make_input(s.m * s.k, 5);
            const auto b = make_input(s.k * s.n, 6);
            const Tensor c = matmul(Tensor::from_vector(a, {s.m, s.k}, device),
                                    Tensor::from_vector(b, {s.k, s.n}, device), Execution::GPU);
            check(c.to_vector<float>(), a, b, s, name(config) + " " + name(s));
        }
        // Inputs that start partway into their buffer: rows 3.. of a taller matrix.
        const Shape3 s{45, 70, 33};
        const auto a = make_input((s.m + 3) * s.k, 7);
        const auto b = make_input((s.k + 5) * s.n, 8);
        const Tensor va = Tensor::from_vector(a, {s.m + 3, s.k}, device).slice(0, 3, 3 + s.m);
        const Tensor vb = Tensor::from_vector(b, {s.k + 5, s.n}, device).slice(0, 5, 5 + s.k);
        test::require(va.offset() > 0 && va.is_contiguous(), "a contiguous view with an offset");
        check(matmul(va, vb, Execution::GPU).to_vector<float>(),
              std::vector<float>(a.begin() + 3 * s.k, a.end()),
              std::vector<float>(b.begin() + 5 * s.n, b.end()), s, name(config) + " offset views");
    }
    metal::set_metal_gemm_config(original);

    // Rules from gemm.hpp; a rejected configuration leaves the current one in place.
    const metal::MetalGemmConfig bad[] = {
        make(3, 64, 64, 16, 4, 4, 2, 2),   // no such kernel
        make(0, 0, 64, 16, 4, 4, 2, 2),    // zero bm
        make(0, 64, 64, 16, 3, 4, 2, 2),   // tm does not divide bm
        make(0, 128, 128, 8, 1, 1, 2, 2),  // 16384 threads
        make(0, 64, 64, 8, 16, 8, 2, 2),   // 128 accumulators per thread
        make(0, 128, 128, 64, 8, 8, 2, 2), // 64 KB of threadgroup memory
        make(1, 64, 64, 12, 4, 4, 2, 2),   // bk not a multiple of 8
        make(1, 48, 64, 16, 4, 4, 4, 2),   // bm not a multiple of 8 * sg_m
        make(1, 64, 64, 16, 4, 4, 8, 8),   // 64 simdgroups
    };
    for (const auto& config : bad) {
        test::require_throws<std::invalid_argument>([&] { metal::set_metal_gemm_config(config); },
                                                    "rejects " + name(config));
    }
    const metal::MetalGemmConfig kept = metal::metal_gemm_config();
    test::require(kept.kernel == original.kernel && kept.bm == original.bm,
                  "a rejected configuration does not replace the current one");
}

#endif

} // namespace

int main() {
    return test::run("matmul_tile_test", [] {
        test_cpu_blocking();
#ifdef MENSURFLOW_TEST_METAL
        for (Device device : list_devices()) {
            if (device.type == DeviceType::Metal) {
                std::cout << "  tile configurations on " << to_string(device) << '\n';
                test_metal_configs(device);
            }
        }
#endif
    });
}
