// matmul with different tile sizes: the measurements the defaults are chosen from, and what an
// auto-tuner would search.
//   metal  (Metal builds) mensurflow::matmul on the GPU through the public API, for each kernel
//          and tile configuration of backends/metal/gemm.hpp. A call allocates its output and
//          waits for the GPU, as every matmul does. "vs MPS" divides MPS's median time by this row's.
//          The GPU is kept busy for a second first so that every row runs at full clock.
//   cpu    matmul_f32_parallel on all threads for several MatmulBlocking tile sizes, called
//          directly into preallocated matrices. "vs MPS" there means vs the default blocking.
// Every configuration is checked against the first one before it is timed (the inputs are exact
// in float32, so they must agree exactly on the CPU and to float rounding on the GPU).
// GFLOP/s counts 2 m n k.
//
//   matmul_tile_benchmark
#include "bench.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#ifdef MENSURFLOW_BENCH_METAL
#include "backends/metal/gemm.hpp"
#endif

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace mf = mensurflow;

namespace {

struct Shape3 {
    std::int64_t m, n, k;
};

std::string label(const Shape3& s) {
    return std::to_string(s.m) + "x" + std::to_string(s.n) + "x" + std::to_string(s.k);
}

double flops(const Shape3& s) {
    return 2.0 * static_cast<double>(s.m) * static_cast<double>(s.n) * static_cast<double>(s.k);
}

// Multiples of 0.25 in [-4, 4]: exact products and sums for these sizes.
std::vector<float> exact_input(std::int64_t n, int seed) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>(static_cast<int>((i * 37 + static_cast<std::size_t>(seed)) % 17) -
                                  8) /
               4.0f;
    }
    return v;
}

#ifdef MENSURFLOW_BENCH_METAL

struct Named {
    const char* name;
    mf::metal::MetalGemmConfig config;
};

mf::metal::MetalGemmConfig make(std::int64_t kernel, std::int64_t bm, std::int64_t bn,
                                std::int64_t bk, std::int64_t tm, std::int64_t tn,
                                std::int64_t sg_m, std::int64_t sg_n) {
    mf::metal::MetalGemmConfig c;
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

void metal_sweep(const bench::Config& config, mf::Device device) {
    const Named configs[] = {
        {"MPS", make(2, 0, 0, 0, 0, 0, 0, 0)},
        {"simd 32x64x16 sg2x2 (default)", mf::metal::MetalGemmConfig{}},
        {"simd 32x32x16 sg2x2", make(1, 32, 32, 16, 0, 0, 2, 2)},
        {"simd 64x32x16 sg2x2", make(1, 64, 32, 16, 0, 0, 2, 2)},
        {"simd 32x64x32 sg2x2", make(1, 32, 64, 32, 0, 0, 2, 2)},
        {"simd 16x64x16 sg1x2", make(1, 16, 64, 16, 0, 0, 1, 2)},
        {"simd 64x64x16 sg4x2", make(1, 64, 64, 16, 0, 0, 4, 2)},
        {"simd 64x64x32 sg2x4", make(1, 64, 64, 32, 0, 0, 2, 4)},
        {"simd 32x128x16 sg2x4", make(1, 32, 128, 16, 0, 0, 2, 4)},
        {"simd 64x64x16 sg2x2 (16 acc)", make(1, 64, 64, 16, 0, 0, 2, 2)},
        {"tiled 64x64x16 t4x4", make(0, 64, 64, 16, 4, 4, 1, 1)},
        {"tiled 128x64x8 t8x4", make(0, 128, 64, 8, 8, 4, 1, 1)},
        {"tiled 32x32x16 t2x2", make(0, 32, 32, 16, 2, 2, 1, 1)},
    };
    const mf::metal::MetalGemmConfig original = mf::metal::metal_gemm_config();
    bench::Table table("GFLOP/s", 1e9, "vs MPS", "");
    table.section("metal: matmul on " + mf::to_string(device) + ", through the public API");
    const Shape3 shapes[] = {{256, 256, 256},    {512, 512, 512},    {1024, 1024, 1024},
                             {2048, 2048, 2048}, {4096, 4096, 4096}, {4096, 1024, 1024},
                             {1000, 1000, 1000}, {64, 4096, 4096}};
    // The GPU raises its clock only after some time under load; one warm-up call is not enough,
    // and without this the first configuration timed (MPS) would run at a lower clock.
    {
        const auto warm =
            mf::Tensor::from_vector(exact_input(1024 * 1024, 3), {1024, 1024}, device);
        const auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
               1.0) {
            (void)mf::matmul(warm, warm, mf::Execution::GPU).to_vector<float>();
        }
    }
    for (const Shape3& s : shapes) {
        const auto av = exact_input(s.m * s.k, 1);
        const auto bv = exact_input(s.k * s.n, 2);
        const mf::Tensor a = mf::Tensor::from_vector(av, {s.m, s.k}, device);
        const mf::Tensor b = mf::Tensor::from_vector(bv, {s.k, s.n}, device);
        mf::Tensor out;
        auto run = [&] { out = mf::matmul(a, b, mf::Execution::GPU); };
        auto observe = [&] { return static_cast<double>(out.to_vector<float>()[0]); };
        std::vector<float> expected;
        bench::Sample mps;
        for (const Named& named : configs) {
            mf::metal::set_metal_gemm_config(named.config);
            run();
            const std::vector<float> got = out.to_vector<float>();
            if (expected.empty()) {
                expected = got;
            } else {
                bench::check_close(expected.data(), got.data(), got.size(), 1e-5,
                                   label(s) + " " + named.name);
            }
            const bench::Sample sample = bench::measure(config, run, observe);
            if (named.config.kernel == 2) {
                mps = sample;
            }
            table.row(label(s), named.name, "gpu", sample, flops(s), &mps);
        }
        table.blank();
    }
    mf::metal::set_metal_gemm_config(original);
}

#endif

void cpu_sweep(const bench::Config& config) {
    struct Blocking {
        const char* name;
        cpu::MatmulBlocking blocking;
    };

    const Blocking blockings[] = {
        {"default band x 128 x 256", {0, 128, 256}}, {"band x 64 x 256", {0, 64, 256}},
        {"band x 256 x 256", {0, 256, 256}},         {"band x 128 x 512", {0, 128, 512}},
        {"band x 128 x 128", {0, 128, 128}},         {"band x 512 x 1024", {0, 512, 1024}},
        {"16 x 128 x 256", {16, 128, 256}},          {"64 x 128 x 256", {64, 128, 256}},
        {"16 x 256 x 512", {16, 256, 512}},          {"4 x 64 x 1024", {4, 64, 1024}},
    };
    const int threads = cpu::max_threads();
    bench::Table table("GFLOP/s", 1e9, "vs default", "");
    table.section("cpu: matmul_f32_parallel on " + std::to_string(threads) +
                  " threads, by tile size (rows x depth x cols)");
    const Shape3 shapes[] = {
        {512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 1024, 1024}};
    for (const Shape3& s : shapes) {
        const auto av = exact_input(s.m * s.k, 1);
        const auto bv = exact_input(s.k * s.n, 2);
        std::vector<float> c(static_cast<std::size_t>(s.m * s.n));
        std::vector<float> expected;
        bench::Sample baseline;
        for (const Blocking& b : blockings) {
            auto run = [&] {
                cpu::matmul_f32_parallel(av.data(), bv.data(), c.data(), s.m, s.n, s.k, b.blocking);
            };
            run();
            if (expected.empty()) {
                expected = c;
            } else {
                bench::check_bits(expected.data(), c.data(), c.size(), label(s) + " " + b.name);
            }
            auto observe = [&] { return bench::checksum(c.data(), c.size()); };
            const bench::Sample sample = bench::measure(config, run, observe);
            if (expected.size() == c.size() && &b == &blockings[0]) {
                baseline = sample;
            }
            table.row(label(s), b.name, std::to_string(threads), sample, flops(s), &baseline);
        }
        table.blank();
    }
}

} // namespace

int main() {
    const bench::Config config;
    bench::print_header("matmul by tile size, float32", config,
                        {"metal: each kernel and tile configuration through mensurflow::matmul; "
                         "vs MPS = MPS median / this row's",
                         "cpu: matmul_f32_parallel with each MatmulBlocking; vs default = the "
                         "default blocking's median / this row's"});
#ifdef MENSURFLOW_BENCH_METAL
    for (const mf::Device device : mf::list_devices()) {
        if (device.type == mf::DeviceType::Metal) {
            metal_sweep(config, device);
        }
    }
#endif
    cpu_sweep(config);
    return 0;
}
