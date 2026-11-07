// Batched GEMM: what batching and the two ways of threading it buy.
//   sequential     gemm_batched_f32_sequential: one thread, one matrix after another (the baseline).
//   over_batch     threads split the batch; each matrix runs on one thread.
//   within_matrix  matrices one after another, each split over all threads.
//   parallel       what gemm_strided_batched runs by default: one of the two, picked from the shapes.
//   matmul() loop  the public matmul called once per matrix, as a user without a batched op would
//                  (dispatch, allocation and the single parallel GEMM for every matrix).
// GFLOP/s counts 2 * batch * m * n * k. "vs seq" divides the sequential median time by this row's.
// Kernels are called directly into preallocated memory, except the matmul() loop.
//
//   gemm_batched_benchmark
#include "bench.hpp"

#include "backends/cpu/ops/gemm/gemm_batched.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace mf = mensurflow;

namespace {

struct Size {
    std::int64_t batch, m, n, k;
};

// Attention-like (many heads of 64-128), many tiny matrices, a handful of large ones, and a batch
// smaller than the thread count.
const Size sizes[] = {{1024, 8, 8, 8},
                      {256, 32, 32, 32},
                      {64, 128, 128, 64},
                      {16, 256, 256, 256},
                      {4, 512, 512, 512}};

} // namespace

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header(
        "CPU batched GEMM (gemm_strided_batched), float32", config,
        {"sequential = one thread; over_batch / within_matrix / parallel use N OpenMP threads",
         "matmul() loop = the public matmul once per matrix (what batching saves)",
         "rate is GFLOP/s for 2*batch*m*n*k; vs seq = sequential time / this time"});
    bench::Table table("GFLOP/s", 1e9, "vs seq", "");

    using Kernel = void (*)(const cpu::BatchedGemm&);
    const Kernel kernels[] = {cpu::gemm_batched_f32_over_batch, cpu::gemm_batched_f32_within_matrix,
                              cpu::gemm_batched_f32_parallel};
    const char* names[] = {"over_batch", "within_matrix", "parallel"};

    for (const Size& s : sizes) {
        const std::string label = std::to_string(s.batch) + "x" + std::to_string(s.m) + "x" +
                                  std::to_string(s.n) + "x" + std::to_string(s.k);
        table.section(label + " (batch x m x n x k)");
        const auto a_len = static_cast<std::size_t>(s.batch * s.m * s.k);
        const auto b_len = static_cast<std::size_t>(s.batch * s.k * s.n);
        const auto c_len = static_cast<std::size_t>(s.batch * s.m * s.n);
        bench::Arrays arrays(std::max(a_len, std::max(b_len, c_len)), 3);
        float* a = arrays[0];
        float* b = arrays[1];
        float* c = arrays[2];
        bench::fill_a(a, a_len);
        bench::fill_b(b, b_len);
        std::vector<std::int64_t> a_offsets, b_offsets;
        for (std::int64_t i = 0; i < s.batch; ++i) {
            a_offsets.push_back(i * s.m * s.k);
            b_offsets.push_back(i * s.k * s.n);
        }
        cpu::BatchedGemm g;
        g.a = a;
        g.a_offsets = a_offsets.data();
        g.rsa = s.k;
        g.csa = 1;
        g.b = b;
        g.b_offsets = b_offsets.data();
        g.rsb = s.n;
        g.csb = 1;
        g.c = c;
        g.batch = s.batch;
        g.m = s.m;
        g.n = s.n;
        g.k = s.k;
        const double flops = 2.0 * static_cast<double>(s.batch) * static_cast<double>(s.m) *
                             static_cast<double>(s.n) * static_cast<double>(s.k);

        cpu::set_max_threads(1);
        cpu::gemm_batched_f32_sequential(g);
        const std::vector<float> expected(c, c + c_len);
        auto observe = [&] { return bench::checksum(c, c_len); };
        const bench::Sample sequential =
            bench::measure(config, [&] { cpu::gemm_batched_f32_sequential(g); }, observe);
        table.row(label, "sequential", "1", sequential, flops, &sequential);

        for (int threads : thread_counts) {
            cpu::set_max_threads(threads);
            for (int v = 0; v < 3; ++v) {
                kernels[v](g);
                bench::check_close(expected.data(), c, c_len, 1e-5,
                                   std::string(names[v]) + " " + label +
                                       " threads=" + std::to_string(threads));
                const bench::Sample sample =
                    bench::measure(config, [&] { kernels[v](g); }, observe);
                table.row(label, names[v], std::to_string(threads), sample, flops, &sequential);
            }
        }

        // The public API, one matrix at a time, at all threads.
        cpu::set_max_threads(maximum_threads);
        std::vector<mf::Tensor> lefts, rights;
        for (std::int64_t i = 0; i < s.batch; ++i) {
            lefts.push_back(mf::Tensor::from_host(a + i * s.m * s.k, {s.m, s.k}, mf::DType::F32));
            rights.push_back(mf::Tensor::from_host(b + i * s.k * s.n, {s.k, s.n}, mf::DType::F32));
        }
        double loop_checksum = 0.0;
        auto loop = [&] {
            loop_checksum = 0.0;
            for (std::size_t i = 0; i < lefts.size(); ++i) {
                const mf::Tensor out = mf::matmul(lefts[i], rights[i], mf::Execution::CPU);
                loop_checksum += static_cast<double>(out.numel());
            }
        };
        const bench::Sample looped = bench::measure(config, loop, [&] { return loop_checksum; });
        table.row(label, "matmul() loop", std::to_string(maximum_threads), looped, flops,
                  &sequential);
        table.blank();
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}
