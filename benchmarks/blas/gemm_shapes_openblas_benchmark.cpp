// The matrix products a neural network makes, against OpenBLAS. Square matrices (see
// matmul_openblas_benchmark) are the easy case for a GEMM; a network multiplies mostly rectangular
// ones: a few rows (the batch) by a wide weight matrix, a short depth (the weight gradient, whose
// depth is the batch), or thousands of rows by a narrow output (a convolution done as im2col).
//
// C[m,n] = A[m,k] * B[k,n], float32. Each shape is timed with both libraries at the same thread
// counts, alternating, in several rounds; the table shows the median GFLOP/s (2*m*n*k per call) and
// ours / OpenBLAS (above 1.00 = we are faster). "best" compares each library at its own best thread
// count, which is what a user who tunes the thread count would get.
//
//   gemm_shapes_openblas_benchmark [key=v1,v2,...] ...
//     threads=   thread counts                (default 1, physical cores, all hardware threads)
//     rounds=    passes over both libraries   (default 3)
//     only=      1-based numbers of the shapes to run (default: all)
//     shape=     m,n,k: time this one product instead of the list (several: m,n,k,m,n,k,...)
//     pool=      PackedGemmBlocking::pool values to compare (default: the current default)
//     kc= mc= nc= tiles= parallel= pin= private= chunk= static= privatea=   one value each: overrides of the blocking settings
//
// Needs OpenBLAS at build time: see docs/BENCHMARKING.md ("Comparing against OpenBLAS").
#include "bench.hpp"

#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"

#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct Shape {
    const char* what;
    std::int64_t m, n, k;
    bool a_transposed; // A is stored as its transpose [k, m] (the weight-gradient product)
};

// Rows are batch (or batch * tokens, or output pixels); B is the weight matrix [inputs, outputs].
const Shape shapes[] = {
    {"mlp 784->512, batch 1", 1, 512, 784, false},
    {"mlp 784->512, batch 16", 16, 512, 784, false},
    {"mlp 784->512, batch 64", 64, 512, 784, false},
    {"mlp 784->512, batch 256", 256, 512, 784, false},
    {"mlp 512->10, batch 64", 64, 10, 512, false},
    {"mlp 4096->4096, batch 32", 32, 4096, 4096, false},
    {"mlp 4096->4096, batch 256", 256, 4096, 4096, false},
    {"transformer 768->768, 128 tok", 128, 768, 768, false},
    {"transformer 768->768, 2048 tok", 2048, 768, 768, false},
    {"transformer 768->3072, 512 tok", 512, 3072, 768, false},
    {"transformer 3072->768, 512 tok", 512, 768, 3072, false},
    {"transformer 768->3072, 8192 tok", 8192, 3072, 768, false},
    {"attention scores 128x128, d 64", 128, 128, 64, false},
    {"logits 768->32000, batch 64", 64, 32000, 768, false},
    {"conv im2col 3136 px, 576->64", 3136, 64, 576, false},
    {"conv im2col 784 px, 1152->128", 784, 128, 1152, false},
    {"conv im2col 196 px, 2304->256", 196, 256, 2304, false},
    {"conv im2col 12544 px, 147->64", 12544, 64, 147, false},
    {"weight grad 768x3072, batch 64", 768, 3072, 64, true},
    {"weight grad 4096x4096, batch 256", 4096, 4096, 256, true},
    {"weight grad 784x512, batch 16", 784, 512, 16, true},
};

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::atoll(item.c_str()));
        }
    }
    return values;
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

} // namespace

int main(int argc, char** argv) {
    const int max = cpu::max_threads();
    const cpu::PackedGemmBlocking original = cpu::packed_gemm_blocking();
    std::map<std::string, std::vector<std::int64_t>> options = {
        {"threads", {1, std::max(1, max / 2), max}},
        {"rounds", {3}},
        {"only", {}},
        {"shape", {}},
        {"pool", {original.pool}},
        {"kc", {original.kc}},
        {"mc", {original.mc}},
        {"nc", {original.nc}},
        {"tiles", {original.tiles_per_thread}},
        {"parallel", {original.parallel}},
        {"pin", {original.pin}},
        {"private", {original.private_b}},
        {"chunk", {original.b_chunk}},
        {"static", {original.static_items}},
        {"privatea", {original.private_a}},
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq == std::string::npos || options.find(arg.substr(0, eq)) == options.end()) {
            std::fprintf(stderr, "unknown argument '%s' (see the comment at the top of the file)\n",
                         argv[i]);
            return 2;
        }
        options[arg.substr(0, eq)] = parse_list(arg.substr(eq + 1));
    }
    std::vector<std::int64_t> threads = options["threads"];
    threads.erase(std::remove_if(threads.begin(), threads.end(),
                                 [&](std::int64_t t) { return t < 1 || t > max; }),
                  threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());
    const std::int64_t rounds = std::max<std::int64_t>(1, options["rounds"][0]);
    const std::vector<std::int64_t> pools = options["pool"];

    bench::Config config;
    config.repeats = 7;
    bench::print_header(
        "Neural-network shaped products C[m,n] = A[m,k] * B[k,n] vs OpenBLAS, float32", config,
        {std::string("OpenBLAS: ") + openblas_get_config() + ", core " + openblas_get_corename(),
         "GFLOP/s are medians over " + std::to_string(rounds) +
             " rounds; ratio = ours / OpenBLAS (above 1.00: we are faster)"});

    std::printf("\n%-34s %6s %6s %6s %4s |", "shape", "m", "n", "k", "pool");
    for (std::int64_t t : threads) {
        std::printf(" %5s %5s %5s |", ("our" + std::to_string(t)).c_str(),
                    ("obl" + std::to_string(t)).c_str(), "ratio");
    }
    std::printf(" %5s %5s %5s\n", "our*", "obl*", "best");

    std::vector<double> best_ratios;
    int number = 0;
    std::vector<Shape> list(std::begin(shapes), std::end(shapes));
    if (options["shape"].size() >= 3) {
        list.clear();
        for (std::size_t i = 0; i + 2 < options["shape"].size(); i += 3) {
            list.push_back({"custom", options["shape"][i], options["shape"][i + 1],
                            options["shape"][i + 2], false});
        }
    }
    for (const Shape& shape : list) {
        ++number;
        const auto& only = options["only"];
        if (!only.empty() && std::find(only.begin(), only.end(), number) == only.end()) {
            continue;
        }
        const std::int64_t m = shape.m;
        const std::int64_t n = shape.n;
        const std::int64_t k = shape.k;
        const double flops =
            2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
        const auto c_len = static_cast<std::size_t>(m * n);
        bench::Arrays a_arrays(static_cast<std::size_t>(m * k), 1);
        bench::Arrays b_arrays(static_cast<std::size_t>(k * n), 1);
        bench::Arrays c_arrays(c_len, 2);
        float* a = a_arrays[0];
        float* b = b_arrays[0];
        float* c = c_arrays[0];
        float* reference = c_arrays[1];
        bench::fill_a(a, static_cast<std::size_t>(m * k));
        bench::fill_b(b, static_cast<std::size_t>(k * n));
        const int mi = static_cast<int>(m);
        const int ni = static_cast<int>(n);
        const int ki = static_cast<int>(k);

        const auto run_openblas_into = [&](float* out) {
            if (shape.a_transposed) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, mi, ni, ki, 1.0f, a, mi, b, ni,
                            0.0f, out, ni);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mi, ni, ki, 1.0f, a, ki, b,
                            ni, 0.0f, out, ni);
            }
        };
        const auto run_ours = [&] {
            if (shape.a_transposed) {
                cpu::matmul_f32_packed_parallel_strided(a, 1, m, b, n, 1, c, n, m, n, k);
            } else {
                cpu::matmul_f32_packed_parallel(a, b, c, m, n, k);
            }
        };
        const auto run_openblas = [&] { run_openblas_into(c); };
        const auto observe = [&] { return bench::checksum(c, c_len); };

        openblas_set_num_threads(max);
        run_openblas_into(reference);

        for (std::int64_t pool : pools) {
            cpu::PackedGemmBlocking blocking = original;
            blocking.pool = pool;
            blocking.kc = options["kc"][0];
            blocking.mc = options["mc"][0];
            blocking.nc = options["nc"][0];
            blocking.tiles_per_thread = options["tiles"][0];
            blocking.parallel = options["parallel"][0];
            blocking.pin = options["pin"][0];
            blocking.private_b = options["private"][0];
            blocking.b_chunk = options["chunk"][0];
            blocking.static_items = options["static"][0];
            blocking.private_a = options["privatea"][0];
            cpu::set_packed_gemm_blocking(blocking);

            std::printf("%-34s %6lld %6lld %6lld %4lld |", shape.what, static_cast<long long>(m),
                        static_cast<long long>(n), static_cast<long long>(k),
                        static_cast<long long>(pool));
            double best_ours = 0.0;
            double best_openblas = 0.0;
            for (std::int64_t t : threads) {
                cpu::set_max_threads(static_cast<int>(t));
                openblas_set_num_threads(static_cast<int>(t));
                run_ours();
                bench::check_close(reference, c, c_len, 1e-3,
                                   std::string(shape.what) + " threads=" + std::to_string(t));
                std::vector<double> ours;
                std::vector<double> theirs;
                for (std::int64_t round = 0; round < rounds; ++round) {
                    settle();
                    run_openblas();
                    theirs.push_back(flops / bench::measure(config, run_openblas, observe).median_s /
                                     1e9);
                    settle();
                    run_ours();
                    ours.push_back(flops / bench::measure(config, run_ours, observe).median_s / 1e9);
                }
                const double o = median_of(ours);
                const double x = median_of(theirs);
                best_ours = std::max(best_ours, o);
                best_openblas = std::max(best_openblas, x);
                std::printf(" %5.0f %5.0f %5.2f |", o, x, o / x);
                std::fflush(stdout);
            }
            std::printf(" %5.0f %5.0f %5.2f\n", best_ours, best_openblas, best_ours / best_openblas);
            std::fflush(stdout);
            if (pool == pools.back()) {
                best_ratios.push_back(best_ours / best_openblas);
            }
        }
    }
    if (!best_ratios.empty()) {
        double product = 1.0;
        int wins = 0;
        for (double r : best_ratios) {
            product *= r;
            wins += r >= 1.0 ? 1 : 0;
        }
        double geomean = 1.0;
        // n-th root by repeated square roots is overkill: use exp/log through pow.
        geomean = std::exp(std::log(product) / static_cast<double>(best_ratios.size()));
        std::printf("\nbest-vs-best: geometric mean ratio %.3f, faster than OpenBLAS on %d of %d "
                    "shapes\n",
                    geomean, wins, static_cast<int>(best_ratios.size()));
    }
    cpu::set_packed_gemm_blocking(original);
    cpu::set_max_threads(max);
    return 0;
}
