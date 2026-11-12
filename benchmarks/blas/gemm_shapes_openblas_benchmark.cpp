// The matrix products a neural network makes, against OpenBLAS, MKL and AOCL-BLIS. Square matrices (see
// matmul_openblas_benchmark) are the easy case for a GEMM; a network multiplies mostly rectangular
// ones: a few rows (the batch) by a wide weight matrix, a short depth (the weight gradient, whose
// depth is the batch), or thousands of rows by a narrow output (a convolution done as im2col).
//
// C[m,n] = A[m,k] * B[k,n], float32. Each shape is timed with every library at the same thread
// counts, alternating, in several rounds; the table shows the median GFLOP/s (2*m*n*k per call).
// The columns marked * are each library at its own best thread count, which is what a user who
// tunes the thread count would get, and the last columns are ours* / theirs* (above 1.00 = we are
// faster).
//
// OpenBLAS is linked. MKL and AOCL-BLIS are loaded at run time when they are installed (see
// reference_blas.hpp for where they are looked for); without them the table has OpenBLAS only.
//
//   gemm_shapes_openblas_benchmark [key=v1,v2,...] ...
//     threads=   thread counts                (default 1, physical cores, all hardware threads)
//     rounds=    passes over all libraries    (default 3)
//     mkl= aocl= 0: leave that library out    (default 1: used when found)
//     settle=    pause in ms before each library is timed (default 300: the OpenMP workers of MKL
//                and AOCL-BLIS spin for 200 ms after a call and would take cores from the next)
//     only=      1-based numbers of the shapes to run (default: all)
//     shape=     m,n,k: time this one product instead of the list (several: m,n,k,m,n,k,...)
//     prepack=   1: our side multiplies by a B packed once with pack_b_f32 (not timed), as a layer
//                in inference would; the other libraries have no such call and are timed as usual
//     pool=      PackedGemmBlocking::pool values to compare (default: the current default)
//     kc= mc= nc= tiles= parallel= pin= private= chunk= static= privatea= finerows= splitrows= indep= narrow= tilefit=   one value each: overrides of the blocking settings
//
// Needs OpenBLAS at build time: see docs/BENCHMARKING.md ("Comparing against OpenBLAS").
#include "bench.hpp"
#include "reference_blas.hpp"

#include "backends/cpu/affinity.hpp"
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
    int layout; // 0 plain; 1 A stored as its transpose [k, m] (the weight gradient); 2 B stored as
                // its transpose [n, k] (the gradient of the input: delta times the weights transposed)
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
    {"input grad 784<-512, batch 64", 64, 784, 512, 2},
    {"input grad 4096<-4096, batch 256", 256, 4096, 4096, 2},
    {"input grad 768<-3072, 512 tok", 512, 768, 3072, 2},
    {"input grad 3072<-768, 512 tok", 512, 3072, 768, 2},
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

void settle(std::int64_t ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

double geomean_of(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += std::log(v);
    }
    return std::exp(sum / static_cast<double>(values.size()));
}

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
        {"finerows", {original.fine_rows}},
        {"splitrows", {original.split_rows}},
        {"indep", {original.independent_rows}},
        {"narrow", {original.narrow}},
        {"tilefit", {original.tile_fit}},
        {"prepack", {0}},
        {"mkl", {1}},
        {"aocl", {1}},
        {"settle", {300}},
        {"unpin", {1}},
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
    const std::int64_t settle_ms = std::max<std::int64_t>(0, options["settle"][0]);

    // The libraries we are compared with: OpenBLAS (linked), then the ones found at run time.
    std::vector<reference_blas::Library> libraries;
    {
        reference_blas::Library openblas;
        openblas.name = "obl";
        openblas.detail = std::string("OpenBLAS: ") + openblas_get_config() + ", core " +
                          openblas_get_corename();
        openblas.set_threads = [](int t) { openblas_set_num_threads(t); };
        openblas.sgemm = [](bool ta, bool tb, int m, int n, int k, const float* a, int lda,
                            const float* b, int ldb, float* c, int ldc) {
            cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                        tb ? CblasTrans : CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
        };
        libraries.push_back(openblas);
        reference_blas::Library found;
        if (options["mkl"][0] != 0 && reference_blas::load_mkl(found)) {
            libraries.push_back(found);
        }
        found = {};
        if (options["aocl"][0] != 0 && reference_blas::load_aocl_blis(found)) {
            libraries.push_back(found);
        }
    }
    const std::size_t library_count = libraries.size();

    bench::Config config;
    config.repeats = 7;
    std::vector<std::string> notes;
    for (const auto& library : libraries) {
        notes.push_back(library.detail);
    }
    notes.push_back("GFLOP/s are medians over " + std::to_string(rounds) +
                    " rounds; * = best thread count of that library; /name = ours* / theirs* "
                    "(above 1.00: we are faster)");
    bench::print_header(
        "Neural-network shaped products C[m,n] = A[m,k] * B[k,n] vs reference BLAS, float32",
        config, notes);

    std::printf("\n%-34s %6s %6s %6s %4s |", "shape", "m", "n", "k", "pool");
    for (std::int64_t t : threads) {
        std::printf(" %6s", ("our" + std::to_string(t)).c_str());
        for (const auto& library : libraries) {
            std::printf(" %6s", (library.name + std::to_string(t)).c_str());
        }
        std::printf(" |");
    }
    std::printf(" %6s", "our*");
    for (const auto& library : libraries) {
        std::printf(" %6s", (library.name + "*").c_str());
    }
    std::printf(" |");
    for (const auto& library : libraries) {
        std::printf(" %6s", ("/" + library.name).c_str());
    }
    std::printf("\n");

    // best_ratios[i]: ours* / library i*, one entry per shape; the last one is against the fastest
    // of the libraries on that shape.
    std::vector<std::vector<double>> best_ratios(library_count + 1);
    int number = 0;
    std::vector<Shape> list(std::begin(shapes), std::end(shapes));
    if (options["shape"].size() >= 3) {
        list.clear();
        for (std::size_t i = 0; i + 2 < options["shape"].size(); i += 3) {
            list.push_back({"custom", options["shape"][i], options["shape"][i + 1],
                            options["shape"][i + 2], 0});
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

        const auto run_library_into = [&](const reference_blas::Library& library, float* out) {
            if (shape.layout == 1) {
                library.sgemm(true, false, mi, ni, ki, a, mi, b, ni, out, ni);
            } else if (shape.layout == 2) {
                library.sgemm(false, true, mi, ni, ki, a, ki, b, ki, out, ni);
            } else {
                library.sgemm(false, false, mi, ni, ki, a, ki, b, ni, out, ni);
            }
        };
        // prepack=1: B is packed once, outside the timing (the weights of a layer in inference).
        const bool prepack = options["prepack"][0] != 0;
        cpu::PackedB packed;
        if (prepack) {
            packed = shape.layout == 2 ? cpu::pack_b_f32(b, 1, k, k, n) : cpu::pack_b_f32(b, n, 1, k, n);
        }
        const auto run_ours = [&] {
            if (prepack) {
                if (shape.layout == 1) {
                    cpu::matmul_f32_prepacked(a, 1, m, packed, c, n, m);
                } else {
                    cpu::matmul_f32_prepacked(a, k, 1, packed, c, n, m);
                }
            } else if (shape.layout == 1) {
                cpu::matmul_f32_packed_parallel_strided(a, 1, m, b, n, 1, c, n, m, n, k);
            } else if (shape.layout == 2) {
                cpu::matmul_f32_packed_parallel_strided(a, k, 1, b, 1, k, c, n, m, n, k);
            } else {
                cpu::matmul_f32_packed_parallel(a, b, c, m, n, k);
            }
        };
        const auto observe = [&] { return bench::checksum(c, c_len); };

        // OpenBLAS gives the reference result; the other libraries are checked against it too.
        for (std::size_t i = 0; i < library_count; ++i) {
            libraries[i].set_threads(max);
            run_library_into(libraries[i], i == 0 ? reference : c);
            if (i > 0) {
                bench::check_close(reference, c, c_len, 1e-3,
                                   std::string(shape.what) + " " + libraries[i].name);
            }
        }

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
            blocking.fine_rows = options["finerows"][0];
            blocking.split_rows = options["splitrows"][0];
            blocking.independent_rows = options["indep"][0];
            blocking.narrow = options["narrow"][0];
            blocking.tile_fit = options["tilefit"][0];
            cpu::set_packed_gemm_blocking(blocking);

            std::printf("%-34s %6lld %6lld %6lld %4lld |", shape.what, static_cast<long long>(m),
                        static_cast<long long>(n), static_cast<long long>(k),
                        static_cast<long long>(pool));
            double best_ours = 0.0;
            std::vector<double> best_theirs(library_count, 0.0);
            for (std::int64_t t : threads) {
                cpu::set_max_threads(static_cast<int>(t));
                for (const auto& library : libraries) {
                    library.set_threads(static_cast<int>(t));
                }
                run_ours();
                bench::check_close(reference, c, c_len, 1e-3,
                                   std::string(shape.what) + " threads=" + std::to_string(t));
                std::vector<double> ours;
                std::vector<std::vector<double>> theirs(library_count);
                for (std::int64_t round = 0; round < rounds; ++round) {
                    for (std::size_t i = 0; i < library_count; ++i) {
                        const auto run_library = [&] { run_library_into(libraries[i], c); };
                        // Our matmul may leave the calling thread pinned to a core; the other
                        // libraries get the thread back as the program started it.
                        if (options["unpin"][0] != 0) {
                            cpu::set_thread_pin(0, 0);
                        }
                        settle(settle_ms);
                        run_library();
                        theirs[i].push_back(
                            flops / bench::measure(config, run_library, observe).median_s / 1e9);
                    }
                    settle(settle_ms);
                    run_ours();
                    ours.push_back(flops / bench::measure(config, run_ours, observe).median_s / 1e9);
                }
                const double o = median_of(ours);
                best_ours = std::max(best_ours, o);
                std::printf(" %6.0f", o);
                for (std::size_t i = 0; i < library_count; ++i) {
                    const double x = median_of(theirs[i]);
                    best_theirs[i] = std::max(best_theirs[i], x);
                    std::printf(" %6.0f", x);
                }
                std::printf(" |");
                std::fflush(stdout);
            }
            std::printf(" %6.0f", best_ours);
            for (double x : best_theirs) {
                std::printf(" %6.0f", x);
            }
            std::printf(" |");
            for (double x : best_theirs) {
                std::printf(" %6.2f", best_ours / x);
            }
            std::printf("\n");
            std::fflush(stdout);
            if (pool == pools.back()) {
                for (std::size_t i = 0; i < library_count; ++i) {
                    best_ratios[i].push_back(best_ours / best_theirs[i]);
                }
                best_ratios[library_count].push_back(
                    best_ours / *std::max_element(best_theirs.begin(), best_theirs.end()));
            }
        }
    }
    if (!best_ratios[0].empty()) {
        std::printf("\nbest-vs-best (ours* / theirs*):\n");
        for (std::size_t i = 0; i <= library_count; ++i) {
            if (i == library_count && library_count == 1) {
                break;
            }
            const std::vector<double>& ratios = best_ratios[i];
            const auto wins = std::count_if(ratios.begin(), ratios.end(),
                                            [](double r) { return r >= 1.0; });
            std::printf("  %-22s geometric mean %.3f, lowest %.2f, we are faster on %d of %d shapes\n",
                        i < library_count ? libraries[i].name.c_str() : "fastest of the above",
                        geomean_of(ratios), *std::min_element(ratios.begin(), ratios.end()),
                        static_cast<int>(wins), static_cast<int>(ratios.size()));
        }
    }
    cpu::set_packed_gemm_blocking(original);
    cpu::set_max_threads(max);
    return 0;
}
