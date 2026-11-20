// Profiling driver: runs ONE production kernel (the parallel version the library calls) in a loop
// for a fixed time, so a profiler sees nothing else: no sequential baselines, no checksums, no other
// shapes. Prints calls, ms per call and GFLOP/s or GB/s (the minimum bytes moved) at the end.
//
//   profile_driver <op> [seconds = 8] [threads = 0, the library default]
//   profile_driver            lists the ops
//
// Shapes are the ones of conv2d_benchmark, pooling_benchmark and argmin_var_std_benchmark (batch 32
// for conv and pooling). With AMD uProf (https://www.amd.com/en/developer/uprof.html), build with
// debug information so it finds the function names, e.g. a separate folder
//   cmake -S . -B build-prof -G "Visual Studio 18 2026" -A x64 -DMENSURFLOW_CUDA=OFF
//         "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /Ob2 /DNDEBUG /Zi"
//         "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF"
//   cmake --build build-prof --config Release --target profile_driver
// then
//   AMDuProfCLI collect --config assess_ext -o build-prof\uprof\maxpool_fwd
//               build-prof\Release\profile_driver.exe maxpool_fwd 8
//   AMDuProfCLI report -i build-prof\uprof\maxpool_fwd\<session folder>
// report.csv lists the 10 hottest functions with CPI, cache refills, branch mispredictions and
// store-to-load conflicts. `--config tbp` is time sampling only; `AMDuProfCLI info --list
// collect-configs` shows the rest. memset/memcpy show up as vcruntime140.dll addresses (no symbols).
#include "backends/cpu/ops/conv/conv2d.hpp"
#include "backends/cpu/ops/conv/pooling.hpp"
#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/ops/reduction/reduction.hpp"
#include "backends/cpu/threads.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;
using i64 = std::int64_t;

static std::vector<float> rnd(i64 n) {
    std::vector<float> v(static_cast<std::size_t>(n));
    std::mt19937 g(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& x : v) x = d(g);
    return v;
}

static cpu::Conv2dShape conv(i64 n, i64 cin, i64 hw, i64 cout, i64 k, i64 s, i64 p) {
    cpu::Conv2dShape c;
    c.n = n; c.c_in = cin; c.h = c.w = hw; c.c_out = cout; c.kh = c.kw = k;
    c.stride_h = c.stride_w = s; c.pad_h = c.pad_w = p;
    c.h_out = c.w_out = (hw + 2 * p - k) / s + 1;
    return c;
}

static cpu::Pool2dShape pool(i64 n, i64 ch, i64 hw, i64 k, i64 s, i64 p) {
    cpu::Pool2dShape c;
    c.n = n; c.c = ch; c.h = c.w = hw; c.kh = c.kw = k;
    c.stride_h = c.stride_w = s; c.pad_h = c.pad_w = p;
    c.h_out = c.w_out = (hw + 2 * p - k) / s + 1;
    return c;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("ops: matmul2048 matmul512 conv_fwd conv1_fwd conv_bwd_data conv_bwd_weight "
                    "maxpool_fwd maxpool_bwd avgpool_fwd avgpool_bwd argmin_classes argmin_kmeans "
                    "var_ln var_bn std_ln var_bwd_ln\n");
        return 1;
    }
    const std::string op = argv[1];
    const double seconds = argc > 2 ? std::atof(argv[2]) : 8.0;
    if (argc > 3 && std::atoi(argv[3]) > 0) cpu::set_max_threads(std::atoi(argv[3]));

    std::function<void()> run;
    std::vector<float> a, b, c, d;
    std::vector<i64> idx;
    cpu::Conv2dShape cs;
    cpu::Pool2dShape ps;
    std::vector<i64> shape;
    double bytes_or_flops = 0; // per call, for a sanity rate
    const char* unit = "GFLOP/s";

    auto mm = [&](i64 n) {
        a = rnd(n * n); b = rnd(n * n); c.assign(n * n, 0.f);
        bytes_or_flops = 2.0 * n * n * n;
        run = [&, n] { cpu::matmul_f32_packed_parallel(a.data(), b.data(), c.data(), n, n, n); };
    };
    auto conv_setup = [&](cpu::Conv2dShape s) {
        cs = s;
        a = rnd(s.n * s.c_in * s.h * s.w);                     // x / grad_x
        b = rnd(s.c_out * s.c_in * s.kh * s.kw);               // weight
        c = rnd(s.n * s.c_out * s.h_out * s.w_out);            // y / grad_y
        d = rnd(s.c_out);                                      // bias
        bytes_or_flops = 2.0 * s.n * s.c_out * s.h_out * s.w_out * s.c_in * s.kh * s.kw;
    };
    auto pool_setup = [&](cpu::Pool2dShape s) {
        ps = s;
        a = rnd(s.n * s.c * s.h * s.w);
        c = rnd(s.n * s.c * s.h_out * s.w_out);
        idx.assign(c.size(), 0);
        cpu::max_pool2d_f32_parallel(a.data(), c.data(), idx.data(), ps); // valid indices
        unit = "GB/s";
    };
    auto moments = [&](std::vector<i64> sh, i64 axis, int which) {
        shape = sh;
        i64 n = 1, out = 1;
        for (std::size_t i = 0; i < sh.size(); ++i) { n *= sh[i]; if ((i64)i != axis) out *= sh[i]; }
        a = rnd(n); c.assign(out, 0.f); b = rnd(out); d.assign(n, 0.f);
        unit = "GB/s";
        const i64 rank = (i64)sh.size();
        if (which == 0) { bytes_or_flops = 4.0 * n;
            run = [&, rank, axis] { cpu::var_axis_f32_parallel(a.data(), c.data(), shape.data(), rank, axis, 1); }; }
        if (which == 1) { bytes_or_flops = 4.0 * n;
            run = [&, rank, axis] { cpu::std_axis_f32_parallel(a.data(), c.data(), shape.data(), rank, axis, 1); }; }
        if (which == 2) { bytes_or_flops = 8.0 * n;
            run = [&, rank, axis] { cpu::var_backward_f32_parallel(a.data(), b.data(), d.data(), shape.data(), rank, axis, 1); }; }
    };

    if (op == "matmul2048") mm(2048);
    else if (op == "matmul512") mm(512);
    else if (op == "conv_fwd" || op == "conv_bwd_data" || op == "conv_bwd_weight" || op == "conv1_fwd") {
        conv_setup(op == "conv1_fwd" ? conv(32, 3, 224, 64, 7, 2, 3)     // r18.conv1 7x7/2
                                     : conv(32, 64, 56, 64, 3, 1, 1));   // r18.layer1 3x3
        if (op == "conv_fwd" || op == "conv1_fwd")
            run = [&] { cpu::conv2d_f32_parallel(a.data(), b.data(), d.data(), c.data(), cs); };
        else if (op == "conv_bwd_data")
            run = [&] { cpu::conv2d_backward_data_f32_parallel(c.data(), b.data(), a.data(), cs); };
        else
            run = [&] { cpu::conv2d_backward_weight_f32_parallel(c.data(), a.data(), b.data(), cs); };
    } else if (op == "maxpool_fwd" || op == "maxpool_bwd") {
        pool_setup(pool(32, 64, 112, 3, 2, 1)); // r18.maxpool 3x3/2
        if (op == "maxpool_fwd") {
            bytes_or_flops = 4.0 * a.size() + 12.0 * c.size();
            run = [&] { cpu::max_pool2d_f32_parallel(a.data(), c.data(), idx.data(), ps); };
        } else {
            bytes_or_flops = 4.0 * a.size() + 12.0 * c.size();
            run = [&] { cpu::max_pool2d_backward_f32_parallel(c.data(), idx.data(), a.data(), ps); };
        }
    } else if (op == "avgpool_fwd" || op == "avgpool_bwd") {
        pool_setup(pool(32, 128, 56, 2, 2, 0)); // 2x2/2 128ch 56px
        bytes_or_flops = 4.0 * a.size() + 4.0 * c.size();
        if (op == "avgpool_fwd") run = [&] { cpu::avg_pool2d_f32_parallel(a.data(), c.data(), ps); };
        else run = [&] { cpu::avg_pool2d_backward_f32_parallel(c.data(), a.data(), ps); };
    } else if (op == "argmin_classes" || op == "argmin_kmeans") {
        const i64 rows = op == "argmin_classes" ? 1024 : 65536, width = op == "argmin_classes" ? 1000 : 64;
        a = rnd(rows * width); idx.assign(rows, 0);
        bytes_or_flops = 4.0 * rows * width; unit = "GB/s";
        run = [&, rows, width] { cpu::argmin_f32_parallel(a.data(), idx.data(), rows, width); };
    } else if (op == "var_ln") moments({6272, 768}, 1, 0);
    else if (op == "var_bn") moments({100352, 64}, 0, 0);
    else if (op == "std_ln") moments({6272, 768}, 1, 1);
    else if (op == "var_bwd_ln") moments({6272, 768}, 1, 2);
    else { std::printf("unknown op %s\n", op.c_str()); return 1; }

    for (int i = 0; i < 3; ++i) run(); // warm up: pool threads, pages, packed buffers
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    i64 calls = 0;
    while (std::chrono::duration<double>(clk::now() - t0).count() < seconds) { run(); ++calls; }
    const double el = std::chrono::duration<double>(clk::now() - t0).count();
    std::printf("%s: %lld calls in %.2f s, %.3f ms/call, %.1f %s, threads=%d\n", op.c_str(),
                (long long)calls, el, 1e3 * el / calls, bytes_or_flops * calls / el / 1e9, unit,
                cpu::max_threads());
    return 0;
}
