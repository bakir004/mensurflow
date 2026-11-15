// conv2d forward and backward (data + weight + bias, what autograd runs): sequential versus
// parallel kernels on ResNet-18/50 layer shapes and a small layer, at batch 1 and 32.
//
//   conv2d_benchmark [--threads=1,6,12,24] [--batches=1,32] [--only=name] [--ref-threads=1,12]
//                    [--more] [--compare [--rounds=3]] [--gemm]
//
// --more adds the other ResNet-18/50 layer shapes and a VGG stem. --compare times only the
// strategies forced (forward: folded, whole images per thread, tiles; data gradient: folded, whole
// images), alternated over --rounds rounds at each thread count, against folded, and marks the one
// Auto picks with "<-".
//
// "fwd GEMM only" rows time only the forward pass's products on precomputed im2col columns, one
// call per image, at the --ref-threads counts: our parallel packed GEMM ("ours", with --gemm or
// whenever a reference is loaded) and, with MENSURFLOW_MKL_DLL and/or MENSURFLOW_AOCL_BLIS_DLL
// set, MKL's and AOCL-BLIS's sgemm.
//
// Rate is GFLOP/s, 2 flops per multiply-add (forward N * C_out * C_in * kH * kW * H_out * W_out
// multiply-adds; the backward pass does twice that, data and weight). Compare it with the FMA peak
// from fma_peak_benchmark. The PyTorch side is benchmarks/reference/conv_pool.py.
#include "bench.hpp"
#include "blas/reference_blas.hpp"

#include "backends/cpu/ops/conv/conv2d.hpp"
#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/threads.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cpu = mensurflow::cpu;

namespace {

struct Layer {
    const char* name;
    std::int64_t c_in, h, c_out, k, stride, pad; // square input and kernel
};

// Layers of torchvision's ResNet-18 and ResNet-50 at a 224 x 224 input, and one small layer.
const Layer layers[] = {
    {"r18.conv1 7x7/2", 3, 224, 64, 7, 2, 3},
    {"r18.layer1 3x3", 64, 56, 64, 3, 1, 1},
    {"r18.layer2.0 3x3/2", 64, 56, 128, 3, 2, 1},
    {"r18.layer3 3x3", 256, 14, 256, 3, 1, 1},
    {"r18.layer4 3x3", 512, 7, 512, 3, 1, 1},
    {"r50.layer1 1x1 256>64", 256, 56, 64, 1, 1, 0},
    {"r50.layer4 1x1 512>2048", 512, 7, 2048, 1, 1, 0},
    {"small 3x3 16ch 16px", 16, 16, 16, 3, 1, 1},
};

// With --more: the other distinct layer shapes of ResNet-18/50 (the 1x1 convolutions of every
// ResNet-50 stage, the first convolution of each ResNet-18 stage, the 3x3 at 28 px, and a VGG
// stem with a depth of only 27), used to
// check that the strategy rule holds beyond the layers it was set on.
const Layer more_layers[] = {
    {"r18.layer2 3x3", 128, 28, 128, 3, 1, 1},
    {"r18.layer3.0 3x3/2", 128, 28, 256, 3, 2, 1},
    {"r18.layer4.0 3x3/2", 256, 14, 512, 3, 2, 1},
    {"r50.layer1 1x1 64>64", 64, 56, 64, 1, 1, 0},
    {"r50.layer1 1x1 64>256", 64, 56, 256, 1, 1, 0},
    {"r50.layer2 1x1 512>128", 512, 28, 128, 1, 1, 0},
    {"r50.layer2 1x1 128>512", 128, 28, 512, 1, 1, 0},
    {"r50.layer3 1x1 1024>256", 1024, 14, 256, 1, 1, 0},
    {"r50.layer3 1x1 256>1024", 256, 14, 1024, 1, 1, 0},
    {"r50.layer4 1x1 2048>512", 2048, 7, 512, 1, 1, 0},
    {"vgg.conv1 3x3 3>64", 3, 224, 64, 3, 1, 1},
};

// OpenMP and pool workers keep spinning after a call; let them go idle before the next variant.
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }

std::vector<std::int64_t> parse_list(const std::string& text) {
    std::vector<std::int64_t> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        values.push_back(std::stoll(item));
    }
    return values;
}

// One sample from the timed runs of several: median, min and max over all of them.
bench::Sample merge(const std::vector<bench::Sample>& samples) {
    bench::Sample merged;
    for (const bench::Sample& s : samples) {
        merged.runs_s.insert(merged.runs_s.end(), s.runs_s.begin(), s.runs_s.end());
        merged.calls_per_sample = s.calls_per_sample;
    }
    std::vector<double> sorted = merged.runs_s;
    std::sort(sorted.begin(), sorted.end());
    merged.median_s = sorted[sorted.size() / 2];
    merged.min_s = sorted.front();
    merged.max_s = sorted.back();
    return merged;
}

cpu::Conv2dShape shape_of(const Layer& layer, std::int64_t batch) {
    cpu::Conv2dShape s;
    s.n = batch;
    s.c_in = layer.c_in;
    s.h = s.w = layer.h;
    s.c_out = layer.c_out;
    s.kh = s.kw = layer.k;
    s.stride_h = s.stride_w = layer.stride;
    s.pad_h = s.pad_w = layer.pad;
    s.h_out = s.w_out = (layer.h + 2 * layer.pad - layer.k) / layer.stride + 1;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    const int maximum_threads = cpu::max_threads();
    std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    std::vector<std::int64_t> batches = {1, 32};
    std::vector<int> reference_threads = {1, std::max(1, maximum_threads / 2)};
    std::string only;
    bool compare = false;
    bool gemm_rows = false;
    bool more = false;
    int rounds = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--threads=", 0) == 0) {
            thread_counts.clear();
            for (std::int64_t t : parse_list(arg.substr(10))) {
                thread_counts.push_back(static_cast<int>(t));
            }
        } else if (arg.rfind("--batches=", 0) == 0) {
            batches = parse_list(arg.substr(10));
        } else if (arg.rfind("--only=", 0) == 0) {
            only = arg.substr(7);
        } else if (arg == "--gemm") {
            gemm_rows = true;
        } else if (arg == "--more") {
            more = true;
        } else if (arg == "--compare") {
            compare = true;
        } else if (arg.rfind("--rounds=", 0) == 0) {
            rounds = std::max(1, std::stoi(arg.substr(9)));
        } else if (arg.rfind("--ref-threads=", 0) == 0) {
            reference_threads.clear();
            for (std::int64_t t : parse_list(arg.substr(14))) {
                reference_threads.push_back(static_cast<int>(t));
            }
        }
    }
    // MKL and AOCL-BLIS, loaded at run time when MENSURFLOW_MKL_DLL / MENSURFLOW_AOCL_BLIS_DLL
    // name their DLLs (see benchmarks/blas/reference_blas.hpp); skipped otherwise.
    std::vector<reference_blas::Library> references;
    for (auto load : {reference_blas::load_mkl, reference_blas::load_aocl_blis}) {
        reference_blas::Library library;
        if (load(library)) {
            references.push_back(std::move(library));
        }
    }
    gemm_rows = gemm_rows || !references.empty(); // our GEMM next to theirs

    const bench::Config config;
    bench::print_header(
        "conv2d forward and backward, float32 NCHW, im2col + packed GEMM",
        config,
        {"seq = conv2d_*_f32_sequential on one thread; par = conv2d_*_f32_parallel (Auto)",
         "fwd GEMM only = our GEMM or a reference sgemm per image on precomputed im2col columns",
         "par tiles = tiles forced, at every thread count",
         "par folded / par per-image = the strategy forced, at the largest thread count",
         "backward = data + weight + bias gradients, as autograd runs them",
         "rate = GFLOP/s at 2 flops per multiply-add; ceiling: fma_peak_benchmark"});
    for (const auto& library : references) {
        std::printf("reference %s: %s\n", library.name.c_str(), library.detail.c_str());
    }
    bench::Table table("GFLOP/s", 1e9, compare ? "vs folded" : "vs seq", "");

    std::vector<Layer> run_layers(std::begin(layers), std::end(layers));
    if (more) {
        run_layers.insert(run_layers.end(), std::begin(more_layers), std::end(more_layers));
    }
    for (const Layer& layer : run_layers) {
        if (!only.empty() && std::string(layer.name).find(only) == std::string::npos) {
            continue;
        }
        for (const std::int64_t batch : batches) {
            const cpu::Conv2dShape s = shape_of(layer, batch);
            const std::int64_t x_size = s.n * s.c_in * s.h * s.w;
            const std::int64_t w_size = s.c_out * s.c_in * s.kh * s.kw;
            const std::int64_t y_size = s.n * s.c_out * s.h_out * s.w_out;
            const double macs = static_cast<double>(s.n * s.c_out * s.c_in * s.kh * s.kw *
                                                    s.h_out * s.w_out);
            bench::Arrays arrays(static_cast<std::size_t>(std::max({x_size, w_size, y_size})), 9);
            float* x = arrays[0];
            float* w = arrays[1];
            float* b = arrays[2];
            float* y = arrays[3];
            float* gy = arrays[4];
            float* gx = arrays[5];
            float* gw = arrays[6];
            float* gb = arrays[7];
            float* expected = arrays[8];
            bench::fill_a(x, static_cast<std::size_t>(x_size));
            bench::fill_b(w, static_cast<std::size_t>(w_size));
            bench::fill_a(b, static_cast<std::size_t>(s.c_out));
            bench::fill_b(gy, static_cast<std::size_t>(y_size));
            for (std::int64_t i = 0; i < w_size; ++i) {
                w[i] *= 0.05f; // keep sums of many products moderate
            }

            const std::string label = std::string(layer.name) + " N=" + std::to_string(batch);
            table.section(label + "  (" + std::to_string(static_cast<long long>(macs / 1e6)) +
                          " M multiply-adds)");

            if (compare) {
                // The two strategies forced, forward and backward data, alternated over `rounds`
                // rounds at each thread count (one process, so a slow spell hits both alike).
                // "<-" marks the one Auto picks.
                struct Variant {
                    const char* pass;
                    cpu::Conv2dStrategy strategy;
                    const char* name;
                };
                const Variant variants[] = {
                    {"forward", cpu::Conv2dStrategy::Folded, "folded"},
                    {"forward", cpu::Conv2dStrategy::ImagePerThread, "per-image"},
                    {"forward", cpu::Conv2dStrategy::Tiles, "tiles"},
                    {"bwd data", cpu::Conv2dStrategy::Folded, "folded"},
                    {"bwd data", cpu::Conv2dStrategy::ImagePerThread, "per-image"},
                };
                for (int threads : thread_counts) {
                    cpu::set_max_threads(threads);
                    std::vector<std::vector<bench::Sample>> samples(std::size(variants));
                    for (int r = 0; r < rounds; ++r) {
                        for (std::size_t v = 0; v < std::size(variants); ++v) {
                            const Variant& variant = variants[v];
                            auto run = [&] {
                                if (variant.pass[0] == 'f') {
                                    cpu::conv2d_f32_parallel(x, w, b, y, s, variant.strategy);
                                } else {
                                    cpu::conv2d_backward_data_f32_parallel_strategy(
                                        gy, w, gx, s, variant.strategy);
                                }
                            };
                            settle();
                            samples[v].push_back(bench::measure(config, run, [&] {
                                return bench::checksum(y, static_cast<std::size_t>(y_size)) +
                                       bench::checksum(gx, static_cast<std::size_t>(x_size));
                            }));
                        }
                    }
                    for (std::size_t v = 0; v < std::size(variants); ++v) {
                        const bool forward = variants[v].pass[0] == 'f';
                        const cpu::Conv2dStrategy picked =
                            forward ? cpu::conv2d_forward_strategy(s)
                                    : cpu::conv2d_backward_data_strategy(s);
                        const bench::Sample sample = merge(samples[v]);
                        const bench::Sample folded = merge(samples[forward ? 0 : 3]);
                        table.row(variants[v].pass,
                                  std::string(variants[v].name) +
                                      (picked == variants[v].strategy ? " <-" : ""),
                                  std::to_string(threads), sample, 2 * macs,
                                  &folded);
                    }
                }
                continue;
            }

            // Forward.
            cpu::set_max_threads(1);
            auto forward_seq = [&] { cpu::conv2d_f32_sequential(x, w, b, y, s); };
            settle();
            forward_seq();
            std::copy(y, y + y_size, expected);
            const bench::Sample fseq = bench::measure(
                config, forward_seq, [&] { return bench::checksum(y, static_cast<std::size_t>(y_size)); });
            table.row("forward", "seq", "1", fseq, 2 * macs, &fseq);
            auto time_forward = [&](cpu::Conv2dStrategy strategy, const char* impl, int threads) {
                cpu::set_max_threads(threads);
                auto run = [&] { cpu::conv2d_f32_parallel(x, w, b, y, s, strategy); };
                settle();
                run();
                bench::check_close(expected, y, static_cast<std::size_t>(y_size), 1e-4,
                                   label + " forward " + impl);
                const bench::Sample sample = bench::measure(config, run, [&] {
                    return bench::checksum(y, static_cast<std::size_t>(y_size));
                });
                table.row("forward", impl, std::to_string(threads), sample, 2 * macs, &fseq);
            };
            for (int threads : thread_counts) {
                time_forward(cpu::Conv2dStrategy::Auto, "par", threads);
                time_forward(cpu::Conv2dStrategy::Tiles, "par tiles", threads);
            }
            const int top = thread_counts.back();
            time_forward(cpu::Conv2dStrategy::Folded, "par folded", top);
            time_forward(cpu::Conv2dStrategy::ImagePerThread, "par per-image", top);

            // Reference BLAS: only the products of the forward pass (weight times each image's
            // column matrix, prepared outside the timing), one sgemm per image. A lower bound for
            // an im2col + GEMM convolution built on that library; our rows above include im2col.
            // "ours" is our parallel packed GEMM on the same columns, the same way (one call per
            // image), so the products can be compared without im2col on either side.
            if (gemm_rows && s.c_in * s.kh * s.kw > 0) {
                const std::int64_t k = s.c_in * s.kh * s.kw;
                const std::int64_t spatial = s.h_out * s.w_out;
                std::vector<float> all_cols(static_cast<std::size_t>(s.n * k * spatial));
                for (std::int64_t img = 0; img < s.n; ++img) {
                    cpu::im2col_f32(x + img * s.c_in * s.h * s.w, all_cols.data() + img * k * spatial,
                                    s);
                }
                for (int threads : reference_threads) {
                    cpu::set_max_threads(threads);
                    auto run = [&] {
                        for (std::int64_t img = 0; img < s.n; ++img) {
                            cpu::matmul_f32_packed_parallel_strided(
                                w, k, 1, all_cols.data() + img * k * spatial, spatial, 1,
                                y + img * s.c_out * spatial, spatial, s.c_out, spatial, k);
                        }
                    };
                    settle();
                    const bench::Sample sample = bench::measure(config, run, [&] {
                        return bench::checksum(y, static_cast<std::size_t>(y_size));
                    });
                    table.row("fwd GEMM only", "ours", std::to_string(threads), sample, 2 * macs,
                              &fseq);
                }
                for (auto& library : references) {
                    for (int threads : reference_threads) {
                        library.set_threads(threads);
                        auto run = [&] {
                            for (std::int64_t img = 0; img < s.n; ++img) {
                                library.sgemm(false, false, static_cast<int>(s.c_out),
                                              static_cast<int>(spatial), static_cast<int>(k), w,
                                              static_cast<int>(k),
                                              all_cols.data() + img * k * spatial,
                                              static_cast<int>(spatial),
                                              y + img * s.c_out * spatial,
                                              static_cast<int>(spatial));
                            }
                        };
                        settle();
                        const bench::Sample sample = bench::measure(config, run, [&] {
                            return bench::checksum(y, static_cast<std::size_t>(y_size));
                        });
                        table.row("fwd GEMM only", library.name + " sgemm",
                                  std::to_string(threads), sample, 2 * macs, &fseq);
                    }
                }
            }

            // Backward: data, weight and bias.
            cpu::set_max_threads(1);
            auto backward_seq = [&] {
                cpu::conv2d_backward_data_f32_sequential(gy, w, gx, s);
                cpu::conv2d_backward_weight_f32_sequential(gy, x, gw, s);
                cpu::conv2d_backward_bias_f32_sequential(gy, gb, s);
            };
            settle();
            backward_seq();
            std::vector<float> want_gx(gx, gx + x_size), want_gw(gw, gw + w_size);
            const bench::Sample bseq = bench::measure(config, backward_seq, [&] {
                return bench::checksum(gx, static_cast<std::size_t>(x_size)) +
                       bench::checksum(gw, static_cast<std::size_t>(w_size));
            });
            table.row("backward", "seq", "1", bseq, 4 * macs, &bseq);
            for (int threads : thread_counts) {
                cpu::set_max_threads(threads);
                auto run = [&] {
                    cpu::conv2d_backward_data_f32_parallel(gy, w, gx, s);
                    cpu::conv2d_backward_weight_f32_parallel(gy, x, gw, s);
                    cpu::conv2d_backward_bias_f32_parallel(gy, gb, s);
                };
                settle();
                run();
                bench::check_close(want_gx.data(), gx, want_gx.size(), 1e-3,
                                   label + " backward data");
                bench::check_close(want_gw.data(), gw, want_gw.size(), 1e-3,
                                   label + " backward weight");
                const bench::Sample sample = bench::measure(config, run, [&] {
                    return bench::checksum(gx, static_cast<std::size_t>(x_size)) +
                           bench::checksum(gw, static_cast<std::size_t>(w_size));
                });
                table.row("backward", "par", std::to_string(threads), sample, 4 * macs, &bseq);
            }
        }
    }
    cpu::set_max_threads(maximum_threads);
}
