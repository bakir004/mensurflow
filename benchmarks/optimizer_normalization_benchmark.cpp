// sgd, adam, adamw, layer_norm, rms_norm and batch_norm, forward and backward.
//   sequential  the kernel on one thread (the baseline).
//   parallel    the same kernel with N OpenMP threads. Its result is checked bit for bit against the
//               sequential one first: every thread owns whole lines/planes/elements and the sums
//               across them are added in blocks fixed by the shape, so the bits never depend on N.
// The first call of each case is also checked against a double-precision scalar reference
// (tests/reference/optimizer_normalization.hpp) on a few lines, before anything is timed.
//
// The normalization sections call the CPU kernels on preallocated buffers, the way PyTorch's
// `out=` variants are timed in benchmarks/reference/optimizer_normalization.py; the last section
// goes through the public API (which allocates its results) for the same shapes. Optimizers are
// in-place and always go through the public API.
//
// Rate is GB/s of bytes the op must move: every input read once and every output written once
// (weights and biases are negligible). "vs seq" divides the sequential median time by this row's.
// Optimizer state keeps evolving across timed calls (Adam is stepped with a fixed step number), which
// does not change the work per element.
//
//   optimizer_normalization_benchmark [optimizers|line_norm|batch_norm|api]
#include "bench.hpp"

#include "backends/cpu/ops/normalization.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"
#include "reference/optimizer_normalization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace mf = mensurflow;
namespace ref = reference;

namespace {

std::string shape_name(std::initializer_list<std::int64_t> dims) {
    std::string out;
    for (const std::int64_t d : dims) {
        out += (out.empty() ? "" : "x") + std::to_string(d);
    }
    return out;
}

// Logits-like values in [-4, 4).
void fill_values(float* data, std::size_t n, int seed) {
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(
                      static_cast<int>((i * 37 + static_cast<std::size_t>(seed) * 101) % 97) - 48) /
                  12.0f;
    }
}

void fill_positive(float* data, std::size_t n, int seed) {
    fill_values(data, n, seed);
    for (std::size_t i = 0; i < n; ++i) {
        data[i] = 0.5f + std::abs(data[i]) / 4.0f;
    }
}

// One row per thread count for `run`, which writes `outputs` (pointers and element counts): the
// sequential baseline, then each thread count checked against it bit for bit. `check` runs once on
// the sequential result, before the timing.
template <class Run, class Check>
void sweep(bench::Table& table, const bench::Config& config, const std::string& label,
           const std::vector<int>& thread_counts, double work,
           const std::vector<std::pair<const float*, std::size_t>>& outputs, Run&& run,
           Check&& check) {
    cpu::set_max_threads(1);
    run();
    check();
    std::vector<std::vector<float>> expected;
    for (const auto& o : outputs) {
        expected.emplace_back(o.first, o.first + o.second);
    }
    auto observe = [&] {
        double sum = 0.0;
        for (const auto& o : outputs) {
            sum += bench::checksum(o.first, std::min<std::size_t>(o.second, 4096));
        }
        return sum;
    };
    const bench::Sample sequential = bench::measure(config, run, observe);
    table.row(label, "sequential", "1", sequential, work, &sequential);
    for (const int threads : thread_counts) {
        if (threads == 1) {
            continue;
        }
        cpu::set_max_threads(threads);
        run();
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            bench::check_bits(expected[i].data(), outputs[i].first, outputs[i].second,
                              label + " threads=" + std::to_string(threads));
        }
        const bench::Sample parallel = bench::measure(config, run, observe);
        table.row(label, "parallel", std::to_string(threads), parallel, work, &sequential);
    }
    cpu::set_max_threads(1);
    table.blank();
}

// Same, for a call that updates its arguments in place: the state is not bitwise comparable after
// an unknown number of timed calls, so the check is one reset-run-compare per thread count.
template <class Reset, class Run, class Snapshot>
void sweep_in_place(bench::Table& table, const bench::Config& config, const std::string& label,
                    const std::vector<int>& thread_counts, double work, Reset&& reset, Run&& run,
                    Snapshot&& snapshot, const std::vector<double>& expected_head) {
    auto observe = [&] { return snapshot()[0]; };
    cpu::set_max_threads(1);
    reset();
    run();
    const std::vector<float> after_one = snapshot();
    // The first elements against the double-precision reference, one step from the same start.
    for (std::size_t i = 0; i < expected_head.size(); ++i) {
        bench::check_value(expected_head[i], after_one[i], 1e-4, label + " vs reference", 1.0);
    }
    const bench::Sample sequential = bench::measure(config, run, observe);
    table.row(label, "sequential", "1", sequential, work, &sequential);
    for (const int threads : thread_counts) {
        if (threads == 1) {
            continue;
        }
        cpu::set_max_threads(threads);
        reset();
        run();
        bench::check_bits(after_one.data(), snapshot().data(), after_one.size(),
                          label + " threads=" + std::to_string(threads));
        const bench::Sample parallel = bench::measure(config, run, observe);
        table.row(label, "parallel", std::to_string(threads), parallel, work, &sequential);
    }
    cpu::set_max_threads(1);
    table.blank();
}

// ---- optimizers ------------------------------------------------------------------------------

void optimizer_section(bench::Table& table, const bench::Config& config,
                       const std::vector<int>& thread_counts) {
    // 1k (call overhead), 64k (around the cutoff), 1M (4 MB per array), 16M (64 MB: beyond L3).
    const std::int64_t sizes[] = {1000, 1 << 16, 1 << 20, 1 << 24};
    for (const char* name : {"sgd_momentum", "adam", "adamw"}) {
        table.section(std::string(name) + " [n elements], in place");
        for (const std::int64_t n : sizes) {
            const auto count = static_cast<std::size_t>(n);
            std::vector<float> p0(count), g0(count);
            fill_values(p0.data(), count, 1);
            fill_values(g0.data(), count, 2);
            for (float& v : g0) {
                v /= 8.0f;
            }
            mf::Tensor p = mf::Tensor::from_vector(p0, {n});
            const mf::Tensor g = mf::Tensor::from_vector(g0, {n});
            mf::Tensor m = mf::Tensor::zeros({n}), v = mf::Tensor::zeros({n});
            auto reset = [&] {
                p.copy_from(mf::Tensor::from_vector(p0, {n}));
                m.copy_from(mf::Tensor::zeros({n}));
                v.copy_from(mf::Tensor::zeros({n}));
            };
            const std::string kind = name;
            auto run = [&] {
                if (kind == "sgd_momentum") {
                    mf::sgd(p, g, 0.01f, 0.9f, 0.01f, &m);
                } else if (kind == "adam") {
                    mf::adam(p, g, m, v, 1e-3f, 1, 0.9f, 0.999f, 1e-8f, 0.01f);
                } else {
                    mf::adamw(p, g, m, v, 1e-3f, 1, 0.9f, 0.999f, 1e-8f, 0.01f);
                }
            };
            auto snapshot = [&] { return p.to_vector<float>(); };
            // One step from the start, in double, for the first 64 elements.
            const std::size_t head = std::min<std::size_t>(64, count);
            const ref::Vec pd(p0.begin(), p0.begin() + static_cast<std::ptrdiff_t>(head));
            const ref::Vec gd(g0.begin(), g0.begin() + static_cast<std::ptrdiff_t>(head));
            ref::Vec state1(head, 0.0), state2(head, 0.0);
            const ref::Vec expected = kind == "sgd_momentum"
                                          ? ref::sgd_step(pd, gd, state1, 0.01, 0.9, 0.01)
                                          : ref::adam_step(pd, gd, state1, state2, 1e-3, 1, 0.9,
                                                           0.999, 1e-8, 0.01, kind == "adamw");
            // sgd: read p, g, v and write p, v; adam: read p, g, m, v and write p, m, v.
            const double arrays = kind == "sgd_momentum" ? 5.0 : 7.0;
            sweep_in_place(table, config, shape_name({n}), thread_counts,
                           arrays * 4.0 * static_cast<double>(n), reset, run, snapshot, expected);
        }
    }
}

// ---- layer norm and RMS norm -----------------------------------------------------------------

void line_norm_section(bench::Table& table, const bench::Config& config,
                       const std::vector<int>& thread_counts) {
    // BERT-sized rows, a wide hidden size, a small batch, and 128 MB of activations.
    const std::int64_t shapes[][2] = {{32, 768}, {1024, 768}, {8192, 1024}, {32768, 1024}};
    const float eps = 1e-5f;
    for (const bool rms : {false, true}) {
        const std::string name = rms ? "rms_norm" : "layer_norm";
        for (const bool backward : {false, true}) {
            table.section(name + (backward ? "_backward" : "") + " [rows x width]");
            for (const auto& s : shapes) {
                const std::int64_t rows = s[0], width = s[1];
                const auto n = static_cast<std::size_t>(rows * width);
                bench::Arrays arrays(n, 4);
                float* x = arrays[0];
                float* dy = arrays[1];
                float* y = arrays[2];
                float* dx = arrays[3];
                fill_values(x, n, 1);
                fill_values(dy, n, 2);
                std::vector<float> w(static_cast<std::size_t>(width)), b(w.size()), dw(w.size()),
                    db(w.size());
                fill_positive(w.data(), w.size(), 3);
                fill_values(b.data(), b.size(), 4);
                const std::string label = shape_name({rows, width});
                // A few lines against the double reference.
                const std::int64_t head = std::min<std::int64_t>(rows, 4);
                const auto head_n = static_cast<std::size_t>(head * width);
                const ref::Vec xd(x, x + head_n), dyd(dy, dy + head_n), wd(w.begin(), w.end()),
                    bd(b.begin(), b.end());
                if (!backward) {
                    const ref::Vec expected = rms ? ref::rms_norm(xd, wd, head, width, eps)
                                                  : ref::layer_norm(xd, wd, bd, head, width, eps);
                    sweep(
                        table, config, label, thread_counts, 8.0 * static_cast<double>(n), {{y, n}},
                        [&] {
                            if (rms) {
                                cpu::rms_norm_f32(x, w.data(), y, rows, width, eps);
                            } else {
                                cpu::layer_norm_f32(x, w.data(), b.data(), y, rows, width, eps);
                            }
                        },
                        [&] {
                            for (std::size_t i = 0; i < head_n; ++i) {
                                bench::check_value(expected[i], y[i], 1e-4, label + " vs reference",
                                                   1.0);
                            }
                        });
                } else {
                    // dx for the first lines from the full Jacobian; dw and db for all lines from
                    // their definition in double (a streaming O(n) loop).
                    const auto g = rms ? ref::rms_norm_backward(xd, dyd, wd, head, width, eps)
                                       : ref::layer_norm_backward(xd, dyd, wd, head, width, eps);
                    sweep(
                        table, config, label, thread_counts, 12.0 * static_cast<double>(n),
                        {{dx, n}, {dw.data(), dw.size()}},
                        [&] {
                            if (rms) {
                                cpu::rms_norm_backward_f32(x, dy, w.data(), dx, dw.data(), rows,
                                                           width, eps);
                            } else {
                                cpu::layer_norm_backward_f32(x, dy, w.data(), dx, dw.data(),
                                                             db.data(), rows, width, eps);
                            }
                        },
                        [&] {
                            for (std::size_t i = 0; i < head_n; ++i) {
                                bench::check_value(g.dx[i], dx[i], 1e-3, label + " dx vs reference",
                                                   1.0);
                            }
                            // dw of the head lines only is checked in the unit test; here the
                            // all-lines dw is checked through its definition.
                            std::vector<double> total(w.size(), 0.0);
                            for (std::int64_t r = 0; r < rows; ++r) {
                                const float* xr = x + r * width;
                                double mean = 0.0, var = 0.0, ms = 0.0;
                                for (std::int64_t j = 0; j < width; ++j) {
                                    mean += xr[j];
                                    ms += static_cast<double>(xr[j]) * xr[j];
                                }
                                mean /= static_cast<double>(width);
                                ms /= static_cast<double>(width);
                                for (std::int64_t j = 0; j < width; ++j) {
                                    var += (xr[j] - mean) * (xr[j] - mean);
                                }
                                var /= static_cast<double>(width);
                                const double inv =
                                    rms ? 1.0 / std::sqrt(ms + eps) : 1.0 / std::sqrt(var + eps);
                                for (std::int64_t j = 0; j < width; ++j) {
                                    total[static_cast<std::size_t>(j)] +=
                                        static_cast<double>(dy[r * width + j]) *
                                        ((xr[j] - (rms ? 0.0 : mean)) * inv);
                                }
                            }
                            double scale = 0.0;
                            for (double t : total)
                                scale += std::abs(t);
                            for (std::size_t j = 0; j < w.size(); ++j) {
                                bench::check_value(total[j], dw[j], 1e-3,
                                                   label + " dw vs reference", scale);
                            }
                        });
                }
            }
        }
    }
}

// ---- batch norm ------------------------------------------------------------------------------

void batch_norm_section(bench::Table& table, const bench::Config& config,
                        const std::vector<int>& thread_counts) {
    // {n, c, spatial}: a ResNet stage (25 MB), a small late stage, and the two [N, C] layouts.
    const std::int64_t shapes[][3] = {
        {32, 64, 3136}, {8, 256, 196}, {1024, 1024, 1}, {256, 512, 1}};
    const float eps = 1e-5f, momentum = 0.1f;
    for (const int mode : {0, 1, 2}) { // 0 train, 1 eval, 2 train backward
        const char* title = mode == 0   ? "batch_norm, training"
                            : mode == 1 ? "batch_norm, evaluation"
                                        : "batch_norm_backward, training";
        table.section(std::string(title) + " [n x c x spatial]");
        for (const auto& s : shapes) {
            const std::int64_t n = s[0], c = s[1], sp = s[2];
            const auto total = static_cast<std::size_t>(n * c * sp);
            bench::Arrays arrays(total, 4);
            float* x = arrays[0];
            float* dy = arrays[1];
            float* y = arrays[2];
            float* dx = arrays[3];
            fill_values(x, total, 1);
            fill_values(dy, total, 2);
            std::vector<float> w(static_cast<std::size_t>(c)), b(w.size()), dw(w.size()),
                db(w.size()), running_mean(w.size()), running_var(w.size());
            fill_positive(w.data(), w.size(), 3);
            fill_values(b.data(), b.size(), 4);
            fill_values(running_mean.data(), w.size(), 5);
            fill_positive(running_var.data(), w.size(), 6);
            const std::string label = shape_name({n, c, sp});
            const std::vector<float> rm0 = running_mean, rv0 = running_var;

            // Reference: the first channels over their whole data (a channel needs every sample).
            const std::int64_t head_c = std::min<std::int64_t>(c, 2);
            ref::Vec xc, dyc;
            for (std::int64_t i = 0; i < n; ++i) {
                for (std::int64_t ch = 0; ch < head_c; ++ch) {
                    for (std::int64_t j = 0; j < sp; ++j) {
                        xc.push_back(x[(i * c + ch) * sp + j]);
                        dyc.push_back(dy[(i * c + ch) * sp + j]);
                    }
                }
            }
            const ref::Vec wd(w.begin(), w.begin() + head_c), bd(b.begin(), b.begin() + head_c),
                rmd(rm0.begin(), rm0.begin() + head_c), rvd(rv0.begin(), rv0.begin() + head_c);
            const bool training = mode != 1;
            const auto fwd =
                ref::batch_norm(xc, wd, bd, rmd, rvd, n, head_c, sp, training, momentum, eps);
            auto reference_y = [&](std::int64_t i, std::int64_t ch, std::int64_t j) {
                return fwd.y[static_cast<std::size_t>((i * head_c + ch) * sp + j)];
            };
            auto check_head = [&] {
                for (std::int64_t i = 0; i < std::min<std::int64_t>(n, 4); ++i) {
                    for (std::int64_t ch = 0; ch < head_c; ++ch) {
                        for (std::int64_t j = 0; j < std::min<std::int64_t>(sp, 8); ++j) {
                            bench::check_value(reference_y(i, ch, j), y[(i * c + ch) * sp + j],
                                               1e-4, label + " vs reference", 1.0);
                        }
                    }
                }
            };
            if (mode == 2) {
                // Backward needs the batch statistics only; the running ones are untouched.
                const auto grads =
                    ref::batch_norm_backward(xc, dyc, wd, rmd, rvd, n, head_c, sp, true, eps);
                sweep(
                    table, config, label, thread_counts, 12.0 * static_cast<double>(total),
                    {{dx, total}, {dw.data(), dw.size()}, {db.data(), db.size()}},
                    [&] {
                        cpu::batch_norm_backward_f32(x, dy, w.data(), running_mean.data(),
                                                     running_var.data(), dx, dw.data(), db.data(),
                                                     n, c, sp, true, eps);
                    },
                    [&] {
                        for (std::int64_t ch = 0; ch < head_c; ++ch) {
                            bench::check_value(grads.db[static_cast<std::size_t>(ch)],
                                               db[static_cast<std::size_t>(ch)], 1e-3,
                                               label + " db vs reference",
                                               static_cast<double>(n * sp) * 4.0);
                            bench::check_value(grads.dw[static_cast<std::size_t>(ch)],
                                               dw[static_cast<std::size_t>(ch)], 1e-3,
                                               label + " dw vs reference",
                                               static_cast<double>(n * sp) * 4.0);
                        }
                    });
            } else {
                // Training updates the running statistics on every call, which does not change the
                // work; the output does not depend on them in training, and evaluation never
                // changes them.
                sweep(
                    table, config, label, thread_counts, 8.0 * static_cast<double>(total),
                    {{y, total}},
                    [&] {
                        cpu::batch_norm_f32(x, w.data(), b.data(), running_mean.data(),
                                            running_var.data(), y, n, c, sp, training, momentum,
                                            eps);
                    },
                    check_head);
            }
        }
    }
}

// ---- public API ------------------------------------------------------------------------------

template <class Run>
void api_row(bench::Table& table, const bench::Config& config, const std::string& label,
             double work, const std::vector<int>& thread_counts, Run&& run) {
    mf::Tensor out;
    auto observe = [&] { return out.defined() ? static_cast<double>(out.numel()) : 0.0; };
    cpu::set_max_threads(1);
    const bench::Sample sequential = bench::measure(config, [&] { out = run(); }, observe);
    table.row(label, "sequential", "1", sequential, work, &sequential);
    for (const int threads : thread_counts) {
        if (threads == 1) {
            continue;
        }
        cpu::set_max_threads(threads);
        const bench::Sample parallel = bench::measure(config, [&] { out = run(); }, observe);
        table.row(label, "parallel", std::to_string(threads), parallel, work, &sequential);
    }
    cpu::set_max_threads(1);
    table.blank();
}

void api_section(bench::Table& table, const bench::Config& config,
                 const std::vector<int>& thread_counts) {
    table.section("public API (allocates its results): layer_norm, rms_norm, batch_norm [shape]");
    const std::int64_t rows = 8192, width = 1024;
    std::vector<float> xv(static_cast<std::size_t>(rows * width)),
        wv(static_cast<std::size_t>(width)), bv(wv.size());
    fill_values(xv.data(), xv.size(), 1);
    fill_positive(wv.data(), wv.size(), 3);
    fill_values(bv.data(), bv.size(), 4);
    const mf::Tensor x = mf::Tensor::from_vector(xv, {rows, width});
    const mf::Tensor w = mf::Tensor::from_vector(wv, {width}),
                     b = mf::Tensor::from_vector(bv, {width});
    const double bytes = 8.0 * static_cast<double>(rows * width);
    api_row(table, config, "layer_norm " + shape_name({rows, width}), bytes, thread_counts,
            [&] { return mf::layer_norm(x, {width}, &w, &b); });
    api_row(table, config, "rms_norm " + shape_name({rows, width}), bytes, thread_counts,
            [&] { return mf::rms_norm(x, {width}, &w); });
    const std::int64_t n = 32, c = 64, sp = 3136;
    std::vector<float> iv(static_cast<std::size_t>(n * c * sp));
    fill_values(iv.data(), iv.size(), 1);
    const mf::Tensor image = mf::Tensor::from_vector(iv, {n, c, 56, 56});
    mf::Tensor rm = mf::Tensor::zeros({c}), rv = mf::Tensor::ones({c});
    const mf::Tensor cw = mf::Tensor::ones({c}), cb = mf::Tensor::zeros({c});
    api_row(table, config, "batch_norm train " + shape_name({n, c, 56, 56}),
            8.0 * static_cast<double>(n * c * sp), thread_counts,
            [&] { return mf::batch_norm(image, rm, rv, &cw, &cb, true); });
}

} // namespace

int main(int argc, char** argv) {
    // Optional argument: run only the groups whose name contains it (optimizers, line_norm,
    // batch_norm, api), for quick experiments. Without it everything runs.
    const std::string only = argc > 1 ? argv[1] : "";
    auto wanted = [&](const char* group) {
        return only.empty() || std::string(group).find(only) != std::string::npos;
    };
    const bench::Config config;
    const std::vector<int> thread_counts = bench::thread_counts(cpu::max_threads());
    bench::print_header(
        "CPU optimizers and normalization, float32", config,
        {"sequential = one thread; parallel = the same kernel on N OpenMP threads",
         "rate is GB/s of bytes moved (inputs read once, outputs written once)",
         "optimizers go through the public API (in place); normalization calls the",
         "kernels on preallocated buffers, then the public API in the last section"});
    bench::Table table("GB/s", 1e9, "vs seq", "");
    if (wanted("optimizers"))
        optimizer_section(table, config, thread_counts);
    if (wanted("line_norm"))
        line_norm_section(table, config, thread_counts);
    if (wanted("batch_norm"))
        batch_norm_section(table, config, thread_counts);
    if (wanted("api"))
        api_section(table, config, thread_counts);
    cpu::set_max_threads(cpu::max_threads());
    return 0;
}
