// Data movement: concat, pad and split, against memcpy of the same bytes as the ceiling.
//   sequential  the kernel on one thread (the baseline).
//   parallel    the same kernel with N OpenMP threads.
//   memcpy      one std::memcpy of the output's size: the fastest one thread can move those bytes.
// split returns views, so its row times split + contiguous() of every piece (what a caller that
// needs packed pieces pays); the views alone cost the same at every size.
// Rate is GB/s of bytes moved: every element read once and written once (fill elements written
// only). "vs seq" divides the sequential median time by this row's.
//
//   movement_benchmark
#include "bench.hpp"

#include "backends/cpu/ops/movement/movement.hpp"
#include "backends/cpu/threads.hpp"
#include "mensurflow/mensurflow.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = mensurflow::cpu;
namespace mf = mensurflow;

namespace {

std::string mb(double bytes) {
    return std::to_string(static_cast<long long>(bytes / 1e6 + 0.5)) + " MB";
}

// One table row per thread count for `run`, plus the sequential baseline and the memcpy ceiling.
template <class Run>
void sweep(bench::Table& table, const bench::Config& config, const std::string& label,
           const std::vector<int>& thread_counts, double bytes, std::size_t out_floats,
           const float* out, Run&& run) {
    std::vector<float> scratch_in(out_floats), scratch_out(out_floats);
    bench::fill_a(scratch_in.data(), out_floats);
    const bench::Sample copy = bench::measure(
        config,
        [&] { std::memcpy(scratch_out.data(), scratch_in.data(), out_floats * sizeof(float)); },
        [&] { return static_cast<double>(scratch_out[out_floats / 2]); });
    table.row(label, "memcpy", "1", copy, 2.0 * static_cast<double>(out_floats) * sizeof(float));

    cpu::set_max_threads(1);
    run();
    const std::vector<float> expected(out, out + out_floats);
    auto observe = [&] { return bench::checksum(out, out_floats); };
    const bench::Sample sequential = bench::measure(config, run, observe);
    table.row(label, "sequential", "1", sequential, bytes, &sequential);
    for (int threads : thread_counts) {
        if (threads == 1) {
            continue; // that is the sequential row
        }
        cpu::set_max_threads(threads);
        run();
        bench::check_bits(expected.data(), out, out_floats,
                          label + " threads=" + std::to_string(threads));
        const bench::Sample parallel = bench::measure(config, run, observe);
        table.row(label, "parallel", std::to_string(threads), parallel, bytes, &sequential);
    }
}

} // namespace

int main() {
    const bench::Config config;
    const int maximum_threads = cpu::max_threads();
    const std::vector<int> thread_counts = bench::thread_counts(maximum_threads);
    bench::print_header("CPU data movement: concat, pad, split, float32", config,
                        {"sequential = one thread; parallel = the same kernel on N OpenMP threads",
                         "memcpy = one thread copying the output's bytes (the ceiling)",
                         "rate is GB/s: input read once and output written once"});
    bench::Table table("GB/s", 1e9, "vs seq", "");

    // ---- concat: two halves along axis 0, and 8 pieces along the last axis -------------------
    table.section("concat, 2 inputs along axis 0");
    for (const std::int64_t n : {1000, 32768, 1 << 20, 1 << 24}) {
        const std::int64_t half = n / 2;
        bench::Arrays arrays(static_cast<std::size_t>(n), 3);
        bench::fill_a(arrays[0], static_cast<std::size_t>(half));
        bench::fill_b(arrays[1], static_cast<std::size_t>(n - half));
        const float* inputs[] = {arrays[0], arrays[1]};
        const std::int64_t widths[] = {half, n - half};
        float* out = arrays[2];
        sweep(table, config, std::to_string(n) + " (" + mb(8.0 * static_cast<double>(n)) + ")",
              thread_counts, 8.0 * static_cast<double>(n), static_cast<std::size_t>(n), out,
              [&] { cpu::concat_f32(inputs, widths, 2, out, 1); });
        table.blank();
    }

    table.section("concat, 8 inputs of [rows, 128] along the last axis");
    for (const std::int64_t rows : {1, 64, 4096, 16384}) {
        constexpr int count = 8;
        const std::int64_t width = 128;
        const auto piece = static_cast<std::size_t>(rows * width);
        bench::Arrays arrays(piece * count, count + 1);
        std::vector<const float*> inputs;
        std::vector<std::int64_t> widths(count, width);
        for (int t = 0; t < count; ++t) {
            bench::fill_a(arrays[t], piece);
            inputs.push_back(arrays[t]);
        }
        float* out = arrays[count];
        const double total = static_cast<double>(piece) * count;
        sweep(table, config, std::to_string(rows) + "x1024 (" + mb(8.0 * total) + ")",
              thread_counts, 8.0 * total, piece * count, out,
              [&] { cpu::concat_f32(inputs.data(), widths.data(), count, out, rows); });
        table.blank();
    }

    // ---- pad: a matrix by 1 on every side, and images by 1 pixel ------------------------------
    table.section("pad [n, n] by 1 on every side");
    for (const std::int64_t n : {32, 181, 1024, 4096}) {
        const std::int64_t shape[] = {n, n};
        const std::int64_t before[] = {1, 1};
        const std::int64_t after[] = {1, 1};
        const auto in_floats = static_cast<std::size_t>(n * n);
        const auto out_floats = static_cast<std::size_t>((n + 2) * (n + 2));
        bench::Arrays arrays(out_floats, 2);
        bench::fill_a(arrays[0], in_floats);
        float* out = arrays[1];
        sweep(table, config,
              std::to_string(n) + "x" + std::to_string(n) + " (" +
                  mb(4.0 * static_cast<double>(in_floats + out_floats)) + ")",
              thread_counts, 4.0 * static_cast<double>(in_floats + out_floats), out_floats, out,
              [&] { cpu::pad_f32(arrays[0], out, shape, before, after, 2, 0.0f); });
        table.blank();
    }

    table.section("pad images [batch, 64, 56, 56] by 1 pixel (conv input)");
    for (const std::int64_t batch : {1, 8, 32}) {
        const std::int64_t shape[] = {batch, 64, 56, 56};
        const std::int64_t before[] = {0, 0, 1, 1};
        const std::int64_t after[] = {0, 0, 1, 1};
        const auto in_floats = static_cast<std::size_t>(batch * 64 * 56 * 56);
        const auto out_floats = static_cast<std::size_t>(batch * 64 * 58 * 58);
        bench::Arrays arrays(out_floats, 2);
        bench::fill_a(arrays[0], in_floats);
        float* out = arrays[1];
        sweep(table, config,
              std::to_string(batch) + "x64x56x56 (" +
                  mb(4.0 * static_cast<double>(in_floats + out_floats)) + ")",
              thread_counts, 4.0 * static_cast<double>(in_floats + out_floats), out_floats, out,
              [&] { cpu::pad_f32(arrays[0], out, shape, before, after, 4, 0.0f); });
        table.blank();
    }

    // ---- split: views, then packing every piece -------------------------------------------------
    table.section("split [4096, 1024] into 8 column pieces, then contiguous() each");
    {
        std::vector<float> data(4096 * 1024);
        bench::fill_a(data.data(), data.size());
        const mf::Tensor x = mf::Tensor::from_vector(data, {4096, 1024});
        double sink = 0.0;
        const bench::Sample views = bench::measure(
            config, [&] { sink = static_cast<double>(mf::split(x, 128, 1).size()); },
            [&] { return sink; });
        table.row("4096x1024", "split (views)", "1", views, 0.0);
        for (int threads : {1, maximum_threads}) {
            cpu::set_max_threads(threads);
            const bench::Sample packed = bench::measure(
                config,
                [&] {
                    sink = 0.0;
                    for (const mf::Tensor& piece : mf::split(x, 128, 1)) {
                        sink += static_cast<double>(piece.contiguous().numel());
                    }
                },
                [&] { return sink; });
            table.row("4096x1024", "split+contiguous", std::to_string(threads), packed,
                      8.0 * static_cast<double>(data.size()));
        }
        table.blank();
    }
    cpu::set_max_threads(maximum_threads);
    return 0;
}
