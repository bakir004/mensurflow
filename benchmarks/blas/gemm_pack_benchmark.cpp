// Packing speed: the scalar reference against the hand-vectorized AVX2 packers, for the block sizes
// the GEMM uses. GB/s counts the bytes read plus the bytes written (a block of A or B is read once
// and written once). Packing is memory-to-memory work, so the ceiling is the cache and memory
// bandwidth, not the FMA units.
#include "bench.hpp"

#include "backends/cpu/ops/blas/gemm_pack.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dt = mensurflow::cpu::detail;

int main() {
    const bench::Config config;
    bench::print_header("Packing: scalar vs AVX2", config,
                        {"GB/s = (bytes read + bytes written) / median time, one thread"});
    bench::Table table("GB/s", 1e9, "vs scalar", "");
    table.section("pack_a (6-row panels, transpose) and pack_b (16-column panels, copy)");

    struct Case {
        const char* name;
        bool is_a;
        std::int64_t rows; // rows of the source block: mc for A, kc for B
        std::int64_t cols; // columns of the source block: kc for A, nc for B
        std::int64_t ld;   // row stride of the source matrix
    };
    const Case cases[] = {
        {"A 240x1024", true, 240, 1024, 2048},  {"A 240x256", true, 240, 256, 2048},
        {"A 24x1024", true, 24, 1024, 2048},    {"B 1024x4080", false, 1024, 4080, 8192},
        {"B 256x4080", false, 256, 4080, 8192}, {"B 1024x256", false, 1024, 256, 2048},
        {"B 1024x2048", false, 1024, 2048, 2048}, {"B 1024x4096", false, 1024, 4096, 4096},
    };
    for (const Case& c : cases) {
        std::vector<float> src(static_cast<std::size_t>(c.rows * c.ld));
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = static_cast<float>(i % 1013);
        }
        const std::int64_t panel = c.is_a ? 6 : 16;
        const std::int64_t extent = c.is_a ? c.rows : c.cols; // the dimension cut into panels
        const std::int64_t depth = c.is_a ? c.cols : c.rows;
        const std::int64_t panels = (extent + panel - 1) / panel;
        const std::size_t out_floats = static_cast<std::size_t>(panels * depth * panel);
        std::vector<float> out_scalar(out_floats + 64), out_vector(out_floats + 64);
        const double bytes = 2.0 * static_cast<double>(c.rows * c.cols) * sizeof(float);

        auto scalar = [&] {
            if (c.is_a) {
                dt::pack_a_scalar(6, src.data(), c.ld, c.rows, c.cols, out_scalar.data());
            } else {
                dt::pack_b_scalar(16, src.data(), c.ld, c.rows, c.cols, out_scalar.data());
            }
        };
        auto vector = [&] {
            if (c.is_a) {
                dt::pack_a(6, src.data(), c.ld, c.rows, c.cols, out_vector.data());
            } else {
                dt::pack_b(16, src.data(), c.ld, c.rows, c.cols, out_vector.data());
            }
        };
        scalar();
        vector();
        bench::check_bits(out_scalar.data(), out_vector.data(), out_floats, c.name);
        const bench::Sample s = bench::measure(
            config, scalar, [&] { return bench::checksum(out_scalar.data(), out_floats); });
        const bench::Sample v = bench::measure(
            config, vector, [&] { return bench::checksum(out_vector.data(), out_floats); });
        table.row(c.name, "scalar", "1", s, bytes, &s);
        table.row(c.name, "avx2", "1", v, bytes, &s);
        if (!c.is_a) {
            // Row-major B, several panels per pass over the rows (1 = one panel at a time).
            for (std::int64_t run : {1, 2, 4, 8}) {
                auto runs = [&] {
                    dt::pack_b(16, src.data(), c.ld, 1, c.rows, c.cols, out_vector.data(), run);
                };
                runs();
                bench::check_bits(out_scalar.data(), out_vector.data(), out_floats, c.name);
                const bench::Sample r = bench::measure(
                    config, runs, [&] { return bench::checksum(out_vector.data(), out_floats); });
                table.row(c.name, "avx2 run " + std::to_string(run), "1", r, bytes, &s);
            }
        }
        table.blank();
    }
    return 0;
}
