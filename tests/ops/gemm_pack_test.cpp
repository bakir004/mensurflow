// Tests the hand-vectorized packing (gemm_pack.hpp) against the scalar reference: every panel shape
// the GEMM uses, sizes around the vector widths (8 depth steps for A, 4 for B), partial panels,
// strided sources, and guard words after the destination that must stay untouched (panels are packed
// by different threads, so a write past the end of a panel would be a data race).
#include "backends/cpu/ops/blas/gemm_pack.hpp"
#include "test_util.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dt = mensurflow::cpu::detail;

namespace {

constexpr float guard_value = -12345.0f;
constexpr float poison_value = 777777.0f;
constexpr std::size_t guard = 64;

std::vector<float> source(std::int64_t rows, std::int64_t ld) {
    std::vector<float> v(static_cast<std::size_t>(rows * ld));
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>(i % 9973) + 0.5f; // distinct enough to catch a misplaced element
    }
    return v;
}

// Destination: the panel area pre-filled with a poison value (so a missed element or missing zero
// padding is visible), then `guard` words that must never change.
std::vector<float> destination(std::size_t floats) {
    std::vector<float> out(floats + guard, guard_value);
    for (std::size_t i = 0; i < floats; ++i) {
        out[i] = poison_value;
    }
    return out;
}

void compare(const std::vector<float>& vec, const std::vector<float>& ref, std::size_t floats,
             const std::string& what) {
    for (std::size_t i = 0; i < floats; ++i) {
        test::require(vec[i] == ref[i], what + ": element " + std::to_string(i) + " is " +
                                            std::to_string(vec[i]) + ", expected " +
                                            std::to_string(ref[i]));
    }
    for (std::size_t i = floats; i < floats + guard; ++i) {
        test::require(vec[i] == guard_value, what + ": wrote past the end at " + std::to_string(i));
    }
}

} // namespace

int main() {
    return test::run("gemm_pack_test", [] {
        const std::int64_t mcs[] = {1, 5, 6, 7, 12, 13, 50};
        const std::int64_t ncs[] = {1, 7, 8, 9, 16, 17, 23, 24, 25, 48, 100};
        const std::int64_t kcs[] = {1, 3, 4, 5, 7, 8, 9, 16, 17, 100};

        for (std::int64_t mr : {6, 4, 12}) {
            for (std::int64_t mc : mcs) {
                for (std::int64_t kc : kcs) {
                    const std::int64_t lda = kc + 5; // strided: rows are longer than the block
                    const auto a = source(mc, lda);
                    const std::int64_t panels = (mc + mr - 1) / mr;
                    const std::size_t floats = static_cast<std::size_t>(panels * kc * mr);
                    auto vec = destination(floats);
                    auto ref = destination(floats);
                    dt::pack_a(mr, a.data(), lda, mc, kc, vec.data());
                    dt::pack_a_scalar(mr, a.data(), lda, mc, kc, ref.data());
                    compare(vec, ref, floats,
                            "pack_a mr=" + std::to_string(mr) + " mc=" + std::to_string(mc) +
                                " kc=" + std::to_string(kc));
                }
            }
        }

        for (std::int64_t nr : {16, 24, 8}) {
            for (std::int64_t nc : ncs) {
                for (std::int64_t kc : kcs) {
                    const std::int64_t ldb = nc + 3;
                    const auto b = source(kc, ldb);
                    const std::int64_t panels = (nc + nr - 1) / nr;
                    const std::size_t floats = static_cast<std::size_t>(panels * kc * nr);
                    auto vec = destination(floats);
                    auto ref = destination(floats);
                    dt::pack_b(nr, b.data(), ldb, kc, nc, vec.data());
                    dt::pack_b_scalar(nr, b.data(), ldb, kc, nc, ref.data());
                    compare(vec, ref, floats,
                            "pack_b nr=" + std::to_string(nr) + " nc=" + std::to_string(nc) +
                                " kc=" + std::to_string(kc));
                }
            }
        }
    });
}
