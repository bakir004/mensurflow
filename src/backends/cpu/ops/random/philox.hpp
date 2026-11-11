// Philox4x32-10, the counter-based generator described by Salmon et al. (SC11).
// Algorithm constants and known-answer data: DEShawResearch/random123, philox.h and
// tests/kat_vectors. This is a standalone implementation, without a Random123 dependency.
#pragma once

#include <array>
#include <cstdint>

namespace mensurflow::cpu {

using PhiloxCounter = std::array<std::uint32_t, 4>;
using PhiloxKey = std::array<std::uint32_t, 2>;

inline PhiloxCounter philox4x32_10(PhiloxCounter counter, PhiloxKey key) {
    for (int round = 0; round < 10; ++round) {
        const std::uint64_t p0 = std::uint64_t{0xD2511F53u} * counter[0];
        const std::uint64_t p1 = std::uint64_t{0xCD9E8D57u} * counter[2];
        counter = {static_cast<std::uint32_t>(p1 >> 32) ^ counter[1] ^ key[0],
                   static_cast<std::uint32_t>(p1),
                   static_cast<std::uint32_t>(p0 >> 32) ^ counter[3] ^ key[1],
                   static_cast<std::uint32_t>(p0)};
        key[0] += 0x9E3779B9u;
        key[1] += 0xBB67AE85u;
    }
    return counter;
}

// Stream block b uses counter {low32(b), high32(b), 0, 0}; seed provides both key words.
inline PhiloxCounter philox_block(std::uint64_t seed, std::uint64_t block) {
    return philox4x32_10({static_cast<std::uint32_t>(block),
                         static_cast<std::uint32_t>(block >> 32), 0, 0},
                        {static_cast<std::uint32_t>(seed),
                         static_cast<std::uint32_t>(seed >> 32)});
}

} // namespace mensurflow::cpu
