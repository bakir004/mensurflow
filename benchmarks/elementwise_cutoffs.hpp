// Common cutoff override for the existing elementwise benchmark executables.
#pragma once

#include "backends/cpu/ops/elementwise/parallel_cutoffs.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace bench {

struct ElementwiseCutoffOption {
    bool specified = false;
    std::int64_t value = 0;

    bool parse(const std::string& argument) {
        if (argument.rfind("--cutoff=", 0) != 0) {
            return false;
        }
        const auto text = argument.substr(9);
        if (specified || text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("--cutoff requires one nonnegative int64 value");
        }
        value = std::stoll(text);
        specified = true;
        return true;
    }

    // Called once per selected operation before correctness checks or timing. The printed value
    // comes from the same library state the kernel reads, including any override.
    std::string configure(const char* name) const {
        namespace cpu = mensurflow::cpu;
        for (const auto& policy : cpu::elementwise_parallel_policies) {
            if (std::string(name) == policy.name) {
                if (specified) {
                    cpu::set_elementwise_parallel_cutoff(policy.op, value);
                }
                return std::string(name) + "=" +
                       std::to_string(cpu::elementwise_parallel_cutoff(policy.op));
            }
        }
        throw std::invalid_argument(std::string("no elementwise cutoff for ") + name);
    }
};

} // namespace bench
