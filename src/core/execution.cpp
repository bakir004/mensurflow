#include "mensurflow/execution.hpp"

#include <stdexcept>

namespace mensurflow {

const char* execution_name(Execution execution) {
    switch (execution) {
    case Execution::Auto:
        return "auto";
    case Execution::CPU:
        return "cpu";
    case Execution::GPU:
        return "gpu";
    }
    throw std::invalid_argument("invalid Execution value");
}

} // namespace mensurflow
