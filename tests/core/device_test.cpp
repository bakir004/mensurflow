// Tests device naming and parsing of MENSURFLOW_DEVICE-style strings.
#include "mensurflow/device.hpp"
#include "test_util.hpp"

#include <stdexcept>

using namespace mensurflow;

int main() {
    return test::run("device_test", [] {
        test::require(parse_device("cpu") == Device{DeviceType::CPU, 0}, "cpu");
        test::require(parse_device("CUDA:1") == Device{DeviceType::CUDA, 1}, "cuda:1");
        test::require(parse_device("vulkan:0") == Device{DeviceType::Vulkan, 0}, "vulkan:0");
        test::require(to_string(Device{DeviceType::Metal, 2}) == "metal:2", "to_string");
        // Round trip for every type.
        for (auto type : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm, DeviceType::Metal,
                          DeviceType::Vulkan}) {
            Device d{type, 3};
            test::require(parse_device(to_string(d)) == d, "round trip");
        }
        for (const char* bad : {"", "gpu", "cuda:", "cuda:-1", "cuda:1x", "cuda:a"}) {
            test::require_throws<std::invalid_argument>([bad] { parse_device(bad); }, bad);
        }
    });
}
