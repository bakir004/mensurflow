// Device detection and the default-device policy. Kept apart from device.cpp, which has no
// backend dependencies.
#include "mensurflow/device.hpp"

#include "core/registry.hpp"

#include <cstdlib>
#include <stdexcept>

namespace mensurflow {

std::vector<Device> list_devices() { return detail::Registry::instance().devices(); }

bool device_available(Device device) {
    const auto devices = list_devices();
    for (Device d : devices) {
        if (d == device) {
            return true;
        }
    }
    return false;
}

Device default_device() {
    if (const char* env = std::getenv("MENSURFLOW_DEVICE")) {
        const Device wanted = parse_device(env);
        if (!device_available(wanted)) {
            throw std::runtime_error("MENSURFLOW_DEVICE=" + std::string(env) + " is not available");
        }
        return wanted;
    }
    // Vendor libraries first, CPU last. This only checks that a device exists, not that it is
    // faster for a given op.
    for (DeviceType type :
         {DeviceType::CUDA, DeviceType::ROCm, DeviceType::Metal, DeviceType::Vulkan}) {
        if (device_available({type, 0})) {
            return {type, 0};
        }
    }
    return {DeviceType::CPU, 0};
}

} // namespace mensurflow
