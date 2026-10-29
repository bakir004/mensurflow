// Device naming and parsing. Kept separate from detection so it has no backend dependencies.
#include "mensurflow/device.hpp"

#include <cctype>
#include <stdexcept>

namespace mensurflow {

namespace {

constexpr DeviceType all_types[] = {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm,
                                    DeviceType::Metal, DeviceType::Vulkan};

std::string lowercase(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

} // namespace

const char* device_type_name(DeviceType type) {
    switch (type) {
    case DeviceType::CPU:
        return "cpu";
    case DeviceType::CUDA:
        return "cuda";
    case DeviceType::ROCm:
        return "rocm";
    case DeviceType::Metal:
        return "metal";
    case DeviceType::Vulkan:
        return "vulkan";
    }
    throw std::invalid_argument("invalid DeviceType");
}

std::string to_string(Device device) {
    return std::string(device_type_name(device.type)) + ":" + std::to_string(device.index);
}

Device parse_device(std::string_view text) {
    const std::string lower = lowercase(text);
    const auto colon = lower.find(':');
    const std::string name = lower.substr(0, colon);

    Device device;
    bool found = false;
    for (DeviceType type : all_types) {
        if (name == device_type_name(type)) {
            device.type = type;
            found = true;
        }
    }
    if (!found) {
        throw std::invalid_argument("unknown device type: " + std::string(text));
    }

    if (colon != std::string::npos) {
        const std::string index = lower.substr(colon + 1);
        // Digits only: rejects "", "-1", "1x", and anything stoi would partially accept.
        if (index.empty() || index.size() > 6 ||
            index.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("invalid device index: " + std::string(text));
        }
        device.index = std::stoi(index);
    }
    return device;
}

} // namespace mensurflow
