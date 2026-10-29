// Device identifies where a Tensor's memory lives and where its ops run.
// Detection (default_device, list_devices) is added with the backend registry.
#pragma once

#include <string>
#include <string_view>

namespace mensurflow {

enum class DeviceType { CPU, CUDA, ROCm, Metal, Vulkan };

// Value type, cheap to copy. index selects one GPU among several of the same type.
struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;
};

inline bool operator==(Device a, Device b) { return a.type == b.type && a.index == b.index; }

inline bool operator!=(Device a, Device b) { return !(a == b); }

// Stable lowercase name ("cpu", "cuda", ...); an invalid value throws invalid_argument.
const char* device_type_name(DeviceType type);

// "cuda:1" style string; index is always printed.
std::string to_string(Device device);

// Parse "cpu", "vulkan", "cuda:1" (case-insensitive). Missing index means 0.
// Used for the MENSURFLOW_DEVICE override. Malformed text throws invalid_argument.
Device parse_device(std::string_view text);

} // namespace mensurflow
