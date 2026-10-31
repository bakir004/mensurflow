// Device identifies where a Tensor's memory lives and where its ops run.
// Naming and parsing are in device.cpp; detection (list_devices, default_device) is in
// device_selection.cpp because it needs the backend registry.
#pragma once

#include <string>
#include <string_view>
#include <vector>

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

// Devices this build can use right now: a backend compiled in and hardware found. CPU is always
// first. A GPU backend whose driver or library is missing simply does not appear.
std::vector<Device> list_devices();

bool device_available(Device device);

// The device to use when the caller does not choose one. MENSURFLOW_DEVICE ("cuda:0", "cpu")
// overrides everything and throws if it names an unavailable device. Otherwise the first of
// CUDA, ROCm, Metal, Vulkan, then CPU that is available. Explicit choices never fall back.
Device default_device();

} // namespace mensurflow
