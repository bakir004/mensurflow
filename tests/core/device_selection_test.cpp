// Tests device detection and the MENSURFLOW_DEVICE override.
#include "mensurflow/device.hpp"
#include "test_util.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace mensurflow;

namespace {

void set_env(const char* value) {
#ifdef _WIN32
    _putenv((std::string("MENSURFLOW_DEVICE=") + (value ? value : "")).c_str());
#else
    if (value) {
        setenv("MENSURFLOW_DEVICE", value, 1);
    } else {
        unsetenv("MENSURFLOW_DEVICE");
    }
#endif
}

void test_listing() {
    const auto devices = list_devices();
    test::require(!devices.empty() && devices[0] == Device{DeviceType::CPU, 0},
                  "cpu:0 is listed first");
    for (Device d : devices) {
        test::require(device_available(d), to_string(d) + " listed so available");
    }
    test::require(!device_available({DeviceType::CPU, 1}), "cpu:1 is not available");
    test::require(!device_available({DeviceType::Vulkan, 0}), "no vulkan backend yet");
}

void test_default() {
    set_env(nullptr);
    test::require(device_available(default_device()), "default is available");

    set_env("cpu");
    test::require(default_device() == Device{DeviceType::CPU, 0}, "env override to cpu");
    set_env("CPU:0");
    test::require(default_device() == Device{DeviceType::CPU, 0}, "override is case-insensitive");

    // An explicit choice never falls back to something else.
    set_env("cpu:99");
    test::require_throws<std::runtime_error>([] { default_device(); }, "unavailable override");
    set_env("gpu");
    test::require_throws<std::invalid_argument>([] { default_device(); }, "malformed override");
    set_env(nullptr);
}

} // namespace

int main() {
    return test::run("device_selection_test", [] {
        test_listing();
        test_default();
    });
}
