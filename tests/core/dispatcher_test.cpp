// Tests the kernel table and dispatch rules: lookup, duplicate registration, and the errors
// raised for missing kernels and unusable inputs.
#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "mensurflow/mensurflow.hpp"
#include "test_util.hpp"

#include <stdexcept>
#include <vector>

using namespace mensurflow;
using namespace mensurflow::detail;

namespace {

void noop_kernel(const OpArgs&, Stream&) {}

void test_table() {
    Dispatcher table;
    const KernelKey key{OpId::Dot, DeviceType::CUDA, DType::F32};
    test::require(table.find(key) == nullptr, "empty table finds nothing");
    table.register_kernel(key, noop_kernel);
    test::require(table.find(key) == noop_kernel, "registered kernel is found");
    test::require(table.lookup(key) == noop_kernel, "lookup returns it");

    // The key is the whole triple: other devices and dtypes stay unregistered.
    test::require(table.find({OpId::Dot, DeviceType::CPU, DType::F32}) == nullptr, "other device");
    test::require(table.find({OpId::Dot, DeviceType::CUDA, DType::I32}) == nullptr, "other dtype");

    test::require_throws<NotImplemented>(
        [&] { table.lookup({OpId::Dot, DeviceType::Metal, DType::F32}); }, "lookup of a miss");
    test::require_throws<std::logic_error>([&] { table.register_kernel(key, noop_kernel); },
                                           "duplicate registration");
    test::require_throws<std::invalid_argument>([&] { table.register_kernel(key, nullptr); },
                                                "null kernel");
}

void test_message_names_the_missing_kernel() {
    try {
        Dispatcher().lookup({OpId::Dot, DeviceType::ROCm, DType::BF16});
    } catch (const NotImplemented& e) {
        const std::string text = e.what();
        test::require(text.find("dot") != std::string::npos &&
                          text.find("rocm") != std::string::npos &&
                          text.find("bf16") != std::string::npos,
                      "message names op, device, and dtype: " + text);
        return;
    }
    test::require(false, "expected NotImplemented");
}

void test_registry() {
    Registry& registry = Registry::instance();
    test::require(registry.find(DeviceType::CPU) != nullptr, "cpu backend is always present");
    test::require(registry.get({DeviceType::CPU, 0}).device_count() == 1, "one cpu device");
    test::require_throws<std::runtime_error>([&] { registry.get({DeviceType::CPU, 1}); },
                                             "cpu:1 does not exist");
    test::require_throws<std::runtime_error>([&] { registry.get({DeviceType::CPU, -1}); },
                                             "negative index");
    // The CPU is the reference backend and must cover every op.
    for (OpId op : {OpId::Dot, OpId::Add, OpId::Sub, OpId::Mul, OpId::Div}) {
        test::require(registry.dispatcher().find({op, DeviceType::CPU, DType::F32}) != nullptr,
                      std::string("cpu ") + op_name(op) + " is registered");
    }
}

void test_dispatch_errors() {
    test::require_throws<std::invalid_argument>([] { dispatch(OpId::Dot, OpArgs{}); }, "no inputs");
    const Tensor ints = Tensor::from_vector(std::vector<int>{1, 2}, {2});
    test::require_throws<NotImplemented>(
        [&] { dispatch(OpId::Dot, {{ints, ints}, {Tensor::empty({}, DType::I32)}}); },
        "no int kernel");
}

} // namespace

int main() {
    return test::run("dispatcher_test", [] {
        test_table();
        test_message_names_the_missing_kernel();
        test_registry();
        test_dispatch_errors();
    });
}
