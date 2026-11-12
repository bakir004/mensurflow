// Optional tracing. With MENSURFLOW_ENABLE_TRACING defined (CMake: -DMENSURFLOW_ENABLE_TRACING=ON),
// MENSURFLOW_TRACE(...) opens a span that lasts until the end of the enclosing scope and hands one
// Event to `hook` when it closes. Without it the macro expands to nothing and its arguments are never
// evaluated, so a normal build pays nothing.
//
// The library only produces events; it never stores or prints them. Whoever wants them sets `hook`
// (the observability tool does, see observability/). With no hook set, a span costs one atomic load.
// GPU kernels may still be running when their span closes: the duration is the time spent in the call.
#pragma once

#ifdef MENSURFLOW_ENABLE_TRACING

#include "mensurflow/device.hpp"
#include "mensurflow/dtype.hpp"
#include "mensurflow/shape.hpp"
#include "mensurflow/tensor.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mensurflow::trace {

struct Event {
    const char* name;           // op name, e.g. "matmul"
    Device device;              // where it ran
    DType dtype;                // dtype the kernel was chosen for
    std::vector<Shape> inputs;  // input shapes
    std::vector<Shape> outputs; // output shapes
    std::int64_t start_ns;      // steady_clock
    std::int64_t end_ns;
};

using Hook = void (*)(const Event& event);

inline std::atomic<Hook> hook{nullptr};

class Span {
  public:
    Span(const char* name, Device device, const std::vector<Tensor>& inputs,
         const std::vector<Tensor>& outputs)
        : active_(hook.load(std::memory_order_relaxed) != nullptr) {
        if (!active_) {
            return;
        }
        event_.name = name;
        event_.device = device;
        const std::vector<Tensor>& first = inputs.empty() ? outputs : inputs;
        event_.dtype = first.empty() ? DType::F32 : first[0].dtype();
        for (const Tensor& t : inputs) {
            event_.inputs.push_back(t.shape());
        }
        for (const Tensor& t : outputs) {
            event_.outputs.push_back(t.shape());
        }
        event_.start_ns = now();
    }

    // A step on one tensor: allocation, a copy between devices.
    Span(const char* name, Device device, const Shape& shape, DType dtype)
        : active_(hook.load(std::memory_order_relaxed) != nullptr) {
        if (!active_) {
            return;
        }
        event_.name = name;
        event_.device = device;
        event_.dtype = dtype;
        event_.outputs.push_back(shape);
        event_.start_ns = now();
    }

    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;

    ~Span() {
        if (!active_) {
            return;
        }
        event_.end_ns = now();
        if (Hook h = hook.load(std::memory_order_relaxed)) {
            h(event_);
        }
    }

  private:
    static std::int64_t now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    bool active_;
    Event event_{};
};

} // namespace mensurflow::trace

#define MENSURFLOW_TRACE(...) ::mensurflow::trace::Span mensurflow_trace_span_(__VA_ARGS__)

#else

#define MENSURFLOW_TRACE(...) ((void)0)

#endif
