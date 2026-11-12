// Collects mensurflow trace events in memory and writes them as Chrome trace-event JSON, which the
// dashboard (observability/index.html), chrome://tracing and https://ui.perfetto.dev all open.
//
// Any program can use it; nothing here is part of the library:
//
//     #include "trace_collector.hpp"
//     observe::TraceCollector trace;   // starts collecting (needs -DMENSURFLOW_ENABLE_TRACING=ON)
//     ... call mensurflow ops ...
//     trace.write("my_run.trace.json"); // load the file in the dashboard's Trace tab
//
// Without tracing compiled in, the collector stays empty and enabled() is false.
#pragma once

#include "mensurflow/mensurflow.hpp"
#include "mensurflow/trace.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace observe {

struct Record {
    std::string name;
    std::string device;
    std::string dtype;
    std::vector<mensurflow::Shape> inputs;
    std::vector<mensurflow::Shape> outputs;
    std::int64_t start_ns = 0;
    std::int64_t end_ns = 0;
    int thread = 0;
};

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
                out += buffer;
            } else {
                out += c;
            }
        }
    }
    return out;
}

inline std::string shape_json(const mensurflow::Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out += (i ? "," : "") + std::to_string(shape[i]);
    }
    return out + "]";
}

class TraceCollector {
  public:
    TraceCollector() { start(); }

    ~TraceCollector() { stop(); }

    TraceCollector(const TraceCollector&) = delete;
    TraceCollector& operator=(const TraceCollector&) = delete;

    static constexpr bool enabled() {
#ifdef MENSURFLOW_ENABLE_TRACING
        return true;
#else
        return false;
#endif
    }

    void start() {
#ifdef MENSURFLOW_ENABLE_TRACING
        current() = this;
        mensurflow::trace::hook.store(&on_event);
#endif
    }

    void stop() {
#ifdef MENSURFLOW_ENABLE_TRACING
        if (current() == this) {
            mensurflow::trace::hook.store(nullptr);
            current() = nullptr;
        }
#endif
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
    }

    // Adds an event measured outside the library (the caller's own spans), on the calling thread.
    void add(Record r) {
        std::lock_guard<std::mutex> lock(mutex_);
        r.thread = thread_index();
        records_.push_back(std::move(r));
    }

    std::vector<Record> records() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_;
    }

    // The events as a JSON array of Chrome "complete" events (ph "X", microseconds). Times are
    // relative to `origin_ns`, or to the first event when it is 0.
    std::string events_json(std::int64_t origin_ns = 0) const {
        const std::vector<Record> all = records();
        if (origin_ns == 0 && !all.empty()) {
            origin_ns = all.front().start_ns;
            for (const Record& r : all) {
                origin_ns = r.start_ns < origin_ns ? r.start_ns : origin_ns;
            }
        }
        std::ostringstream out;
        out << "[";
        for (std::size_t i = 0; i < all.size(); ++i) {
            const Record& r = all[i];
            out << (i ? ",\n" : "\n") << "{\"name\":\"" << json_escape(r.name)
                << "\",\"ph\":\"X\",\"pid\":0,\"tid\":" << r.thread
                << ",\"ts\":" << static_cast<double>(r.start_ns - origin_ns) / 1e3
                << ",\"dur\":" << static_cast<double>(r.end_ns - r.start_ns) / 1e3
                << ",\"args\":{\"device\":\"" << json_escape(r.device) << "\",\"dtype\":\""
                << json_escape(r.dtype) << "\",\"inputs\":[";
            for (std::size_t k = 0; k < r.inputs.size(); ++k) {
                out << (k ? "," : "") << shape_json(r.inputs[k]);
            }
            out << "],\"outputs\":[";
            for (std::size_t k = 0; k < r.outputs.size(); ++k) {
                out << (k ? "," : "") << shape_json(r.outputs[k]);
            }
            out << "]}}";
        }
        out << "]";
        return out.str();
    }

    bool write(const std::string& path) const {
        std::ofstream file(path);
        file << "{\"traceEvents\":" << events_json() << ",\"displayTimeUnit\":\"ms\"}\n";
        return static_cast<bool>(file);
    }

  private:
    static TraceCollector*& current() {
        static TraceCollector* collector = nullptr;
        return collector;
    }

#ifdef MENSURFLOW_ENABLE_TRACING
    static void on_event(const mensurflow::trace::Event& event) {
        TraceCollector* self = current();
        if (self == nullptr) {
            return;
        }
        Record r;
        r.name = event.name;
        r.device = mensurflow::to_string(event.device);
        r.dtype = mensurflow::dtype_name(event.dtype);
        r.inputs = event.inputs;
        r.outputs = event.outputs;
        r.start_ns = event.start_ns;
        r.end_ns = event.end_ns;
        std::lock_guard<std::mutex> lock(self->mutex_);
        r.thread = self->thread_index();
        self->records_.push_back(std::move(r));
    }
#endif

    // Small stable number for the calling thread. Caller holds mutex_.
    int thread_index() {
        const auto id = std::this_thread::get_id();
        auto it = threads_.find(id);
        if (it == threads_.end()) {
            it = threads_.emplace(id, static_cast<int>(threads_.size())).first;
        }
        return it->second;
    }

    mutable std::mutex mutex_;
    std::vector<Record> records_;
    std::map<std::thread::id, int> threads_;
};

} // namespace observe
