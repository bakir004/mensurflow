// Shared OpenCL helpers for device discovery, one cached context and queue, runtime kernel
// compilation, buffer ownership, and synchronous error reporting in both operations.
#pragma once
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace prs::detail {
// OpenCL reports failures as negative integers; name the common ones for readable errors.
inline const char* opencl_error_name(cl_int status) {
    switch (status) {
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    default:
        return "OpenCL error";
    }
}

// Include the failed operation in errors reported through the public API.
inline void opencl_check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + opencl_error_name(status) + " (" +
                                 std::to_string(status) + ")");
    }
}

// The platform/device pair chosen once per process. found is false when none qualifies.
struct OpenCLDevice {
    bool found = false;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};

// Pick the first usable GPU. Software runtimes such as PoCL expose the CPU through OpenCL;
// they are ignored unless PRS_OPENCL_ALLOW_CPU is set, so Auto never prefers a slow CPU
// device over the native CPU backend. The variable also lets CI test this backend without a GPU.
inline OpenCLDevice opencl_discover() {
    const char* allow_cpu = std::getenv("PRS_OPENCL_ALLOW_CPU");
    const cl_device_type type = (allow_cpu && *allow_cpu) ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_GPU;
    cl_uint platform_count = 0;
    // With no installed driver (ICD) the loader reports an error rather than zero platforms.
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
        return {};
    }
    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return {};
    }
    for (cl_platform_id platform : platforms) {
        cl_device_id device = nullptr;
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platform, type, 1, &device, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }
        // Kernels are compiled at run time, so the device must offer a compiler.
        cl_bool usable = CL_FALSE, compiler = CL_FALSE;
        clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(usable), &usable, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, nullptr);
        if (usable && compiler) {
            return {true, platform, device};
        }
    }
    return {};
}

inline const OpenCLDevice& opencl_device() {
    // Discovery can be slow (driver loading), so it runs once and available() reuses it.
    static const OpenCLDevice selected = opencl_discover();
    return selected;
}

struct OpenCLContext {
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::uint64_t max_allocation = 0;

    OpenCLContext() {
        const OpenCLDevice& selected = opencl_device();
        if (!selected.found) {
            throw std::runtime_error("No OpenCL GPU device available");
        }
        device = selected.device;
        cl_int status = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        opencl_check(status, "Create OpenCL context");
        // An in-order queue (properties 0) runs commands in submission order.
        queue = clCreateCommandQueue(context, device, 0, &status);
        if (status != CL_SUCCESS) {
            clReleaseContext(context);
            opencl_check(status, "Create OpenCL command queue");
        }
        // A single buffer may not exceed this limit, often only a quarter of device memory.
        cl_ulong limit = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(limit), &limit, nullptr);
        max_allocation = limit;
    }
};

inline OpenCLContext& opencl_context() {
    // Lazily create one context and queue shared by AXPY and matmul calls. It is deliberately
    // never destroyed: drivers may already be unloaded when static destructors run at exit.
    static OpenCLContext* context = new OpenCLContext();
    return *context;
}

// Operation-local kernel() functions cache the compiled result of this helper.
inline cl_kernel opencl_kernel(const char* source, const char* name) {
    auto& ctx = opencl_context();
    cl_int status = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ctx.context, 1, &source, nullptr, &status);
    opencl_check(status, "Create OpenCL program");
    status = clBuildProgram(program, 1, &ctx.device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        // The compiler's message is the only useful clue when a kernel fails to build.
        std::size_t size = 0;
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
        std::string log(size, '\0');
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("Compile OpenCL kernel: " + log);
    }
    cl_kernel kernel = clCreateKernel(program, name, &status);
    // The kernel keeps its program alive, so this releases only our own reference.
    clReleaseProgram(program);
    opencl_check(status, "Create OpenCL kernel");
    return kernel;
}

// Largest work-group this compiled kernel can launch on the device (register use limits it).
inline std::size_t opencl_group_limit(cl_kernel kernel) {
    std::size_t limit = 0;
    opencl_check(clGetKernelWorkGroupInfo(kernel, opencl_context().device,
                                          CL_KERNEL_WORK_GROUP_SIZE, sizeof(limit), &limit,
                                          nullptr),
                 "Query OpenCL work-group size");
    return limit;
}

// Kernel arguments are copied by value, so a temporary may be passed safely.
template <typename T> void opencl_arg(cl_kernel kernel, cl_uint index, const T& value) {
    opencl_check(clSetKernelArg(kernel, index, sizeof(T), &value), "Set OpenCL kernel argument");
}

inline void opencl_check_count(std::size_t count) {
    // Kernels use uint addressing; reject buffers that would wrap an index or exceed the device.
    if (count > std::numeric_limits<std::uint32_t>::max() ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        count * sizeof(float) > opencl_context().max_allocation) {
        throw std::runtime_error("Operation exceeds OpenCL buffer/index limits");
    }
}

// RAII releases device memory when an operation returns or throws.
// Copying is disabled so a device allocation always has one owner.
struct OpenCLBuffer {
    cl_mem handle = nullptr;

    // With host input the contents are copied at creation; output-only buffers start uninitialized.
    OpenCLBuffer(cl_mem_flags access, std::size_t count, const float* input = nullptr) {
        opencl_check_count(count);
        cl_int status = CL_SUCCESS;
        const cl_mem_flags flags = input ? (access | CL_MEM_COPY_HOST_PTR) : access;
        handle = clCreateBuffer(opencl_context().context, flags, count * sizeof(float),
                                const_cast<float*>(input), &status);
        opencl_check(status, "Allocate OpenCL buffer");
    }

    ~OpenCLBuffer() {
        if (handle) {
            clReleaseMemObject(handle);
        }
    }

    OpenCLBuffer(const OpenCLBuffer&) = delete;

    OpenCLBuffer& operator=(const OpenCLBuffer&) = delete;
};

inline void opencl_wait() {
    // Public operations return only after queued GPU work has finished and reported errors.
    opencl_check(clFinish(opencl_context().queue), "Execute OpenCL operation");
}

inline void opencl_read(const OpenCLBuffer& buffer, float* output, std::size_t count) {
    // CL_TRUE makes the read blocking, so the host array is complete when this returns.
    opencl_check(clEnqueueReadBuffer(opencl_context().queue, buffer.handle, CL_TRUE, 0,
                                     count * sizeof(float), output, 0, nullptr, nullptr),
                 "Copy result to CPU");
}
} // namespace prs::detail