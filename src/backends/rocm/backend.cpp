// The ROCm backend: device memory from the HIP runtime and dot() through rocBLAS. Mirrors the
// CUDA backend call for call. Plain C++ against the HIP and rocBLAS APIs.
// Registered only when CMake found HIP and rocBLAS (MENSURFLOW_HAS_ROCM).
#include "backends/backend.hpp"

#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "core/tensor_access.hpp"

#include <climits>
#include <hip/hip_runtime_api.h>
#include <mutex>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow::detail {

namespace {

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("HIP ") + what +
                                 " failed: " + hipGetErrorString(status));
    }
}

void check(rocblas_status status, const char* what) {
    if (status != rocblas_status_success) {
        throw std::runtime_error(std::string("rocBLAS ") + what +
                                 " failed: " + rocblas_status_to_string(status));
    }
}

class RocmBuffer final : public Buffer {
  public:
    RocmBuffer(int device, std::size_t bytes) : device_(device), size_(bytes) {
        if (bytes > 0) {
            check(hipSetDevice(device), "hipSetDevice");
            check(hipMalloc(&data_, bytes), "hipMalloc");
        }
    }

    ~RocmBuffer() override {
        if (data_ != nullptr) {
            (void)hipSetDevice(device_);
            (void)hipFree(data_); // Errors in a destructor cannot be reported; ignore them.
        }
    }

    RocmBuffer(const RocmBuffer&) = delete;
    RocmBuffer& operator=(const RocmBuffer&) = delete;

    std::size_t size() const override { return size_; }

    void* native() const override { return data_; }

  private:
    int device_;
    std::size_t size_;
    void* data_ = nullptr; // nullptr when size_ == 0
};

// One HIP stream plus the rocBLAS handle bound to it, for one device.
class RocmStream final : public Stream {
  public:
    explicit RocmStream(int device) : device_(device) {
        activate();
        check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreate");
        check(rocblas_create_handle(&blas_), "rocblas_create_handle");
        check(rocblas_set_stream(blas_, stream_), "rocblas_set_stream");
        // Results are written to device memory, so dot() never has to wait for the host.
        check(rocblas_set_pointer_mode(blas_, rocblas_pointer_mode_device),
              "rocblas_set_pointer_mode");
    }

    ~RocmStream() override {
        (void)hipSetDevice(device_);
        (void)rocblas_destroy_handle(blas_);
        (void)hipStreamDestroy(stream_);
    }

    RocmStream(const RocmStream&) = delete;
    RocmStream& operator=(const RocmStream&) = delete;

    // Makes this stream's device current for the calling thread. Kernels call it first.
    void activate() const { check(hipSetDevice(device_), "hipSetDevice"); }

    rocblas_handle blas() const { return blas_; }

    hipStream_t raw() const { return stream_; }

    void copy_h2d(Buffer& dst, std::size_t offset, const void* src, std::size_t bytes) override {
        activate();
        check(hipMemcpyAsync(static_cast<std::byte*>(dst.native()) + offset, src, bytes,
                             hipMemcpyHostToDevice, stream_),
              "hipMemcpyAsync (host to device)");
        synchronize(); // src is pageable host memory the caller may reuse as soon as we return
    }

    void copy_d2h(void* dst, const Buffer& src, std::size_t offset, std::size_t bytes) override {
        activate();
        check(hipMemcpyAsync(dst, static_cast<const std::byte*>(src.native()) + offset, bytes,
                             hipMemcpyDeviceToHost, stream_),
              "hipMemcpyAsync (device to host)");
        synchronize(); // also waits for the kernels queued before this copy
    }

    void synchronize() override {
        activate();
        check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
    }

  private:
    int device_;
    hipStream_t stream_ = nullptr;
    rocblas_handle blas_ = nullptr;
};

// Ops have already checked shapes and made inputs contiguous, so n and the stride-1 increments
// only need to fit rocBLAS's 32-bit arguments.
void dot_kernel(const OpArgs& args, Stream& stream) {
    auto& s = static_cast<RocmStream&>(stream);
    const Tensor& x = args.inputs[0];
    const Tensor& y = args.inputs[1];
    const std::int64_t n = x.size(0);
    if (n > INT_MAX) {
        throw std::invalid_argument("dot: rocBLAS handles at most 2^31-1 elements per call");
    }
    s.activate();
    check(rocblas_sdot(s.blas(), static_cast<rocblas_int>(n), TensorAccess::data<float>(x), 1,
                       TensorAccess::data<float>(y), 1, TensorAccess::data<float>(args.outputs[0])),
          "rocblas_sdot");
}

// Ops hand elementwise kernels contiguous tensors of equal shape, so n is the output's size.
rocblas_int elementwise_length(const char* op, const Tensor& out) {
    if (out.numel() > INT_MAX) {
        throw std::invalid_argument(std::string(op) + ": rocBLAS handles at most 2^31-1 elements");
    }
    return static_cast<rocblas_int>(out.numel());
}

// Calls that take a host-side scalar need the handle in host-pointer mode; dot() leaves it in
// device-pointer mode, so switch for one call and always switch back.
class HostPointerMode {
  public:
    explicit HostPointerMode(rocblas_handle handle) : handle_(handle) {
        check(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host),
              "rocblas_set_pointer_mode");
    }

    ~HostPointerMode() { (void)rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_device); }

    HostPointerMode(const HostPointerMode&) = delete;
    HostPointerMode& operator=(const HostPointerMode&) = delete;

  private:
    rocblas_handle handle_;
};

// rocBLAS has no elementwise add, so out = base + alpha * scaled is a device-to-device copy of
// `base` followed by an axpy of `scaled`. Costs three memory passes instead of one; a dedicated
// kernel would be faster, and is needed anyway for ops rocBLAS cannot express (div, relu, ...).
void copy_then_axpy(Stream& stream, const Tensor& base, const Tensor& scaled, float alpha,
                    const Tensor& out, const char* op) {
    auto& s = static_cast<RocmStream&>(stream);
    const rocblas_int n = elementwise_length(op, out);
    s.activate();
    check(hipMemcpyAsync(TensorAccess::data<float>(out), TensorAccess::data<float>(base),
                         static_cast<std::size_t>(n) * sizeof(float), hipMemcpyDeviceToDevice,
                         s.raw()),
          "hipMemcpyAsync (device to device)");
    HostPointerMode mode(s.blas());
    check(rocblas_saxpy(s.blas(), n, &alpha, TensorAccess::data<float>(scaled), 1,
                        TensorAccess::data<float>(out), 1),
          "rocblas_saxpy");
}

void add_kernel(const OpArgs& args, Stream& stream) {
    copy_then_axpy(stream, args.inputs[1], args.inputs[0], 1.0f, args.outputs[0], "add");
}

void sub_kernel(const OpArgs& args, Stream& stream) {
    copy_then_axpy(stream, args.inputs[0], args.inputs[1], -1.0f, args.outputs[0], "sub");
}

// Elementwise product as a diagonal-matrix multiply: an n x 1 matrix A times diag(y) is A * y.
void mul_kernel(const OpArgs& args, Stream& stream) {
    auto& s = static_cast<RocmStream&>(stream);
    const Tensor& out = args.outputs[0];
    const rocblas_int n = elementwise_length("mul", out);
    s.activate();
    check(rocblas_sdgmm(
              s.blas(), rocblas_side_left, n, 1, TensorAccess::data<float>(args.inputs[0]), n,
              TensorAccess::data<float>(args.inputs[1]), 1, TensorAccess::data<float>(out), n),
          "rocblas_sdgmm");
}

class RocmBackend final : public Backend {
  public:
    RocmBackend() {
        int count = 0;
        // No driver or no GPU makes this fail; that just means zero devices.
        count_ = hipGetDeviceCount(&count) == hipSuccess ? count : 0;
        streams_.resize(static_cast<std::size_t>(count_));
    }

    DeviceType type() const override { return DeviceType::ROCm; }

    int device_count() const override { return count_; }

    DeviceInfo info(int index) const override {
        hipDeviceProp_t props;
        check(hipGetDeviceProperties(&props, index), "hipGetDeviceProperties");
        return {props.name, props.totalGlobalMem};
    }

    std::shared_ptr<Buffer> allocate(int index, std::size_t bytes) override {
        return std::make_shared<RocmBuffer>(index, bytes);
    }

    Stream& stream(int index) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& slot = streams_[static_cast<std::size_t>(index)];
        if (!slot) {
            slot = std::make_unique<RocmStream>(index);
        }
        return *slot;
    }

    void register_kernels(Dispatcher& dispatcher) override {
        dispatcher.register_kernel({OpId::Dot, DeviceType::ROCm, DType::F32}, dot_kernel);
        dispatcher.register_kernel({OpId::Add, DeviceType::ROCm, DType::F32}, add_kernel);
        dispatcher.register_kernel({OpId::Sub, DeviceType::ROCm, DType::F32}, sub_kernel);
        dispatcher.register_kernel({OpId::Mul, DeviceType::ROCm, DType::F32}, mul_kernel);
        // No Div: rocBLAS cannot divide elementwise. A call reports NotImplemented until a
        // dedicated HIP kernel exists.
    }

  private:
    int count_ = 0;
    std::mutex mutex_;
    std::vector<std::unique_ptr<RocmStream>> streams_; // created on first use
};

} // namespace

void register_rocm_backend(Registry& registry) { registry.add(std::make_unique<RocmBackend>()); }

} // namespace mensurflow::detail
