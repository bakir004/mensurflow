// The CUDA backend: device memory from the CUDA runtime and dot() through cuBLAS. Plain C++ that
// calls the runtime and cuBLAS APIs, so it builds without nvcc.
// Registered only when CMake found the CUDA Toolkit (MENSURFLOW_HAS_CUDA).
#include "backends/backend.hpp"

#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "core/tensor_access.hpp"

#include <climits>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mensurflow::detail {

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what +
                                 " failed: " + cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS ") + what + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class CudaBuffer final : public Buffer {
  public:
    CudaBuffer(int device, std::size_t bytes) : device_(device), size_(bytes) {
        if (bytes > 0) {
            check(cudaSetDevice(device), "cudaSetDevice");
            check(cudaMalloc(&data_, bytes), "cudaMalloc");
        }
    }

    ~CudaBuffer() override {
        if (data_ != nullptr) {
            cudaSetDevice(device_);
            cudaFree(data_); // Errors in a destructor cannot be reported; ignore them.
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    std::size_t size() const override { return size_; }

    void* native() const override { return data_; }

  private:
    int device_;
    std::size_t size_;
    void* data_ = nullptr; // nullptr when size_ == 0
};

// One CUDA stream plus the cuBLAS handle bound to it, for one device.
class CudaStream final : public Stream {
  public:
    explicit CudaStream(int device) : device_(device) {
        activate();
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
        check(cublasCreate(&cublas_), "cublasCreate");
        check(cublasSetStream(cublas_, stream_), "cublasSetStream");
        // Results are written to device memory, so dot() never has to wait for the host.
        check(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
    }

    ~CudaStream() override {
        cudaSetDevice(device_);
        cublasDestroy(cublas_);
        cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Makes this stream's device current for the calling thread. Kernels call it first.
    void activate() const { check(cudaSetDevice(device_), "cudaSetDevice"); }

    cublasHandle_t cublas() const { return cublas_; }

    cudaStream_t raw() const { return stream_; }

    void copy_h2d(Buffer& dst, std::size_t offset, const void* src, std::size_t bytes) override {
        activate();
        check(cudaMemcpyAsync(static_cast<std::byte*>(dst.native()) + offset, src, bytes,
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync (host to device)");
        synchronize(); // src is pageable host memory the caller may reuse as soon as we return
    }

    void copy_d2h(void* dst, const Buffer& src, std::size_t offset, std::size_t bytes) override {
        activate();
        check(cudaMemcpyAsync(dst, static_cast<const std::byte*>(src.native()) + offset, bytes,
                              cudaMemcpyDeviceToHost, stream_),
              "cudaMemcpyAsync (device to host)");
        synchronize(); // also waits for the kernels queued before this copy
    }

    void synchronize() override {
        activate();
        check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

  private:
    int device_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
};

// Ops have already checked shapes and made inputs contiguous, so n and the stride-1 increments
// only need to fit cuBLAS's 32-bit arguments.
void dot_kernel(const OpArgs& args, Stream& stream) {
    auto& s = static_cast<CudaStream&>(stream);
    const Tensor& x = args.inputs[0];
    const Tensor& y = args.inputs[1];
    const std::int64_t n = x.size(0);
    if (n > INT_MAX) {
        throw std::invalid_argument("dot: cuBLAS handles at most 2^31-1 elements per call");
    }
    s.activate();
    check(cublasSdot(s.cublas(), static_cast<int>(n), TensorAccess::data<float>(x), 1,
                     TensorAccess::data<float>(y), 1, TensorAccess::data<float>(args.outputs[0])),
          "cublasSdot");
}

// Ops hand elementwise kernels contiguous tensors of equal shape, so n is the output's size.
int elementwise_length(const char* op, const Tensor& out) {
    if (out.numel() > INT_MAX) {
        throw std::invalid_argument(std::string(op) + ": cuBLAS handles at most 2^31-1 elements");
    }
    return static_cast<int>(out.numel());
}

// Calls that take a host-side scalar need the handle in host-pointer mode; dot() leaves it in
// device-pointer mode, so switch for one call and always switch back.
class HostPointerMode {
  public:
    explicit HostPointerMode(cublasHandle_t handle) : handle_(handle) {
        check(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    }

    ~HostPointerMode() { cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE); }

    HostPointerMode(const HostPointerMode&) = delete;
    HostPointerMode& operator=(const HostPointerMode&) = delete;

  private:
    cublasHandle_t handle_;
};

// cuBLAS has no elementwise add, so out = base + alpha * scaled is a device-to-device copy of
// `base` followed by an axpy of `scaled`. Costs three memory passes instead of one; a dedicated
// kernel would be faster, and is needed anyway for ops cuBLAS cannot express (div, relu, ...).
void copy_then_axpy(Stream& stream, const Tensor& base, const Tensor& scaled, float alpha,
                    const Tensor& out, const char* op) {
    auto& s = static_cast<CudaStream&>(stream);
    const int n = elementwise_length(op, out);
    s.activate();
    check(cudaMemcpyAsync(TensorAccess::data<float>(out), TensorAccess::data<float>(base),
                          static_cast<std::size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice,
                          s.raw()),
          "cudaMemcpyAsync (device to device)");
    HostPointerMode mode(s.cublas());
    check(cublasSaxpy(s.cublas(), n, &alpha, TensorAccess::data<float>(scaled), 1,
                      TensorAccess::data<float>(out), 1),
          "cublasSaxpy");
}

void add_kernel(const OpArgs& args, Stream& stream) {
    copy_then_axpy(stream, args.inputs[1], args.inputs[0], 1.0f, args.outputs[0], "add");
}

void sub_kernel(const OpArgs& args, Stream& stream) {
    copy_then_axpy(stream, args.inputs[0], args.inputs[1], -1.0f, args.outputs[0], "sub");
}

// Elementwise product as a diagonal-matrix multiply: an n x 1 matrix A times diag(y) is A * y.
void mul_kernel(const OpArgs& args, Stream& stream) {
    auto& s = static_cast<CudaStream&>(stream);
    const Tensor& out = args.outputs[0];
    const int n = elementwise_length("mul", out);
    s.activate();
    check(cublasSdgmm(s.cublas(), CUBLAS_SIDE_LEFT, n, 1, TensorAccess::data<float>(args.inputs[0]),
                      n, TensorAccess::data<float>(args.inputs[1]), 1,
                      TensorAccess::data<float>(out), n),
          "cublasSdgmm");
}

class CudaBackend final : public Backend {
  public:
    CudaBackend() {
        int count = 0;
        // No driver or no GPU makes this fail; that just means zero devices.
        count_ = cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
        streams_.resize(static_cast<std::size_t>(count_));
    }

    DeviceType type() const override { return DeviceType::CUDA; }

    int device_count() const override { return count_; }

    DeviceInfo info(int index) const override {
        cudaDeviceProp props;
        check(cudaGetDeviceProperties(&props, index), "cudaGetDeviceProperties");
        return {props.name, props.totalGlobalMem};
    }

    std::shared_ptr<Buffer> allocate(int index, std::size_t bytes) override {
        return std::make_shared<CudaBuffer>(index, bytes);
    }

    Stream& stream(int index) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& slot = streams_[static_cast<std::size_t>(index)];
        if (!slot) {
            slot = std::make_unique<CudaStream>(index);
        }
        return *slot;
    }

    void register_kernels(Dispatcher& dispatcher) override {
        dispatcher.register_kernel({OpId::Dot, DeviceType::CUDA, DType::F32}, dot_kernel);
        dispatcher.register_kernel({OpId::Add, DeviceType::CUDA, DType::F32}, add_kernel);
        dispatcher.register_kernel({OpId::Sub, DeviceType::CUDA, DType::F32}, sub_kernel);
        dispatcher.register_kernel({OpId::Mul, DeviceType::CUDA, DType::F32}, mul_kernel);
        // No Div: cuBLAS cannot divide elementwise. A call reports NotImplemented until a
        // dedicated CUDA kernel exists.
    }

  private:
    int count_ = 0;
    std::mutex mutex_;
    std::vector<std::unique_ptr<CudaStream>> streams_; // created on first use
};

} // namespace

void register_cuda_backend(Registry& registry) { registry.add(std::make_unique<CudaBackend>()); }

} // namespace mensurflow::detail
