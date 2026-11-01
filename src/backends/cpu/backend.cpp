// The CPU backend: host memory, synchronous "streams", and the CPU kernels registered with the
// dispatcher. Always built and always available.
#include "backends/backend.hpp"

#include "backends/cpu/blas.hpp"
#include "backends/cpu/elementwise.hpp"
#include "backends/cpu/matmul.hpp"
#include "backends/cpu/parallel.hpp"
#include "backends/cpu/reduction.hpp"
#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "core/tensor_access.hpp"

#include <cstring>
#include <new>

namespace mensurflow::detail {

namespace {

// 64-byte alignment fits a cache line and the widest SIMD loads (AVX-512).
constexpr std::size_t alignment = 64;

class CpuBuffer final : public Buffer {
  public:
    explicit CpuBuffer(std::size_t bytes) : size_(bytes) {
        if (bytes > 0) {
            data_ = ::operator new(bytes, std::align_val_t{alignment});
        }
    }

    ~CpuBuffer() override {
        if (data_ != nullptr) {
            ::operator delete(data_, std::align_val_t{alignment});
        }
    }

    CpuBuffer(const CpuBuffer&) = delete;
    CpuBuffer& operator=(const CpuBuffer&) = delete;

    std::size_t size() const override { return size_; }

    void* native() const override { return data_; }

    void* host_ptr() const override { return data_; }

  private:
    std::size_t size_;
    void* data_ = nullptr; // nullptr when size_ == 0
};

// Work on the CPU runs inline, so there is never anything to wait for.
class CpuStream final : public Stream {
  public:
    void copy_h2d(Buffer& dst, std::size_t offset, const void* src, std::size_t bytes) override {
        std::memcpy(static_cast<std::byte*>(dst.native()) + offset, src, bytes);
    }

    void copy_d2h(void* dst, const Buffer& src, std::size_t offset, std::size_t bytes) override {
        std::memcpy(dst, static_cast<const std::byte*>(src.native()) + offset, bytes);
    }

    void synchronize() override {}
};

// Ops hand matmul contiguous [m,k] and [k,n] inputs and a [m,n] output.
void matmul_kernel(const OpArgs& args, Stream&) {
    const Tensor& a = args.inputs[0];
    const Tensor& b = args.inputs[1];
    cpu::matmul_f32(TensorAccess::data<float>(a), TensorAccess::data<float>(b),
                    TensorAccess::data<float>(args.outputs[0]), a.size(0), b.size(1), a.size(1));
}

void dot_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const Tensor& y = args.inputs[1];
    *TensorAccess::data<float>(args.outputs[0]) =
        cpu::dot_f32(TensorAccess::data<float>(x), TensorAccess::data<float>(y), x.size(0));
}

// Ops hand kernels contiguous tensors of equal shape, so n is simply the output's element count.
template <void (*Kernel)(const float*, const float*, float*, std::int64_t)>
void binary_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    Kernel(TensorAccess::data<float>(args.inputs[0]), TensorAccess::data<float>(args.inputs[1]),
           TensorAccess::data<float>(out), out.numel());
}

template <void (*Kernel)(const float*, float*, std::int64_t)>
void unary_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    if (out.numel() == 0) {
        return; // Empty buffers have no data pointer; dispatch still checks kernel support.
    }
    Kernel(TensorAccess::data<float>(args.inputs[0]), TensorAccess::data<float>(out), out.numel());
}

void axpy_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    if (out.numel() == 0) {
        return;
    }
    cpu::axpy_f32(args.alpha, TensorAccess::data<float>(args.inputs[0]),
                  TensorAccess::data<float>(args.inputs[1]), TensorAccess::data<float>(out),
                  out.numel());
}

void bias_add_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    if (out.numel() == 0) {
        return;
    }

    cpu::bias_add_f32(TensorAccess::data<float>(args.inputs[0]),
                      TensorAccess::data<float>(args.inputs[1]), TensorAccess::data<float>(out),
                      out.numel(), out.size(1));
}

void sum_kernel(const OpArgs& args, Stream&) {
    *TensorAccess::data<float>(args.outputs[0]) =
        cpu::sum_f32(TensorAccess::data<float>(args.inputs[0]), args.inputs[0].numel());
}

void mean_kernel(const OpArgs& args, Stream&) {
    *TensorAccess::data<float>(args.outputs[0]) =
        cpu::mean_f32(TensorAccess::data<float>(args.inputs[0]), args.inputs[0].numel());
}

void max_kernel(const OpArgs& args, Stream&) {
    *TensorAccess::data<float>(args.outputs[0]) =
        cpu::max_f32(TensorAccess::data<float>(args.inputs[0]), args.inputs[0].numel());
}

class CpuBackend final : public Backend {
  public:
    DeviceType type() const override { return DeviceType::CPU; }

    int device_count() const override { return 1; }

    DeviceInfo info(int) const override {
        return {"CPU (" + std::to_string(cpu::max_threads()) + " threads)", 0};
    }

    std::shared_ptr<Buffer> allocate(int, std::size_t bytes) override {
        return std::make_shared<CpuBuffer>(bytes);
    }

    Stream& stream(int) override { return stream_; }

    void register_kernels(Dispatcher& dispatcher) override {
        constexpr DeviceType cpu_type = DeviceType::CPU;
        dispatcher.register_kernel({OpId::Dot, cpu_type, DType::F32}, dot_kernel);
        dispatcher.register_kernel({OpId::Matmul, cpu_type, DType::F32}, matmul_kernel);
        dispatcher.register_kernel({OpId::Add, cpu_type, DType::F32}, binary_kernel<cpu::add_f32>);
        dispatcher.register_kernel({OpId::Sub, cpu_type, DType::F32}, binary_kernel<cpu::sub_f32>);
        dispatcher.register_kernel({OpId::Mul, cpu_type, DType::F32}, binary_kernel<cpu::mul_f32>);
        dispatcher.register_kernel({OpId::Div, cpu_type, DType::F32}, binary_kernel<cpu::div_f32>);
        dispatcher.register_kernel({OpId::Neg, cpu_type, DType::F32}, unary_kernel<cpu::neg_f32>);
        dispatcher.register_kernel({OpId::Abs, cpu_type, DType::F32}, unary_kernel<cpu::abs_f32>);
        dispatcher.register_kernel({OpId::Square, cpu_type, DType::F32},
                                   unary_kernel<cpu::square_f32>);
        dispatcher.register_kernel({OpId::Relu, cpu_type, DType::F32}, unary_kernel<cpu::relu_f32>);
        dispatcher.register_kernel({OpId::Sqrt, cpu_type, DType::F32}, unary_kernel<cpu::sqrt_f32>);
        dispatcher.register_kernel({OpId::Exp, cpu_type, DType::F32}, unary_kernel<cpu::exp_f32>);
        dispatcher.register_kernel({OpId::Sigmoid, cpu_type, DType::F32},
                                   unary_kernel<cpu::sigmoid_f32>);
        dispatcher.register_kernel({OpId::Axpy, cpu_type, DType::F32}, axpy_kernel);
        dispatcher.register_kernel({OpId::BiasAdd, cpu_type, DType::F32}, bias_add_kernel);
        dispatcher.register_kernel({OpId::Sum, cpu_type, DType::F32}, sum_kernel);
        dispatcher.register_kernel({OpId::Mean, cpu_type, DType::F32}, mean_kernel);
        dispatcher.register_kernel({OpId::Max, cpu_type, DType::F32}, max_kernel);
    }

  private:
    CpuStream stream_;
};

} // namespace

void register_cpu_backend(Registry& registry) { registry.add(std::make_unique<CpuBackend>()); }

} // namespace mensurflow::detail
