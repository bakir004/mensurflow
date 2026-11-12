// The CPU backend: host memory, synchronous "streams", and the adapters that connect the
// dispatcher to the kernels in src/backends/cpu/ops. Always built and always available.
//
// Kernels run on multiple threads when the library is built with OpenMP (see threads.hpp).
#include "backends/backend.hpp"

#include "backends/cpu/ops/blas/blas.hpp"
#include "backends/cpu/ops/elementwise/cast.hpp"
#include "backends/cpu/ops/elementwise/elementwise.hpp"
#include "backends/cpu/ops/gemm/gemm_batched.hpp"
#include "backends/cpu/ops/gemm/gemm_packed.hpp"
#include "backends/cpu/ops/indexing/indexing.hpp"
#include "backends/cpu/ops/movement/movement.hpp"
#include "backends/cpu/ops/random/random.hpp"
#include "backends/cpu/ops/reduction/reduction.hpp"
#include "backends/cpu/threads.hpp"
#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "core/tensor_access.hpp"

#include <cstring>
#include <new>
#include <vector>

namespace mensurflow::detail {

namespace {

// matmul, max and min have a separate sequential and parallel kernel: run the parallel one when the
// build has OpenMP (matmul prefers the packed AVX2 kernel when the CPU has it). Every other kernel is a single function that parallelizes itself.
constexpr bool use_parallel = cpu::has_openmp();

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

template <typename T> T* data(const Tensor& t) { return TensorAccess::data<T>(t); }

// ---- BLAS ----------------------------------------------------------------------------------

void dot_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    *data<float>(args.outputs[0]) =
        cpu::dot_f32(data<float>(x), data<float>(args.inputs[1]), x.numel());
}

void scal_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::scal_f32(args.alpha, data<float>(args.inputs[0]), data<float>(out), out.numel());
}

void asum_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    *data<float>(args.outputs[0]) = cpu::asum_f32(data<float>(x), x.numel());
}

void nrm2_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    *data<float>(args.outputs[0]) = cpu::nrm2_f32(data<float>(x), x.numel());
}

void copy_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::copy_f32(data<float>(args.inputs[0]), data<float>(out), out.numel());
}

void matvec_kernel(const OpArgs& args, Stream&) {
    const Tensor& a = args.inputs[0];
    cpu::matvec_f32(data<float>(a), data<float>(args.inputs[1]), data<float>(args.outputs[0]),
                    a.size(0), a.size(1));
}

void gemv_kernel(const OpArgs& args, Stream&) {
    const Tensor& a = args.inputs[0];
    const float* initial = args.inputs.size() > 2 ? data<float>(args.inputs[2]) : nullptr;
    cpu::gemv_f32(data<float>(a), data<float>(args.inputs[1]), data<float>(args.outputs[0]),
                  a.size(0), a.size(1), args.alpha, args.beta, initial, args.transpose);
}

// Ops hand matmul [m,k] and [k,n] inputs with any strides (a transposed or sliced view is not copied)
// and a contiguous [m,n] output. The packed kernel reads them in place; the plain kernels need
// contiguous memory, so on a CPU without AVX2 the operands are made contiguous here.
void matmul_kernel(const OpArgs& args, Stream&) {
    const Tensor& a = args.inputs[0];
    const Tensor& b = args.inputs[1];
    float* c = data<float>(args.outputs[0]);
    const std::int64_t m = a.size(0), k = a.size(1), n = b.size(1);
    if (cpu::packed_gemm_supported()) {
        // The AVX2 + FMA packed kernel; it runs on one thread when the build has no OpenMP and for
        // products too small to split.
        const auto kernel =
            use_parallel ? cpu::matmul_f32_packed_parallel_strided : cpu::matmul_f32_packed_strided;
        const Strides& sa = a.strides();
        const Strides& sb = b.strides();
        kernel(data<float>(a), sa[0], sa[1], data<float>(b), sb[0], sb[1], c, n, m, n, k);
        return;
    }
    const Tensor x = a.is_contiguous() ? a : a.contiguous();
    const Tensor y = b.is_contiguous() ? b : b.contiguous();
    if (use_parallel) {
        cpu::matmul_f32_parallel(data<float>(x), data<float>(y), c, m, n, k);
    } else {
        cpu::matmul_f32_sequential(data<float>(x), data<float>(y), c, m, n, k);
    }
}

void ger_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    const float* initial = args.inputs.size() > 2 ? data<float>(args.inputs[2]) : nullptr;
    cpu::ger_f32(args.alpha, data<float>(args.inputs[0]), data<float>(args.inputs[1]), initial,
                 data<float>(out), out.size(0), out.size(1));
}

// Ops hand gemm_strided_batched two views with the full batch shape, [batch..., m, k] and
// [batch..., k, n] (broadcast batch dimensions have stride 0, matrices any strides), a contiguous
// [batch..., m, n] output and, when beta != 0, a contiguous c of the output's shape. Each matrix's
// starting offset is found by walking the batch index through both views' batch strides.
template <void (*Kernel)(const cpu::BatchedGemm&)>
void gemm_batched_kernel(const OpArgs& args, Stream&) {
    const Tensor& a = args.inputs[0];
    const Tensor& b = args.inputs[1];
    const Tensor& out = args.outputs[0];
    const int batch_dims = out.dim() - 2;
    std::int64_t batch = 1;
    for (int d = 0; d < batch_dims; ++d) {
        batch *= out.size(d);
    }
    std::vector<std::int64_t> a_offsets(static_cast<std::size_t>(batch));
    std::vector<std::int64_t> b_offsets(static_cast<std::size_t>(batch));
    std::vector<std::int64_t> index(static_cast<std::size_t>(batch_dims), 0);
    std::int64_t a_offset = 0, b_offset = 0;
    for (std::int64_t i = 0; i < batch; ++i) {
        a_offsets[static_cast<std::size_t>(i)] = a_offset;
        b_offsets[static_cast<std::size_t>(i)] = b_offset;
        // Advance the batch index like an odometer, last batch dimension fastest.
        for (int d = batch_dims - 1; d >= 0; --d) {
            const auto du = static_cast<std::size_t>(d);
            if (++index[du] < out.size(d)) {
                a_offset += a.strides()[du];
                b_offset += b.strides()[du];
                break;
            }
            a_offset -= (index[du] - 1) * a.strides()[du];
            b_offset -= (index[du] - 1) * b.strides()[du];
            index[du] = 0;
        }
    }
    cpu::BatchedGemm g;
    g.a = data<float>(a);
    g.a_offsets = a_offsets.data();
    g.rsa = a.strides()[static_cast<std::size_t>(batch_dims)];
    g.csa = a.strides()[static_cast<std::size_t>(batch_dims) + 1];
    g.b = data<float>(b);
    g.b_offsets = b_offsets.data();
    g.rsb = b.strides()[static_cast<std::size_t>(batch_dims)];
    g.csb = b.strides()[static_cast<std::size_t>(batch_dims) + 1];
    g.c = data<float>(out);
    g.c0 = args.inputs.size() > 2 ? data<float>(args.inputs[2]) : nullptr;
    g.batch = batch;
    g.m = out.size(-2);
    g.n = out.size(-1);
    g.k = a.size(-1);
    g.alpha = args.alpha;
    g.beta = args.beta;
    Kernel(g);
}

// The default: the parallel version (which picks over-batch or within-matrix from the shapes) when
// the build has OpenMP, the sequential one otherwise.
constexpr void (*gemm_batched_default)(const cpu::BatchedGemm&) =
    use_parallel ? cpu::gemm_batched_f32_parallel : cpu::gemm_batched_f32_sequential;

// ---- Data movement -------------------------------------------------------------------------

// Ops hand concat contiguous inputs, all of the output's shape except along args.axis.
void concat_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    const int axis = args.axis;
    std::int64_t outer = 1;
    for (int d = 0; d < axis; ++d) {
        outer *= out.size(d);
    }
    std::int64_t inner = 1;
    for (int d = axis + 1; d < out.dim(); ++d) {
        inner *= out.size(d);
    }
    std::vector<const float*> inputs;
    std::vector<std::int64_t> widths;
    for (const Tensor& t : args.inputs) {
        inputs.push_back(data<float>(t));
        widths.push_back(t.size(axis) * inner);
    }
    cpu::concat_f32(inputs.data(), widths.data(), static_cast<std::int64_t>(inputs.size()),
                    data<float>(out), outer);
}

// Ops hand pad a contiguous input and args.padding as (before, after) per dimension, in order.
void pad_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const int ndim = x.dim();
    std::vector<std::int64_t> before(static_cast<std::size_t>(ndim));
    std::vector<std::int64_t> after(static_cast<std::size_t>(ndim));
    for (std::size_t d = 0; d < static_cast<std::size_t>(ndim); ++d) {
        before[d] = args.padding[2 * d];
        after[d] = args.padding[2 * d + 1];
    }

    cpu::pad_f32(data<float>(x), data<float>(args.outputs[0]), x.shape().data(), before.data(),
                 after.data(), ndim, args.alpha);
}

void gather_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const Tensor& indices = args.inputs[1];
    const Tensor& out = args.outputs[0];
    cpu::gather_f32(data<float>(x), data<std::int64_t>(indices), data<float>(out), x.shape().data(),
                    x.strides().data(), out.shape().data(), x.dim(), args.axis);
}

void index_select_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const Tensor& indices = args.inputs[1];
    const Tensor& out = args.outputs[0];
    cpu::index_select_f32(data<float>(x), data<std::int64_t>(indices), data<float>(out),
                          x.shape().data(), x.strides().data(), x.dim(), args.axis,
                          indices.numel());
}

// ---- Random generation and indexing --------------------------------------------------------

void uniform_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::uniform_f32(data<float>(out), out.numel(), args.alpha, args.beta, args.seed, args.offset);
}

void normal_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::normal_f32(data<float>(out), out.numel(), args.alpha, args.beta, args.seed, args.offset);
}

void bernoulli_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::bernoulli_f32(data<float>(out), out.numel(), args.alpha, args.seed, args.offset);
}

void dropout_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::dropout_f32(data<float>(args.inputs[0]), data<float>(out),
                     data<std::uint8_t>(args.outputs[1]), out.numel(), args.alpha,
                     args.training, args.seed, args.offset);
}

void dropout_backward_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::dropout_backward_f32(data<float>(args.inputs[0]), data<std::uint8_t>(args.inputs[1]),
                              data<float>(out), out.numel(), args.alpha, args.training);
}

void one_hot_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::one_hot_i64(data<std::int64_t>(args.inputs[0]), data<std::int64_t>(out),
                     args.inputs[0].numel(), out.size(-1));
}

void embedding_kernel(const OpArgs& args, Stream&) {
    const Tensor& weight = args.inputs[0];
    cpu::embedding_f32(data<float>(weight), data<std::int64_t>(args.inputs[1]),
                       data<float>(args.outputs[0]), args.inputs[1].numel(), weight.size(1));
}

void embedding_backward_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::embedding_backward_f32(data<float>(args.inputs[0]), data<std::int64_t>(args.inputs[1]),
                                data<float>(out), args.inputs[1].numel(), out.size(0),
                                out.size(1));
}

// ---- Elementwise ---------------------------------------------------------------------------

// Ops hand elementwise kernels contiguous tensors of the output's shape, so n is its size.
template <void (*Kernel)(const float*, const float*, float*, std::int64_t)>
void binary_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    Kernel(data<float>(args.inputs[0]), data<float>(args.inputs[1]), data<float>(out), out.numel());
}

template <void (*Kernel)(const float*, float*, std::int64_t)>
void unary_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    Kernel(data<float>(args.inputs[0]), data<float>(out), out.numel());
}

template <void (*Kernel)(const float*, const float*, float*, std::int64_t)>
void backward_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    Kernel(data<float>(args.inputs[0]), data<float>(args.inputs[1]), data<float>(out), out.numel());
}

void gelu_backward_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::gelu_backward_f32(data<float>(args.inputs[0]), data<float>(args.inputs[1]),
                           data<float>(out), out.numel(), args.approximate);
}

template <void (*Kernel)(const float*, const float*, std::uint8_t*, std::int64_t)>
void comparison_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    Kernel(data<float>(args.inputs[0]), data<float>(args.inputs[1]), data<std::uint8_t>(out),
           out.numel());
}

void where_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    // Value inputs come first so the existing dispatcher selects their F32 kernel key.
    cpu::where_f32(data<std::uint8_t>(args.inputs[2]), data<float>(args.inputs[0]),
                   data<float>(args.inputs[1]), data<float>(out), out.numel());
}

void* cast_data(const Tensor& tensor) {
    switch (tensor.dtype()) {
    case DType::F32:
        return data<float>(tensor);
    case DType::I32:
        return data<std::int32_t>(tensor);
    case DType::F16:
    case DType::BF16:
        return data<std::uint16_t>(tensor);
    default:
        throw NotImplemented("cast: unsupported dtype");
    }
}

void cast_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const Tensor& out = args.outputs[0];
    cpu::cast_buffer(cast_data(x), x.dtype(), cast_data(out), out.dtype(), out.numel());
}

void leaky_relu_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const Tensor& out = args.outputs[0];
    cpu::leaky_relu_f32(data<float>(x), data<float>(out), out.numel(), args.alpha);
}

void gelu_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::gelu_f32(data<float>(args.inputs[0]), data<float>(out), out.numel(), args.approximate);
}

void elu_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::elu_f32(data<float>(args.inputs[0]), data<float>(out), out.numel(), args.alpha);
}

void softplus_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::softplus_f32(data<float>(args.inputs[0]), data<float>(out), out.numel(), args.alpha,
                      args.beta);
}

void axpy_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::axpy_f32(args.alpha, data<float>(args.inputs[0]), data<float>(args.inputs[1]),
                  data<float>(out), out.numel());
}

void clamp_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::clamp_f32(data<float>(args.inputs[0]), data<float>(out), out.numel(), args.alpha,
                   args.beta);
}

// Ops hand bias_add a contiguous [batch, features] matrix and a contiguous [features] bias.
void bias_add_kernel(const OpArgs& args, Stream&) {
    const Tensor& out = args.outputs[0];
    cpu::bias_add_f32(data<float>(args.inputs[0]), data<float>(args.inputs[1]), data<float>(out),
                      out.numel(), out.size(1));
}

// ---- Reductions ----------------------------------------------------------------------------

template <float (*Kernel)(const float*, std::int64_t)>
void reduce_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    *data<float>(args.outputs[0]) = Kernel(data<float>(x), x.numel());
}

void max_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    *data<float>(args.outputs[0]) = use_parallel
                                        ? cpu::max_f32_parallel(data<float>(x), x.numel())
                                        : cpu::max_f32_sequential(data<float>(x), x.numel());
}

void min_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    *data<float>(args.outputs[0]) = use_parallel
                                        ? cpu::min_f32_parallel(data<float>(x), x.numel())
                                        : cpu::min_f32_sequential(data<float>(x), x.numel());
}

template <void (*Kernel)(const float*, float*, const std::int64_t*, std::int64_t, std::int64_t)>
void axis_reduce_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    if (args.outputs[0].numel() == 0) {
        return;
    }
    Kernel(data<float>(x), data<float>(args.outputs[0]), x.shape().data(), x.dim(), args.axis);
}

void argmax_kernel(const OpArgs& args, Stream&) {
    const Tensor& x = args.inputs[0];
    const std::int64_t width = x.size(x.dim() - 1);
    cpu::argmax_f32(data<float>(x), data<std::int64_t>(args.outputs[0]), x.numel() / width, width);
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
        using namespace cpu;
        auto add = [&dispatcher](OpId op, KernelFn kernel) {
            dispatcher.register_kernel({op, DeviceType::CPU, DType::F32}, kernel);
        };

        add(OpId::Dot, dot_kernel);
        add(OpId::Scal, scal_kernel);
        add(OpId::Asum, asum_kernel);
        add(OpId::Nrm2, nrm2_kernel);
        add(OpId::Copy, copy_kernel);
        add(OpId::Matvec, matvec_kernel);
        add(OpId::Gemv, gemv_kernel);
        add(OpId::Matmul, matmul_kernel);

        add(OpId::Add, binary_kernel<add_f32>);
        add(OpId::Sub, binary_kernel<sub_f32>);
        add(OpId::Mul, binary_kernel<mul_f32>);
        add(OpId::Div, binary_kernel<div_f32>);
        add(OpId::Pow, binary_kernel<pow_f32>);
        add(OpId::Minimum, binary_kernel<minimum_f32>);
        add(OpId::Maximum, binary_kernel<maximum_f32>);
        add(OpId::Eq, comparison_kernel<eq_f32>);
        add(OpId::Ne, comparison_kernel<ne_f32>);
        add(OpId::Lt, comparison_kernel<lt_f32>);
        add(OpId::Le, comparison_kernel<le_f32>);
        add(OpId::Gt, comparison_kernel<gt_f32>);
        add(OpId::Ge, comparison_kernel<ge_f32>);
        add(OpId::Where, where_kernel);
        for (DType type : {DType::F32, DType::I32, DType::F16, DType::BF16}) {
            dispatcher.register_kernel({OpId::Cast, DeviceType::CPU, type}, cast_kernel);
        }
        add(OpId::Axpy, axpy_kernel);
        add(OpId::BiasAdd, bias_add_kernel);
        add(OpId::Neg, unary_kernel<neg_f32>);
        add(OpId::Abs, unary_kernel<abs_f32>);
        add(OpId::Square, unary_kernel<square_f32>);
        add(OpId::Relu, unary_kernel<relu_f32>);
        add(OpId::LeakyRelu, leaky_relu_kernel);
        add(OpId::Gelu, gelu_kernel);
        add(OpId::Silu, unary_kernel<silu_f32>);
        add(OpId::Elu, elu_kernel);
        add(OpId::Softplus, softplus_kernel);
        add(OpId::ReluBackward, backward_kernel<relu_backward_f32>);
        add(OpId::SigmoidBackward, backward_kernel<sigmoid_backward_f32>);
        add(OpId::TanhBackward, backward_kernel<tanh_backward_f32>);
        add(OpId::GeluBackward, gelu_backward_kernel);
        add(OpId::SiluBackward, backward_kernel<silu_backward_f32>);
        add(OpId::Sqrt, unary_kernel<sqrt_f32>);
        add(OpId::Exp, unary_kernel<exp_f32>);
        add(OpId::Sigmoid, unary_kernel<sigmoid_f32>);
        add(OpId::Log, unary_kernel<log_f32>);
        add(OpId::Rsqrt, unary_kernel<rsqrt_f32>);
        add(OpId::Tanh, unary_kernel<tanh_f32>);
        add(OpId::Sin, unary_kernel<sin_f32>);
        add(OpId::Cos, unary_kernel<cos_f32>);
        add(OpId::Reciprocal, unary_kernel<reciprocal_f32>);
        add(OpId::Clamp, clamp_kernel);

        add(OpId::Sum, reduce_kernel<sum_f32>);
        add(OpId::Mean, reduce_kernel<mean_f32>);
        add(OpId::Max, max_kernel);
        add(OpId::Min, min_kernel);
        add(OpId::Prod, reduce_kernel<prod_f32>);
        add(OpId::AxisSum, axis_reduce_kernel<sum_axis_f32>);
        add(OpId::AxisMean, axis_reduce_kernel<mean_axis_f32>);
        add(OpId::AxisMax, axis_reduce_kernel<max_axis_f32>);
        add(OpId::AxisMin, axis_reduce_kernel<min_axis_f32>);
        add(OpId::AxisProd, axis_reduce_kernel<prod_axis_f32>);
        add(OpId::Argmax, argmax_kernel);

        add(OpId::Ger, ger_kernel);
        add(OpId::GemmBatched, gemm_batched_kernel<gemm_batched_default>);
        add(OpId::Concat, concat_kernel);
        add(OpId::Pad, pad_kernel);
        add(OpId::Gather, gather_kernel);
        add(OpId::IndexSelect, index_select_kernel);
        add(OpId::Uniform, uniform_kernel);
        add(OpId::Normal, normal_kernel);
        add(OpId::Bernoulli, bernoulli_kernel);
        add(OpId::Dropout, dropout_kernel);
        add(OpId::DropoutBackward, dropout_backward_kernel);
        dispatcher.register_kernel({OpId::OneHot, DeviceType::CPU, DType::I64}, one_hot_kernel);
        add(OpId::Embedding, embedding_kernel);
        add(OpId::EmbeddingBackward, embedding_backward_kernel);
    }

  private:
    CpuStream stream_;
};

} // namespace

void register_cpu_backend(Registry& registry) { registry.add(std::make_unique<CpuBackend>()); }

} // namespace mensurflow::detail
