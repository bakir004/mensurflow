// The Metal backend: shared-memory MTLBuffers, with dot() and elementwise add/sub/mul/div
// through MPSGraph, and matmul through our own shaders (gemm.mm). Objective-C++ compiled with ARC. Registered only on macOS
// (MENSURFLOW_HAS_METAL).
#include "backends/backend.hpp"
#include "backends/metal/gemm.hpp"

#include "core/dispatcher.hpp"
#include "core/registry.hpp"
#include "core/tensor_access.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace mensurflow::detail {

namespace {

// Shared storage mode: the CPU and GPU see the same bytes (Apple silicon is unified memory),
// so host copies are plain memcpy and Tensor can read the buffer directly.
class MetalBuffer final : public Buffer {
  public:
    MetalBuffer(id<MTLDevice> device, std::size_t bytes) : size_(bytes) {
        if (bytes > 0) {
            buffer_ = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (buffer_ == nil) {
                throw std::bad_alloc();
            }
        }
    }

    std::size_t size() const override { return size_; }

    // The MTLBuffer itself, not an address: Metal kernels bridge it back to id<MTLBuffer>.
    void* native() const override { return (__bridge void*)buffer_; }

    void* host_ptr() const override { return buffer_ == nil ? nullptr : [buffer_ contents]; }

  private:
    std::size_t size_;
    id<MTLBuffer> buffer_ = nil; // nil when size_ == 0
};

enum class GraphKind { Dot, Add, Sub, Mul, Div };

// A compiled graph for one op and vector length: two float inputs, one float result.
struct CompiledGraph {
    MPSGraph* graph = nil;
    MPSGraphTensor* x = nil;
    MPSGraphTensor* y = nil;
    MPSGraphTensor* result = nil;
};

class MetalStream final : public Stream {
  public:
    explicit MetalStream(id<MTLDevice> device) : device_(device) {
        queue_ = [device newCommandQueue];
        if (queue_ == nil) {
            throw std::runtime_error("Metal: could not create a command queue");
        }
    }

    id<MTLDevice> device() const { return device_; }

    id<MTLCommandQueue> queue() const { return queue_; }

    void copy_h2d(Buffer& dst, std::size_t offset, const void* src, std::size_t bytes) override {
        std::memcpy(static_cast<std::byte*>(dst.host_ptr()) + offset, src, bytes);
    }

    void copy_d2h(void* dst, const Buffer& src, std::size_t offset, std::size_t bytes) override {
        synchronize(); // the graph that produced this data may still be running
        std::memcpy(dst, static_cast<const std::byte*>(src.host_ptr()) + offset, bytes);
    }

    void synchronize() override {
        // Command buffers on one queue run in order, so an empty one finishing means all
        // earlier work has finished.
        id<MTLCommandBuffer> fence = [queue_ commandBuffer];
        [fence commit];
        [fence waitUntilCompleted];
    }

    // Building a graph is slow, so keep one per (op, length). Bounded so varied lengths cannot
    // grow it without limit.
    const CompiledGraph& graph(GraphKind kind, std::int64_t n) {
        const auto key = std::make_pair(static_cast<int>(kind), n);
        const auto found = graphs_.find(key);
        if (found != graphs_.end()) {
            return found->second;
        }
        if (graphs_.size() >= 64) {
            graphs_.clear();
        }
        CompiledGraph g;
        g.graph = [MPSGraph new];
        NSArray<NSNumber*>* shape = @[ @(n) ];
        g.x = [g.graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
        g.y = [g.graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
        switch (kind) {
        case GraphKind::Dot: {
            MPSGraphTensor* product = [g.graph multiplicationWithPrimaryTensor:g.x
                                                               secondaryTensor:g.y
                                                                          name:nil];
            g.result = [g.graph reductionSumWithTensor:product axes:@[ @0 ] name:nil];
            break;
        }
        case GraphKind::Add:
            g.result = [g.graph additionWithPrimaryTensor:g.x secondaryTensor:g.y name:nil];
            break;
        case GraphKind::Sub:
            g.result = [g.graph subtractionWithPrimaryTensor:g.x secondaryTensor:g.y name:nil];
            break;
        case GraphKind::Mul:
            g.result = [g.graph multiplicationWithPrimaryTensor:g.x secondaryTensor:g.y name:nil];
            break;
        case GraphKind::Div:
            g.result = [g.graph divisionWithPrimaryTensor:g.x secondaryTensor:g.y name:nil];
            break;
        }
        return graphs_.emplace(key, g).first->second;
    }

  private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    std::map<std::pair<int, std::int64_t>, CompiledGraph> graphs_;
};

// The tensor's data as an MTLBuffer starting at byte 0, which is what MPSGraph accepts. A view
// that starts partway into its allocation is copied out first (shared memory, so a memcpy).
id<MTLBuffer> buffer_at_origin(MetalStream& stream, const Tensor& t) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)TensorAccess::buffer(t).native();
    if (t.offset() == 0) {
        return buffer;
    }
    const std::size_t bytes = static_cast<std::size_t>(t.numel()) * sizeof(float);
    id<MTLBuffer> copy = [stream.device() newBufferWithLength:bytes
                                                      options:MTLResourceStorageModeShared];
    if (copy == nil) {
        throw std::bad_alloc();
    }
    std::memcpy([copy contents],
                static_cast<const std::byte*>([buffer contents]) +
                    static_cast<std::size_t>(t.offset()) * sizeof(float),
                bytes);
    return copy;
}

// Runs one op on two length-n vectors, writing `result_length` floats (1 for dot, n otherwise)
// straight into the output tensor's buffer.
void run_graph(Stream& base, GraphKind kind, const Tensor& x, const Tensor& y, const Tensor& out,
               std::int64_t n, std::int64_t result_length) {
    auto& stream = static_cast<MetalStream&>(base);
    @autoreleasepool {
        const CompiledGraph& g = stream.graph(kind, n);
        NSArray<NSNumber*>* shape = @[ @(n) ];
        MPSGraphTensorData* xd =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer_at_origin(stream, x)
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* yd =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer_at_origin(stream, y)
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* od = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:(__bridge id<MTLBuffer>)TensorAccess::buffer(out).native()
                        shape:@[ @(result_length) ]
                     dataType:MPSDataTypeFloat32];
        [g.graph runWithMTLCommandQueue:stream.queue()
                                  feeds:@{g.x : xd, g.y : yd}
                       targetOperations:nil
                      resultsDictionary:@{g.result : od}];
    }
}

void dot_kernel(const OpArgs& args, Stream& stream) {
    const Tensor& x = args.inputs[0];
    run_graph(stream, GraphKind::Dot, x, args.inputs[1], args.outputs[0], x.size(0), 1);
}

// Ops hand elementwise kernels contiguous tensors of equal shape, so n is the output's size.
template <GraphKind Kind> void binary_kernel(const OpArgs& args, Stream& stream) {
    const Tensor& out = args.outputs[0];
    run_graph(stream, Kind, args.inputs[0], args.inputs[1], out, out.numel(), out.numel());
}

// The op hands Metal kernels contiguous inputs; a view may still start partway into its buffer,
// which the encoder takes as a byte offset.
void matmul_kernel(const OpArgs& args, Stream& base) {
    auto& stream = static_cast<MetalStream&>(base);
    const Tensor& a = args.inputs[0];
    const Tensor& b = args.inputs[1];
    const Tensor& c = args.outputs[0];
    auto buffer = [](const Tensor& t) {
        return (__bridge id<MTLBuffer>)TensorAccess::buffer(t).native();
    };
    auto bytes = [](const Tensor& t) {
        return static_cast<std::size_t>(t.offset()) * sizeof(float);
    };
    metal::MetalGemmConfig config = metal::metal_gemm_config();
    if (args.tuning.size == 8) { // chosen by the auto-tuner, in matmul_tunable's order
        const auto& v = args.tuning.values;
        config = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
    metal::gemm_f32(stream.device(), stream.queue(), buffer(a), bytes(a), buffer(b), bytes(b),
                    buffer(c), bytes(c), a.size(0), b.size(1), a.size(1), config);
}

// Parameters the chosen kernel does not use are 0, so the results file and the dashboard show
// only what matters (MPS uses none; tiled ignores the simdgroups; simdgroup ignores tm and tn).
TuningConfig to_tuning(const metal::MetalGemmConfig& c) {
    TuningConfig config;
    if (c.kernel == 2) {
        config.values = {2, 0, 0, 0, 0, 0, 0, 0};
    } else if (c.kernel == 0) {
        config.values = {0, c.bm, c.bn, c.bk, c.tm, c.tn, 0, 0};
    } else {
        config.values = {c.kernel, c.bm, c.bn, c.bk, 0, 0, c.sg_m, c.sg_n};
    }
    config.size = 8;
    return config;
}

// The kernel, tile sizes and simdgroup layout of the Metal matmul (gemm.hpp) are tunable. The
// candidates: MPS; simdgroup tiles with 1 to 8 accumulator blocks per simdgroup (16 ran at a third
// of the speed on an M3 Pro) and 64 to 512 threads; three tiled-kernel shapes. Tiles more than twice
// the matrix are left out, since most of their threads would have nothing to do.
TunableKernel matmul_tunable() {
    TunableKernel t;
    t.name = "matmul";
    t.version = 1;
    t.key_names = {"m", "n", "k"};
    t.key = [](const OpArgs& args) {
        return std::vector<std::int64_t>{args.inputs[0].size(0), args.inputs[1].size(1),
                                         args.inputs[0].size(1)};
    };
    t.params = {"kernel", "bm", "bn", "bk", "tm", "tn", "sg_m", "sg_n"};
    t.defaults = [] { return to_tuning(metal::metal_gemm_config()); };
    t.candidates = [](const OpArgs& args) {
        const std::int64_t m = args.inputs[0].size(0), n = args.inputs[1].size(1);
        std::vector<TuningConfig> list;
        auto add = [&](const metal::MetalGemmConfig& c) {
            if (c.kernel != 2 && (c.bm > 2 * m || c.bn > 2 * n) && !(c.bm <= 16 && c.bn <= 32)) {
                return;
            }
            try {
                metal::validate(c);
            } catch (const std::invalid_argument&) {
                return;
            }
            list.push_back(to_tuning(c));
        };
        add({2, 0, 0, 0, 0, 0, 0, 0});
        for (std::int64_t bm : {16, 32, 64}) {
            for (std::int64_t bn : {32, 64, 128}) {
                for (std::int64_t bk : {16, 32}) {
                    for (std::int64_t sg_m : {1, 2, 4}) {
                        for (std::int64_t sg_n : {1, 2, 4}) {
                            const std::int64_t blocks = (bm / sg_m / 8) * (bn / sg_n / 8);
                            const std::int64_t threads = sg_m * sg_n * 32;
                            if (bm % (8 * sg_m) != 0 || bn % (8 * sg_n) != 0 || blocks < 1 ||
                                blocks > 8 || threads < 64 || threads > 512) {
                                continue;
                            }
                            add({1, bm, bn, bk, 0, 0, sg_m, sg_n});
                        }
                    }
                }
            }
        }
        add({0, 64, 64, 16, 4, 4, 0, 0});
        add({0, 128, 64, 8, 8, 4, 0, 0});
        add({0, 32, 32, 16, 2, 2, 0, 0});
        return list;
    };
    return t;
}

class MetalBackend final : public Backend {
  public:
    MetalBackend() {
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ != nil) {
            stream_ = std::make_unique<MetalStream>(device_);
        }
    }

    DeviceType type() const override { return DeviceType::Metal; }

    int device_count() const override { return device_ == nil ? 0 : 1; }

    DeviceInfo info(int) const override {
        return {std::string([[device_ name] UTF8String]),
                static_cast<std::size_t>([device_ recommendedMaxWorkingSetSize])};
    }

    std::shared_ptr<Buffer> allocate(int, std::size_t bytes) override {
        return std::make_shared<MetalBuffer>(device_, bytes);
    }

    Stream& stream(int) override { return *stream_; }

    void register_kernels(Dispatcher& dispatcher) override {
        constexpr DeviceType metal = DeviceType::Metal;
        dispatcher.register_kernel({OpId::Dot, metal, DType::F32}, dot_kernel);
        dispatcher.register_kernel({OpId::Add, metal, DType::F32}, binary_kernel<GraphKind::Add>);
        dispatcher.register_kernel({OpId::Sub, metal, DType::F32}, binary_kernel<GraphKind::Sub>);
        dispatcher.register_kernel({OpId::Mul, metal, DType::F32}, binary_kernel<GraphKind::Mul>);
        dispatcher.register_kernel({OpId::Div, metal, DType::F32}, binary_kernel<GraphKind::Div>);
        dispatcher.register_kernel({OpId::Matmul, metal, DType::F32}, matmul_kernel);
        dispatcher.register_tunable({OpId::Matmul, metal, DType::F32}, matmul_tunable());
    }

  private:
    id<MTLDevice> device_ = nil;
    std::unique_ptr<MetalStream> stream_;
};

} // namespace

void register_metal_backend(Registry& registry) { registry.add(std::make_unique<MetalBackend>()); }

} // namespace mensurflow::detail
