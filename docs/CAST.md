# CPU dtype casts

`cast(x, dtype, execution)` converts among F32, I32, F16 and BF16. All 16 pairs are supported on
the CPU, including same-dtype copies. The result is a fresh contiguous tensor with the input's
shape and device. Scalar, empty, transposed, sliced and expanded inputs follow existing Tensor
packing rules. Casting provides forward execution only and does not record an autograd graph.

```cpp
Tensor x = Tensor::from_vector<float>({1.0f, 1.00390625f, -2.75f}, {3});
Tensor compact = cast(x, DType::BF16);
Tensor restored = cast(compact, DType::F32); // {1.0f, 1.0f, -2.75f}
Tensor integers = cast(x, DType::I32);      // {1, 1, -2}
```

F16/BF16 storage already exists in Tensor and uses two bytes per element. To inspect the stored
bits, call `copy_to_host` into `std::vector<std::uint16_t>`. To inspect numerical values, cast back
to F32 and call `to_vector<float>()`. There is no new half C++ type, Tensor constructor or dtype
conversion hidden in `Tensor::to(Device)`.

## Architecture

The public declaration lives with elementwise operations. `src/ops/elementwise.cpp` validates the
input and supported types, packs views, allocates the requested output dtype, then dispatches
`OpId::Cast`. The existing registry key uses the source dtype; four CPU registrations use the
same adapter. The adapter reads the destination dtype from the output Tensor and passes correctly
offset typed pointers to `cpu::cast_buffer` in `ops/elementwise/cast.cpp`.

No dispatcher policy or `OpArgs` field is added. The CPU implementation selects source/destination
types once, then executes a shared contiguous loop. It uses the existing OpenMP macro and the
independent `ElementwiseOp::cast` cutoff, initially 32,768 elements. Conversion uses portable C++
integer/bit operations without ISA-specific intrinsics.

## Numerical behavior

| Conversion | Behavior |
|---|---|
| Same dtype | Copy bits exactly, including signed zero and NaN payloads |
| To floating dtype | Round to nearest, ties to even |
| To F16/BF16 | Retain subnormals; underflow can round to signed zero; overflow rounds to infinity |
| Between floating dtypes | Preserve NaN classification and infinity sign; converted NaN payloads are not guaranteed |
| Floating to I32 | Truncate toward zero; NaN, infinity and out-of-range values become `INT32_MIN` |
| I32 to floating dtype | Round directly from the integer, avoiding double rounding through F32 |

The floating-to-I32 range is checked before the C++ integer conversion, so invalid values do not
invoke undefined behavior. I32 to F16/BF16 does not use an intermediate rounded F32 value. F16 and
BF16 widening to F32 is exact for finite values. The floating-point format is IEEE binary32.

Undefined inputs or invalid dtype enum values throw `invalid_argument`. Valid but unsupported
types (I64, U8, Bool), as source or destination, throw `NotImplemented`, including empty inputs.
There are no GPU cast kernels. Explicit GPU execution follows the existing missing-backend/kernel
errors; CPU/Auto execution uses the dispatcher's current transfer behavior and returns output on
the input device.

## Verification and benchmark

`cast_test` uses an independent long-double mathematical reference. Coverage includes all
65,536 encodings of both 16-bit formats, every adjacent finite F16/BF16 midpoint with F32 values
just below/at/above it, signs, subnormals, overflow, NaNs, infinities, integer boundaries and
integer double-rounding traps. It checks all pairs, source registrations, fresh outputs, views,
nonzero offsets, empty/scalar inputs, unsupported types, thread counts and cutoff boundaries.

`cast_benchmark` uses the existing nine-sample harness, preallocated buffers and a 500 ms pause
before each timed implementation. Buffers contain objects of the actual storage type, with
input/output/reference starts at offsets 0/1024/2048 modulo 4 KB. Allocation and initialization stay
outside the timer. It checks all outputs against the mathematical reference before timing. Normal
builds compare thread counts; the existing `MENSURFLOW_TORCH_BENCHMARK=ON` option
adds native ATen `copy_` references and prints Torch build/parallel configuration. Torch remains
an optional benchmark dependency.

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./build-cpu/cast_benchmark \
  f32_to_f16 f32_to_bf16 bf16_to_f32 --sizes=1000,32769,4000003 --threads=1,2,4
```

Names use `<source>_to_<destination>`; omit names to run all 16 pairs. `--cutoff=n` changes only
the MensurFlow cast cutoff in that benchmark process. Native `--reverse` times ATen first.
The I32 input repeats integers from -2046 to 2046; floating-source inputs use those values
divided by four, so fractional conversions are exercised. Each input is encoded independently
into its source dtype. Unit tests separately cover special and adversarial values. Rates count
source bytes plus output bytes per element, not physical DRAM bandwidth. Direct MensurFlow kernel calls are compared with
ATen operators including dispatch; allocation, Tensor wrapping, checksums and Python are excluded.

Build/test results and the limits of the current cutoff policy are recorded in the daily worklog.
