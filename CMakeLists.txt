# Build the mensurflow library and its tests. Backends are added one by one as the plan progresses.
cmake_minimum_required(VERSION 3.20)
project(mensurflow LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Reuse object files across clean rebuilds when ccache is installed.
find_program(MENSURFLOW_CCACHE ccache)
if(MENSURFLOW_CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${MENSURFLOW_CCACHE}")
endif()

# Benchmarks are only meaningful optimized, so default to Release when none is chosen.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MENSURFLOW_OPENMP "Parallelize CPU kernels with OpenMP when the compiler has it" ON)
# Off by default: a binary tuned with -march=native crashes on older CPUs. Turn it on locally
# to see what AVX2/FMA buys.
option(MENSURFLOW_NATIVE "Compile for this machine's CPU (not portable to other machines)" OFF)

# GPU backends are chosen at build time: AUTO builds one when its SDK is found, ON requires it
# (configure fails without it), OFF skips it. A built backend is still dropped at run time if
# the machine has no matching device, so one binary works on machines with and without a GPU.
set(MENSURFLOW_CUDA AUTO CACHE STRING "CUDA + cuBLAS backend: AUTO, ON or OFF")
set(MENSURFLOW_ROCM AUTO CACHE STRING "ROCm + rocBLAS backend: AUTO, ON or OFF")
set(MENSURFLOW_METAL AUTO CACHE STRING "Metal + MPSGraph backend: AUTO, ON or OFF (macOS only)")
foreach(backend CUDA ROCM METAL)
    set_property(CACHE MENSURFLOW_${backend} PROPERTY STRINGS AUTO ON OFF)
endforeach()

# Core library: public types, backend registry, dispatcher, ops, and the CPU backend.
add_library(mensurflow
    src/core/dtype.cpp
    src/core/device.cpp
    src/core/device_selection.cpp
    src/core/dispatcher.cpp
    src/core/registry.cpp
    src/core/shape.cpp
    src/core/storage.cpp
    src/core/tensor.cpp
    src/ops/blas.cpp
    src/ops/elementwise.cpp
    src/ops/reduction.cpp
    src/ops/softmax.cpp
    src/backends/cpu/backend.cpp
    src/backends/cpu/blas.cpp
    src/backends/cpu/argmax.cpp
    src/backends/cpu/elementwise.cpp
    src/backends/cpu/matmul.cpp
    src/backends/cpu/reduction.cpp
    src/backends/cpu/scalar.cpp
    src/backends/cpu/sequential.cpp
    src/backends/cpu/sequential_reduction.cpp
    src/backends/cpu/sequential_argmax.cpp
    src/backends/cpu/softmax.cpp
    src/backends/cpu/sequential_softmax.cpp)
# scalar.cpp is sequential.cpp compiled again with auto-vectorization off: the baseline that
# separates what SIMD adds from what threads add in the benchmarks (see docs/BENCHMARKING.md).
set_source_files_properties(src/backends/cpu/scalar.cpp PROPERTIES COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU>:-fno-tree-vectorize>;$<$<CXX_COMPILER_ID:Clang,AppleClang>:-fno-vectorize>;$<$<CXX_COMPILER_ID:Clang,AppleClang>:-fno-slp-vectorize>;$<$<CXX_COMPILER_ID:MSVC>:/Qvec->")
# PUBLIC requirements also apply to programs that link against the library.
target_include_directories(mensurflow
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(mensurflow PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(mensurflow PRIVATE -Wall -Wextra -Wpedantic)
    if(MENSURFLOW_NATIVE)
        target_compile_options(mensurflow PRIVATE -march=native)
    endif()
endif()

# OpenMP is optional: without it parallel_for runs serially and everything still builds.
if(MENSURFLOW_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(mensurflow PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not used: CPU kernels will run single-threaded")
endif()

# CUDA: cuBLAS and the runtime are called from plain C++, so nvcc is not needed.
if(NOT MENSURFLOW_CUDA STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_sources(mensurflow PRIVATE src/backends/cuda/backend.cpp)
        target_link_libraries(mensurflow PRIVATE CUDA::cudart CUDA::cublas)
        target_compile_definitions(mensurflow PRIVATE MENSURFLOW_HAS_CUDA=1)
        message(STATUS "CUDA backend: enabled (CUDA ${CUDAToolkit_VERSION})")
    elseif(MENSURFLOW_CUDA STREQUAL "ON")
        message(FATAL_ERROR "MENSURFLOW_CUDA=ON but the CUDA Toolkit was not found")
    else()
        message(STATUS "CUDA backend: disabled (CUDA Toolkit not found)")
    endif()
endif()

# ROCm: HIP runtime and rocBLAS, also from plain C++.
if(NOT MENSURFLOW_ROCM STREQUAL "OFF")
    find_package(hip CONFIG QUIET)
    find_package(rocblas CONFIG QUIET)
    if(hip_FOUND AND rocblas_FOUND)
        target_sources(mensurflow PRIVATE src/backends/rocm/backend.cpp)
        target_link_libraries(mensurflow PRIVATE hip::host roc::rocblas)
        target_compile_definitions(mensurflow PRIVATE MENSURFLOW_HAS_ROCM=1)
        message(STATUS "ROCm backend: enabled")
    elseif(MENSURFLOW_ROCM STREQUAL "ON")
        message(FATAL_ERROR "MENSURFLOW_ROCM=ON but HIP or rocBLAS was not found")
    else()
        message(STATUS "ROCm backend: disabled (HIP or rocBLAS not found)")
    endif()
endif()

# Metal: Objective-C++ calling MPSGraph, macOS only.
if(NOT MENSURFLOW_METAL STREQUAL "OFF")
    if(APPLE)
        enable_language(OBJCXX)
        set(CMAKE_OBJCXX_STANDARD 17)
        find_library(MENSURFLOW_FOUNDATION Foundation)
        find_library(MENSURFLOW_METAL_FW Metal)
        find_library(MENSURFLOW_MPS MetalPerformanceShaders)
        find_library(MENSURFLOW_MPSGRAPH MetalPerformanceShadersGraph)
        target_sources(mensurflow PRIVATE src/backends/metal/backend.mm)
        set_source_files_properties(src/backends/metal/backend.mm PROPERTIES COMPILE_FLAGS -fobjc-arc)
        target_link_libraries(mensurflow PRIVATE ${MENSURFLOW_FOUNDATION} ${MENSURFLOW_METAL_FW}
                              ${MENSURFLOW_MPS} ${MENSURFLOW_MPSGRAPH})
        target_compile_definitions(mensurflow PRIVATE MENSURFLOW_HAS_METAL=1)
        message(STATUS "Metal backend: enabled")
    elseif(MENSURFLOW_METAL STREQUAL "ON")
        message(FATAL_ERROR "MENSURFLOW_METAL=ON but this is not macOS")
    endif()
endif()

# Each test is a small executable registered with CTest.
enable_testing()
function(mensurflow_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mensurflow)
    # Tests may include internal kernel headers (src/backends/...) and tests/reference.
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests
                                               ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

mensurflow_add_test(dtype_test tests/core/dtype_test.cpp)
mensurflow_add_test(device_test tests/core/device_test.cpp)
mensurflow_add_test(shape_test tests/core/shape_test.cpp)
mensurflow_add_test(tensor_test tests/core/tensor_test.cpp)
mensurflow_add_test(view_test tests/core/view_test.cpp)
mensurflow_add_test(device_selection_test tests/core/device_selection_test.cpp)
mensurflow_add_test(dispatcher_test tests/core/dispatcher_test.cpp)
mensurflow_add_test(transfer_test tests/core/transfer_test.cpp)
mensurflow_add_test(elementwise_test tests/ops/elementwise_test.cpp)
mensurflow_add_test(dot_test tests/ops/dot_test.cpp)
mensurflow_add_test(elementwise_ops_test tests/ops/elementwise_ops_test.cpp)
mensurflow_add_test(sequential_test tests/ops/sequential_test.cpp)
mensurflow_add_test(scalar_test tests/ops/scalar_test.cpp)
mensurflow_add_test(matmul_test tests/ops/matmul_test.cpp)
mensurflow_add_test(matvec_test tests/ops/matvec_test.cpp)
mensurflow_add_test(softmax_test tests/ops/softmax_test.cpp)
mensurflow_add_test(argmax_test tests/ops/argmax_test.cpp)
mensurflow_add_test(reduction_test tests/ops/reduction_test.cpp)

# Benchmarks are not tests: run them by hand, e.g. ./elementwise_benchmark.
# The shared harness every benchmark uses (timing, checks, header, table): benchmarks/bench.hpp.
add_library(mensurflow_bench STATIC benchmarks/bench.cpp)
target_link_libraries(mensurflow_bench PUBLIC mensurflow)
target_include_directories(mensurflow_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/tests)

function(mensurflow_add_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mensurflow_bench)
endfunction()

mensurflow_add_benchmark(elementwise_benchmark benchmarks/elementwise_benchmark.cpp)
mensurflow_add_benchmark(dot_benchmark benchmarks/blas/dot_benchmark.cpp)
mensurflow_add_benchmark(matmul_benchmark benchmarks/blas/matmul_benchmark.cpp)
mensurflow_add_benchmark(matvec_benchmark benchmarks/blas/matvec_benchmark.cpp)
mensurflow_add_benchmark(softmax_benchmark benchmarks/blas/softmax_benchmark.cpp)
mensurflow_add_benchmark(argmax_benchmark benchmarks/blas/argmax_benchmark.cpp)
mensurflow_add_benchmark(reduction_benchmark benchmarks/blas/reduction_benchmark.cpp)
mensurflow_add_benchmark(cpu_parallel_benchmark benchmarks/cpu_parallel_benchmark.cpp)
