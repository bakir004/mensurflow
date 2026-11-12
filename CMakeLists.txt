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
set(MENSURFLOW_METAL AUTO CACHE STRING "Metal + MPSGraph backend: AUTO, ON or OFF (macOS only)")
foreach(backend CUDA METAL)
    set_property(CACHE MENSURFLOW_${backend} PROPERTY STRINGS AUTO ON OFF)
endforeach()

# Core library: public types, backend registry, dispatcher, ops, and the CPU backend.
add_library(mensurflow
    src/core/dtype.cpp
    src/core/device.cpp
    src/core/device_selection.cpp
    src/core/dispatcher.cpp
    src/core/execution.cpp
    src/core/registry.cpp
    src/core/shape.cpp
    src/core/storage.cpp
    src/core/autograd.cpp
    src/core/tensor.cpp
    src/ops/blas.cpp
    src/ops/elementwise.cpp
    src/ops/indexing.cpp
    src/ops/movement.cpp
    src/ops/random.cpp
    src/ops/reduction.cpp
    src/backends/cpu/backend.cpp
    src/backends/cpu/affinity.cpp
    src/backends/cpu/thread_pool.cpp
    src/backends/cpu/ops/blas/blas.cpp
    src/backends/cpu/ops/gemm/gemm_batched.cpp
    src/backends/cpu/ops/gemm/gemm_packed.cpp
    src/backends/cpu/ops/elementwise/parallel_cutoffs.cpp
    src/backends/cpu/ops/elementwise/elementwise.cpp
    src/backends/cpu/ops/elementwise/cast.cpp
    src/backends/cpu/ops/indexing/indexing.cpp
    src/backends/cpu/ops/movement/movement.cpp
    src/backends/cpu/ops/random/random.cpp
    src/backends/cpu/ops/reduction/reduction.cpp)
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

# OpenMP is optional: without it the CPU backend runs the sequential kernels.
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
mensurflow_add_test(activation_test tests/ops/activation_test.cpp)
mensurflow_add_test(selection_test tests/ops/selection_test.cpp)
mensurflow_add_test(parallel_cutoff_test tests/ops/parallel_cutoff_test.cpp)
mensurflow_add_test(cast_test tests/ops/cast_test.cpp)
mensurflow_add_test(thread_count_test tests/ops/thread_count_test.cpp)
mensurflow_add_test(matmul_test tests/ops/matmul_test.cpp)
mensurflow_add_test(affinity_test tests/core/affinity_test.cpp)
mensurflow_add_test(thread_pool_test tests/core/thread_pool_test.cpp)
mensurflow_add_test(gemm_packed_test tests/ops/gemm_packed_test.cpp)
mensurflow_add_test(gemm_pack_test tests/ops/gemm_pack_test.cpp)
mensurflow_add_test(matvec_test tests/ops/matvec_test.cpp)
mensurflow_add_test(argmax_test tests/ops/argmax_test.cpp)
mensurflow_add_test(reduction_test tests/ops/reduction_test.cpp)
mensurflow_add_test(p0_extended_test tests/ops/p0_extended_test.cpp)
mensurflow_add_test(gemm_batched_test tests/ops/gemm_batched_test.cpp)
mensurflow_add_test(ger_test tests/ops/ger_test.cpp)
mensurflow_add_test(movement_test tests/ops/movement_test.cpp)
mensurflow_add_test(activation_movement_test tests/ops/activation_movement_test.cpp)
mensurflow_add_test(random_test tests/ops/random_test.cpp)
mensurflow_add_test(indexing_test tests/ops/indexing_test.cpp)

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
mensurflow_add_benchmark(bandwidth_benchmark benchmarks/bandwidth_benchmark.cpp)
mensurflow_add_benchmark(activation_movement_benchmark benchmarks/activation_movement_benchmark.cpp)
mensurflow_add_benchmark(dot_benchmark benchmarks/blas/dot_benchmark.cpp)
mensurflow_add_benchmark(matmul_benchmark benchmarks/blas/matmul_benchmark.cpp)
# Optional reference: matmul against OpenBLAS. AUTO builds it when OpenBLAS is found, ON requires
# it, OFF skips it. Point CMake at an install with -DOPENBLAS_ROOT=<dir containing include/ lib/ bin/>
# (the prebuilt Windows zip from the OpenBLAS releases works) or with vcpkg's toolchain file.
set(MENSURFLOW_OPENBLAS AUTO CACHE STRING "matmul_openblas_benchmark: AUTO, ON or OFF")
set_property(CACHE MENSURFLOW_OPENBLAS PROPERTY STRINGS AUTO ON OFF)
set(OPENBLAS_ROOT "" CACHE PATH "OpenBLAS install prefix (include/, lib/, bin/)")
if(NOT MENSURFLOW_OPENBLAS STREQUAL "OFF")
    find_package(OpenBLAS CONFIG QUIET)
    set(MENSURFLOW_OPENBLAS_TARGET "")
    set(MENSURFLOW_OPENBLAS_DLL "")
    if(TARGET OpenBLAS::OpenBLAS)
        set(MENSURFLOW_OPENBLAS_TARGET OpenBLAS::OpenBLAS)
    else()
        find_path(MENSURFLOW_OPENBLAS_INCLUDE cblas.h
                  HINTS ${OPENBLAS_ROOT} ENV OPENBLAS_ROOT
                  PATH_SUFFIXES include include/openblas)
        find_library(MENSURFLOW_OPENBLAS_LIB NAMES openblas libopenblas
                     HINTS ${OPENBLAS_ROOT} ENV OPENBLAS_ROOT PATH_SUFFIXES lib)
        find_file(MENSURFLOW_OPENBLAS_DLL_FILE NAMES libopenblas.dll openblas.dll
                  HINTS ${OPENBLAS_ROOT} ENV OPENBLAS_ROOT PATH_SUFFIXES bin)
        if(MENSURFLOW_OPENBLAS_INCLUDE AND MENSURFLOW_OPENBLAS_LIB)
            add_library(mensurflow_openblas INTERFACE)
            target_include_directories(mensurflow_openblas INTERFACE ${MENSURFLOW_OPENBLAS_INCLUDE})
            target_link_libraries(mensurflow_openblas INTERFACE ${MENSURFLOW_OPENBLAS_LIB})
            set(MENSURFLOW_OPENBLAS_TARGET mensurflow_openblas)
            set(MENSURFLOW_OPENBLAS_DLL "${MENSURFLOW_OPENBLAS_DLL_FILE}")
        endif()
    endif()
    if(MENSURFLOW_OPENBLAS_TARGET)
        mensurflow_add_benchmark(matmul_openblas_benchmark benchmarks/blas/matmul_openblas_benchmark.cpp)
        target_link_libraries(matmul_openblas_benchmark PRIVATE ${MENSURFLOW_OPENBLAS_TARGET})
        if(MENSURFLOW_OPENBLAS_DLL)
            add_custom_command(TARGET matmul_openblas_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MENSURFLOW_OPENBLAS_DLL}"
                        "$<TARGET_FILE_DIR:matmul_openblas_benchmark>")
        endif()
        # The DLL is copied next to matmul_openblas_benchmark, which is in the same directory.
        mensurflow_add_benchmark(gemm_shapes_openblas_benchmark
                                 benchmarks/blas/gemm_shapes_openblas_benchmark.cpp)
        target_link_libraries(gemm_shapes_openblas_benchmark PRIVATE ${MENSURFLOW_OPENBLAS_TARGET})
        message(STATUS "OpenBLAS benchmark: enabled")
    elseif(MENSURFLOW_OPENBLAS STREQUAL "ON")
        message(FATAL_ERROR "MENSURFLOW_OPENBLAS=ON but OpenBLAS was not found (set OPENBLAS_ROOT)")
    else()
        message(STATUS "OpenBLAS benchmark: disabled (OpenBLAS not found, set OPENBLAS_ROOT)")
    endif()
endif()

mensurflow_add_benchmark(gemm_tune_benchmark benchmarks/blas/gemm_tune_benchmark.cpp)
mensurflow_add_benchmark(gemm_pack_benchmark benchmarks/blas/gemm_pack_benchmark.cpp)
mensurflow_add_benchmark(gemm_thread_benchmark benchmarks/blas/gemm_thread_benchmark.cpp)
mensurflow_add_benchmark(thread_pool_benchmark benchmarks/blas/thread_pool_benchmark.cpp)
mensurflow_add_benchmark(gemm_stride_benchmark benchmarks/blas/gemm_stride_benchmark.cpp)
mensurflow_add_benchmark(gemm_kernel_benchmark benchmarks/blas/gemm_kernel_benchmark.cpp)
mensurflow_add_benchmark(fma_peak_benchmark benchmarks/blas/fma_peak_benchmark.cpp)
mensurflow_add_benchmark(matvec_benchmark benchmarks/blas/matvec_benchmark.cpp)
mensurflow_add_benchmark(argmax_benchmark benchmarks/blas/argmax_benchmark.cpp)
mensurflow_add_benchmark(reduction_benchmark benchmarks/blas/reduction_benchmark.cpp)
mensurflow_add_benchmark(cpu_parallel_benchmark benchmarks/cpu_parallel_benchmark.cpp)
mensurflow_add_benchmark(selection_benchmark benchmarks/selection_benchmark.cpp)
mensurflow_add_benchmark(cast_benchmark benchmarks/cast_benchmark.cpp)
mensurflow_add_benchmark(gemm_batched_benchmark benchmarks/blas/gemm_batched_benchmark.cpp)
mensurflow_add_benchmark(ger_benchmark benchmarks/blas/ger_benchmark.cpp)
mensurflow_add_benchmark(movement_benchmark benchmarks/movement_benchmark.cpp)
mensurflow_add_benchmark(random_indexing_benchmark benchmarks/random_indexing_benchmark.cpp)

# LibTorch from a CPU PyTorch wheel is an optional native reference, not a core dependency.
# Configure a separate build directory with Torch's CMake prefix when enabling this target.
option(MENSURFLOW_TORCH_BENCHMARK "Build native CPU operation comparisons with LibTorch" OFF)
if(MENSURFLOW_TORCH_BENCHMARK)
    find_package(Torch CONFIG REQUIRED)
    if(TORCH_CXX_FLAGS)
        # Keep the ABI consistent across libraries that exchange std::string/vector values.
        separate_arguments(MENSURFLOW_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(mensurflow PUBLIC ${MENSURFLOW_TORCH_CXX_FLAGS})
    endif()
    mensurflow_add_benchmark(activation_torch_benchmark benchmarks/activation_torch_benchmark.cpp)
    target_link_libraries(activation_torch_benchmark PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(selection_benchmark PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(selection_benchmark PRIVATE MENSURFLOW_TORCH_REFERENCE=1)
    target_link_libraries(cast_benchmark PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(cast_benchmark PRIVATE MENSURFLOW_TORCH_REFERENCE=1)
    target_link_libraries(random_indexing_benchmark PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(random_indexing_benchmark PRIVATE MENSURFLOW_TORCH_REFERENCE=1)
    # The imported torch target supplies the C++ standard its installed headers require.
endif()

add_executable(mlp_xor examples/mlp_xor.cpp)
target_link_libraries(mlp_xor PRIVATE mensurflow)

add_executable(client client/main.cpp)
target_link_libraries(client PRIVATE mensurflow)

add_executable(client_perf client/perf.cpp)
target_link_libraries(client_perf PRIVATE mensurflow)

# Tracing and the observability dashboard (observability/README.md).
add_subdirectory(observability)
