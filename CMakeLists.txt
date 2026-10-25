# Build description for the prs library, its correctness tests, and benchmarks.
# Configure with cmake -S . -B build, then compile with cmake --build build.
cmake_minimum_required(VERSION 3.18)
project(prs_kernels LANGUAGES CXX)

# Optional backends: CPU is always built; Metal defaults to enabled on Apple hosts.
option(PRS_ENABLE_CUDA "Build the NVIDIA CUDA backend when available" ON)
option(PRS_ENABLE_METAL "Build the Apple Metal backend" ${APPLE})

# Start with the shared API implementation and the portable CPU backend.
add_library(prs
    src/ops.cpp src/backend.cpp
    src/backends/cpu/backend.cpp src/backends/cpu/axpy.cpp src/backends/cpu/matmul.cpp)
# PUBLIC requirements also reach executables that link against the prs target.
target_include_directories(prs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(prs PUBLIC cxx_std_17)

# OpenMP enables the large-vector CPU AXPY path; it is not required to build.
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(prs PRIVATE OpenMP::OpenMP_CXX)
endif()

# Metal sources use Objective-C++ and link to Apple's system frameworks.
if(PRS_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal requires macOS")
    endif()
    enable_language(OBJCXX)
    target_sources(prs PRIVATE
        src/backends/metal/backend.mm src/backends/metal/axpy.mm src/backends/metal/matmul.mm)
    # Internal #ifdef checks use this flag; consumers do not need Metal details.
    target_compile_definitions(prs PRIVATE PRS_HAS_METAL=1)
    # Apply automatic Objective-C object lifetime management only to .mm sources.
    target_compile_options(prs PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    set_target_properties(prs PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES)
    find_library(PRS_METAL_FRAMEWORK Metal REQUIRED)
    find_library(PRS_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(prs PRIVATE ${PRS_METAL_FRAMEWORK} ${PRS_FOUNDATION_FRAMEWORK})
endif()

if(PRS_ENABLE_CUDA)
    # Find standard toolkit installations even when nvcc is not on PATH.
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
        find_program(PRS_NVCC nvcc HINTS /opt/cuda/bin /usr/local/cuda/bin)
        if(PRS_NVCC)
            set(CMAKE_CUDA_COMPILER "${PRS_NVCC}" CACHE FILEPATH "CUDA compiler")
        endif()
    endif()

    # Arch's CUDA package depends on a supported versioned GCC. The system GCC
    # may be newer than nvcc supports. Respect explicit user/toolchain choices.
    if(EXISTS /etc/arch-release AND NOT CMAKE_CROSSCOMPILING
       AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
        find_program(PRS_PACMAN pacman)
        if(PRS_PACMAN)
            execute_process(COMMAND "${PRS_PACMAN}" -Qi cuda
                OUTPUT_VARIABLE PRS_CUDA_PACKAGE ERROR_QUIET)
            if(PRS_CUDA_PACKAGE MATCHES "gcc([0-9]+)")
                find_program(PRS_CUDA_HOST_CXX "g++-${CMAKE_MATCH_1}")
                if(PRS_CUDA_HOST_CXX)
                    set(CMAKE_CUDA_HOST_COMPILER "${PRS_CUDA_HOST_CXX}"
                        CACHE FILEPATH "CUDA host compiler")
                endif()
            endif()
        endif()
    endif()

    # Compile for the local NVIDIA GPUs, unless the user supplied architectures.
    # With no local device (e.g. CI), leave the compiler's default unchanged.
    if(NOT CMAKE_CROSSCOMPILING AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES
       AND NOT DEFINED ENV{CUDAARCHS})
        find_program(PRS_NVIDIA_SMI nvidia-smi)
        if(PRS_NVIDIA_SMI)
            execute_process(COMMAND "${PRS_NVIDIA_SMI}"
                --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE PRS_GPU_CAPABILITIES RESULT_VARIABLE PRS_SMI_RESULT
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 10)
            if(PRS_SMI_RESULT EQUAL 0 AND PRS_GPU_CAPABILITIES MATCHES "^[0-9.\n\r ]+$")
                string(REGEX MATCHALL "[0-9]+\\.[0-9]+" PRS_GPU_ARCHS "${PRS_GPU_CAPABILITIES}")
                # CMake architecture names omit the decimal point: 8.6 becomes 86.
                string(REPLACE "." "" PRS_GPU_ARCHS "${PRS_GPU_ARCHS}")
                list(REMOVE_DUPLICATES PRS_GPU_ARCHS)
                if(PRS_GPU_ARCHS)
                    set(CMAKE_CUDA_ARCHITECTURES "${PRS_GPU_ARCHS}"
                        CACHE STRING "CUDA architectures detected from local GPUs")
                    message(STATUS "Detected NVIDIA GPU architectures: ${PRS_GPU_ARCHS}")
                endif()
            endif()
        endif()
    endif()

    # Probe first so missing CUDA support still permits a CPU-only build.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_sources(prs PRIVATE
            src/backends/cuda/backend.cu src/backends/cuda/axpy.cu src/backends/cuda/matmul.cu)
        # Compile CUDA dispatch branches only when their implementations exist.
        target_compile_definitions(prs PRIVATE PRS_HAS_CUDA=1)
        set_target_properties(prs PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "CUDA compiler not found; NVIDIA backend disabled")
    endif()
endif()

# Each executable links the shared library; add_test registers it with CTest.
enable_testing()
add_executable(axpy_test tests/axpy_test.cpp)
target_link_libraries(axpy_test PRIVATE prs)
add_test(NAME axpy COMMAND axpy_test)

# Benchmarks measure performance separately from the correctness test suite.
add_executable(axpy_benchmark benchmarks/axpy_benchmark.cpp)
target_link_libraries(axpy_benchmark PRIVATE prs)

add_executable(matmul_test tests/matmul_test.cpp)
target_link_libraries(matmul_test PRIVATE prs)
add_test(NAME matmul COMMAND matmul_test)

add_executable(matmul_benchmark benchmarks/matmul_benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE prs)

# Convenience target used by build.sh/build.bat. Dependencies build both programs
# before execution; TARGET_FILE resolves their paths for the chosen generator.
add_custom_target(run_benchmarks
    COMMAND $<TARGET_FILE:axpy_benchmark>
    COMMAND $<TARGET_FILE:matmul_benchmark>
    DEPENDS axpy_benchmark matmul_benchmark
    USES_TERMINAL
    COMMENT "Running benchmarks on the available backend")
