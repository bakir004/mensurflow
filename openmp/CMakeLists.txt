cmake_minimum_required(VERSION 3.18)
project(openmp_dot_product LANGUAGES CXX)

# Standalone example; does not configure any of the main project's GPU backends.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Apple Clang needs a separately installed runtime. Find common Homebrew installs
# without requiring teammates to hard-code their machine's paths.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(COMMAND "${BREW_EXECUTABLE}" --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    find_path(LIBOMP_INCLUDE_DIR omp.h HINTS
        "${LIBOMP_PREFIX}/include" /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
    find_library(LIBOMP_LIBRARY omp HINTS
        "${LIBOMP_PREFIX}/lib" /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "OpenMP compiler flags")
        set(OpenMP_CXX_LIB_NAMES omp CACHE STRING "OpenMP libraries")
        set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}" CACHE FILEPATH "OpenMP runtime")
        set(OpenMP_CXX_INCLUDE_DIR "${LIBOMP_INCLUDE_DIR}" CACHE PATH "OpenMP headers")
    endif()
endif()
# Classic MSVC /openmp does not enable SIMD directives. Use its SIMD-capable
# experimental mode; GCC/Clang remain the recommended benchmark toolchains.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(OpenMP_CXX_FLAGS "/openmp:experimental" CACHE STRING "OpenMP compiler flags")
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(dot_product main.cpp sequential.cpp)
target_link_libraries(dot_product PRIVATE OpenMP::OpenMP_CXX)

# A normal optimized loop may auto-vectorize. Disable that for the baseline only,
# leaving SIMD enabled for the other implementations.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(sequential.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-tree-vectorize")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(sequential.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize")
elseif(MSVC)
    set_source_files_properties(sequential.cpp PROPERTIES COMPILE_OPTIONS "/Qvec-")
endif()
set_property(TARGET dot_product PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

# Native ISA tuning is opt-in: binaries built this way may not run on other CPUs.
option(DOT_NATIVE "Tune for the build machine (GCC/Clang)" OFF)
if(DOT_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dot_product PRIVATE -march=native)
endif()

# BLAS is optional. Verify that CBLAS headers AND the actual dot symbol work.
option(DOT_ENABLE_BLAS "Compare against an installed BLAS library when available" ON)
if(DOT_ENABLE_BLAS)
    include(CheckCXXSourceCompiles)
    if(APPLE)
        find_library(DOT_ACCELERATE Accelerate)
        if(DOT_ACCELERATE)
            set(CMAKE_REQUIRED_LIBRARIES "${DOT_ACCELERATE}")
            check_cxx_source_compiles("#include <Accelerate/Accelerate.h>
int main() { double x[1]={1}; return cblas_ddot(1,x,1,x,1)==1 ? 0 : 1; }" DOT_HAS_ACCELERATE)
            if(DOT_HAS_ACCELERATE)
                target_compile_definitions(dot_product PRIVATE DOT_USE_ACCELERATE=1)
                target_link_libraries(dot_product PRIVATE "${DOT_ACCELERATE}")
            endif()
        endif()
    else()
        find_package(BLAS QUIET)
        find_path(DOT_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas openblas-pthread openblas-openmp)
        if(BLAS_FOUND AND DOT_CBLAS_INCLUDE_DIR)
            set(CMAKE_REQUIRED_INCLUDES "${DOT_CBLAS_INCLUDE_DIR}")
            set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}")
            check_cxx_source_compiles("#include <cblas.h>
int main() { double x[1]={1}; return cblas_ddot(1,x,1,x,1)==1 ? 0 : 1; }" DOT_HAS_CBLAS)
            if(DOT_HAS_CBLAS)
                target_compile_definitions(dot_product PRIVATE DOT_USE_CBLAS=1)
                target_include_directories(dot_product PRIVATE "${DOT_CBLAS_INCLUDE_DIR}")
                target_link_libraries(dot_product PRIVATE ${BLAS_LIBRARIES})
            endif()
        endif()
    endif()
    if(NOT DOT_HAS_ACCELERATE AND NOT DOT_HAS_CBLAS)
        message(STATUS "CBLAS unavailable: vendor BLAS comparison will be skipped")
    endif()
endif()

enable_testing()
add_test(NAME dot_small COMMAND dot_product 1003 3 2)
add_test(NAME dot_single_element COMMAND dot_product 1 3 1)
