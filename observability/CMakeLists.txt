# Observability: the trace switch for the library and the runner behind the dashboard.
# Nothing in the library depends on this directory.

option(MENSURFLOW_ENABLE_TRACING "Compile MENSURFLOW_TRACE spans into the library" OFF)
if(MENSURFLOW_ENABLE_TRACING)
    # PUBLIC: programs that link mensurflow see the same macro and the trace hook.
    target_compile_definitions(mensurflow PUBLIC MENSURFLOW_ENABLE_TRACING)
    message(STATUS "Tracing: enabled")
endif()

add_executable(observe observe.cpp)
target_link_libraries(observe PRIVATE mensurflow)
target_include_directories(observe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# observe_scalar: the same runner linked against a CPU-only copy of the library built from the same
# sources with the compiler's auto-vectorization turned off, so the dashboard can compare
# scalar, SIMD and threaded SIMD versions of each kernel. Hand-written intrinsics (the AVX2 GEMM
# micro-kernel on x86) and libc's memcpy stay vectorized. MSVC has no switch for this, so the
# target is skipped there.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MENSURFLOW_NO_VECTORIZE -fno-vectorize -fno-slp-vectorize)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(MENSURFLOW_NO_VECTORIZE -fno-tree-vectorize)
endif()

if(MENSURFLOW_NO_VECTORIZE)
    get_target_property(library_sources mensurflow SOURCES)
    set(scalar_sources "")
    foreach(source IN LISTS library_sources)
        if(NOT source MATCHES "backends/(cuda|metal)/")
            list(APPEND scalar_sources "${PROJECT_SOURCE_DIR}/${source}")
        endif()
    endforeach()
    add_library(mensurflow_scalar STATIC ${scalar_sources})
    target_include_directories(mensurflow_scalar
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(mensurflow_scalar PUBLIC cxx_std_17)
    target_compile_options(mensurflow_scalar PRIVATE ${MENSURFLOW_NO_VECTORIZE})
    if(MENSURFLOW_NATIVE)
        target_compile_options(mensurflow_scalar PRIVATE -march=native)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mensurflow_scalar PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(MENSURFLOW_ENABLE_TRACING)
        target_compile_definitions(mensurflow_scalar PUBLIC MENSURFLOW_ENABLE_TRACING)
    endif()

    add_executable(observe_scalar observe.cpp)
    target_link_libraries(observe_scalar PRIVATE mensurflow_scalar)
    target_include_directories(observe_scalar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(observe_scalar PRIVATE OBSERVE_SCALAR_BUILD=1)
    target_compile_options(observe_scalar PRIVATE ${MENSURFLOW_NO_VECTORIZE})
endif()
