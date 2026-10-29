# Build the mensurflow library and its tests. Backends are added one by one as the plan progresses.
cmake_minimum_required(VERSION 3.20)
project(mensurflow LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Reuse object files across clean rebuilds when ccache is installed.
find_program(MENSURFLOW_CCACHE ccache)
if(MENSURFLOW_CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${MENSURFLOW_CCACHE}")
endif()

# Core library: public types, dispatcher, and (later) the CPU backend.
add_library(mensurflow
    src/core/dtype.cpp
    src/core/device.cpp
    src/core/shape.cpp
    src/core/storage.cpp
    src/core/tensor.cpp)
# PUBLIC requirements also apply to programs that link against the library.
target_include_directories(mensurflow
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(mensurflow PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(mensurflow PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Each test is a small executable registered with CTest.
enable_testing()
function(mensurflow_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mensurflow)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

mensurflow_add_test(dtype_test tests/core/dtype_test.cpp)
mensurflow_add_test(device_test tests/core/device_test.cpp)
mensurflow_add_test(shape_test tests/core/shape_test.cpp)
mensurflow_add_test(tensor_test tests/core/tensor_test.cpp)
