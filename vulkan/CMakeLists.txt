cmake_minimum_required(VERSION 3.18)
project(vulkan_matmul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Install the Vulkan SDK and configure its environment before running CMake.
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS
    "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install/source the Vulkan SDK, or set GLSLC_EXECUTABLE.")
endif()

set(SHADER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/matmul.comp.spv")
add_custom_command(OUTPUT "${SHADER_BINARY}"
    COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.0 -O
        "${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp" -o "${SHADER_BINARY}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp"
    COMMENT "Compiling Vulkan compute shader to SPIR-V"
    VERBATIM)
add_custom_target(matmul_shader DEPENDS "${SHADER_BINARY}")

add_executable(vulkan_matmul main.cpp)
add_dependencies(vulkan_matmul matmul_shader)
target_link_libraries(vulkan_matmul PRIVATE Vulkan::Vulkan)
target_compile_definitions(vulkan_matmul PRIVATE
    MATMUL_SHADER_PATH="${SHADER_BINARY}")

enable_testing()
# These tests execute on a real Vulkan device, including MoltenVK on macOS.
add_test(NAME vulkan_rectangular COMMAND vulkan_matmul 17 23 19)
add_test(NAME vulkan_single_element COMMAND vulkan_matmul 1 1 1)
