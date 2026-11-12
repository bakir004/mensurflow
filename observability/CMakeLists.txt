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
