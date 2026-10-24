cmake_minimum_required(VERSION 3.24)
project(lamp VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)

# Generate lamp_version.h from template (CMake @VARIABLE@ substitution)
configure_file(
    src/lamp_version.h.in
    ${CMAKE_BINARY_DIR}/lamp_version.h
    @ONLY
)

set(COMMON_SOURCES
        src/linear_algebra/lamp_matrix.h
        src/linear_algebra/lamp_matrix.c
        src/neural_network/lamp_nn.h
        src/neural_network/lamp_nn.c)

set_source_files_properties(${CMAKE_BINARY_DIR}/lamp_version.h PROPERTIES GENERATED TRUE)
set(COMMON_SOURCES ${COMMON_SOURCES} ${CMAKE_BINARY_DIR}/lamp_version.h)

set(LAMP_INCLUDE_DIRS
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

set(LAMP_TARGETS
    lamp
    lamp_tests
    lamp_example_logic_gates
    lamp_example_adder_circuits
    perf_benchmark
)

set(LAMP_MAIN_SOURCES
    src/main.c
    tests/main.c
    examples/logic_gates.c
    examples/adder_circuits.c
    performance_analysis/perf_benchmark.c
)

list(LENGTH LAMP_TARGETS TARGET_COUNT)
math(EXPR LAST_IDX "${TARGET_COUNT} - 1")
foreach(target_index RANGE 0 ${LAST_IDX})
    list(GET LAMP_TARGETS ${target_index} target_name)
    list(GET LAMP_MAIN_SOURCES ${target_index} main_source)
    add_executable(${target_name} ${main_source} ${COMMON_SOURCES})
    target_include_directories(${target_name} PRIVATE ${LAMP_INCLUDE_DIRS})
endforeach()