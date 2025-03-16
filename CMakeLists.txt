cmake_minimum_required(VERSION 3.24)
project(lamp C)

set(CMAKE_C_STANDARD 17)

set(COMMON_SOURCES
        src/linear_algebra/lamp_matrix.h
        src/linear_algebra/lamp_matrix.c
        src/neural_network/lamp_nn.h
        src/neural_network/lamp_nn.c)

add_executable(lamp src/main.c ${COMMON_SOURCES})
add_executable(lamp_tests tests/main.c ${COMMON_SOURCES})
add_executable(lamp_example_logic_gates examples/logic_gates.c ${COMMON_SOURCES})
add_executable(lamp_example_adder_circuits examples/adder_circuits.c ${COMMON_SOURCES})