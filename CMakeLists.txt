cmake_minimum_required(VERSION 3.24)
project(lamp C)

set(CMAKE_C_STANDARD 17)

add_executable(lamp src/main.c src/linear_algebra/lamp_matrix.c src/linear_algebra/lamp_matrix.h)
add_executable(lamp_tests tests/main.c src/linear_algebra/lamp_matrix.c src/linear_algebra/lamp_matrix.h)