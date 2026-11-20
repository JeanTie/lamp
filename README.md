# Learning Artificial Machine Pedagogy

## Overview

This project is a small, C-based framework for exploring and learning about artificial neural networks and machine learning. It is designed as a personal learning exercise, not intended for production or any serious use. Expect bugs and potentially incorrect results!

## Goals

* Build a basic understanding of neural network concepts through hands-on coding.
* Experiment with different network architectures, activation functions, and training methods.
* Improve C programming skills.
* Do something out of my comfort zone.

## Features

* **LampMatrix** — hand-rolled linear algebra: matrix allocation, multiplication, addition, transpose, and more.
* **LampNN** — feed-forward neural network with backpropagation, configurable hidden layers and activation functions.
* **Arena allocator** — zero mallocs during backpropagation; all per-sample temporaries come from a bump allocator embedded in the network struct.
* **XOR demo** (`lamp`) — trains a network to solve XOR, with comparisons to logic gate implementations.
* **Logic gates** (`lamp_example_logic_gates`) — trains the network to behave like AND, OR, XOR, NAND, and XNOR.
* **Adder circuits** (`lamp_example_adder_circuits`) — trains the network to perform binary addition.
* **Performance benchmark** (`perf_benchmark`) — measures backpropagation throughput on small and medium networks.

## Planned

* More example problems beyond logic gates and adders.
* Visualization tools.

## Getting Started

You need CMake (3.24+) and a C17 compiler.

Configure the build:

```
cmake -G Ninja -S . -B build
```

Build individual targets:

```
cmake --build build --target lamp
cmake --build build --target lamp_tests
cmake --build build --target lamp_example_logic_gates
cmake --build build --target lamp_example_adder_circuits
cmake --build build --target perf_benchmark
```
