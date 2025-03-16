# Learning Artificial Machine Pedagogy

## Overview

This project is a small, C-based framework for exploring and learning about artifical neural networks and machine learning. It is designed as a personal learning exercise, not intended for prdouction or any serious use. Expect bugs and potentially incorret results!

## Goals

* To build a basic understanding of neural network concepts trough hands-on coding.
* To experiment with different network architectures, activation functions and training methods.
* To improve my programming skills.
* To do something out of my comfort zone.

## Current State

This is just the beginning! The framework is in its early stages of development. There are no concrete plans beyond implementing fundamental building blocks.\
Expect everything to change!

### Features
* Basic feed forward neural network
* Examples for training the network to behave like logic gates and adder circuits

## Features (Planned)
* Examples of different problems that the neural network can solve
* Backpropagation
* Visualization

## Getting started

To build the project use [cmake](https://cmake.org/cmake/help/latest/guide/tutorial/index.html), which configures the build system for the project using the `CMakeList.txt` file.

```
cmake -G Ninja -S . -B build
-- The C compiler identification is GNU 11.2.0
...
-- Build files have been written to: .../lamp/build
```

Build the tests:
```
cmake --build build --target lamp_test
-- The C compiler identification is GNU 11.2.0
...
-- Build files have been written to: .../lamp/build
```


Build the logic gates example:
```
cmake --build build --target lamp_example_logic_gates
...
-- Build finished
```
