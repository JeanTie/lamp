//
// Created by Jan Thieme on 22.02.2025.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#ifndef LAMP_LAMP_NN_H
#define LAMP_LAMP_NN_H

#include "../linear_algebra/lamp_matrix.h"

// Basic building block of the nn that defines its "structure".
// A layer contains artificial neurons - most of the time depicted as circles.
// These neurons are "activated". For our purpose activation describes a value
// between 0 (not activated) and 1 (fully activated).
typedef struct {
    LampMatrix *activations;
} LampNNLayer;

// A connection in this context describes the - well - connection between two layers.
// Those are mostly depicted as simple straight lines from one node of a layer to all other nodes of another layer.
// For the ease of understanding we think of the layers as a beginning and end point of the connection.
typedef struct {
    LampNNLayer *layer_begin;
    LampNNLayer *layer_end;
    LampMatrix *weights;
    LampMatrix *bias;
} LampNNConnection;

// The neural network combining layers and connections in one convenient structure.
// For easy reference we also store the number of individual layers, as well as the
// amount of connections.
typedef struct {
    LampNNLayer *layers;
    size_t layer_count;
    LampNNConnection *connections;
    size_t connection_count;
} LampNN;

// Allocate neural network with specified architecture.
// The architecture is specified by an array of values, that describe the number of neurons
// of their corresponding layer.
// E.g [2, 2, 1] => 2 input neuron, 2 hidden neuron, 1 output neuron => 3 layers
// TODO: Maybe have a more specific way of defining the architecture?
//       Since we always have an input and output layer it may be more intuitive to specify them
//       explicitly and providing the hidden layer description separately?
LampNN *lamp_nn_alloc(const size_t architecture[], size_t layer_count);

void lamp_nn_free(LampNN *nn);

void lamp_nn_forward(LampNN *nn);

LAMP_FLOAT_TYPE lamp_nn_loss(LampNN *nn, const LampMatrix *input, const LampMatrix *target);

void lamp_nn_apply_finite_diff_gradients(LampNN *nn, const LampMatrix *input, const LampMatrix *target,
                                         LAMP_FLOAT_TYPE finite_diff_step, LAMP_FLOAT_TYPE learning_rate);

void lamp_nn_print(const LampNN *nn);

#endif //LAMP_LAMP_NN_H
