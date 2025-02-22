#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include "linear_algebra/lamp_matrix.h"

#define LAMP_EXP(x) expf(x)

// We create a 2x2x1 network
// 2 inputs, 2 hidden nodes and one output
// | i1 | - | h1 |\
// |    | - |    | - | o |
// | i2 | - | h2 |/
#define NUM_INPUT_NODES 2
#define NUM_HIDDEN_NODES 2
#define NUM_OUTPUT_NODES 1

#define LEARNING_RATE 1e-1f
#define FINITE_DIFF_STEP 1e-1f

// Use sigmoid because it is easy and convenient for this test
LAMP_FLOAT_TYPE sigmoidf(LAMP_FLOAT_TYPE x) {
    return 1.0f / (1 + LAMP_EXP(-x));
}

// Basic building block of the nn that defines its "structure".
// A layer contains artificial neurons - most of the time depicted as circles.
typedef struct {
    LampMatrix *neurons;
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
// It contains of one input layer, one or multiple hidden layers and one output layer.
// I do not know how we are going to handle all this after we used it, so we use static arrays for
// the hidden layers and connections.
typedef struct {
    LampNNLayer input;
    LampNNLayer hidden[1]; // Looks silly, but is a reminder, this is intended to store more than one layer
    LampNNLayer out;
    LampNNConnection connections[2];
} LampNN;

void forward_pass(LampNN nn) {
    // In the forward pass we perform
    // [w.rows, w.cols] * [in.rows, in.cols] + [b] = [a]
    // weights * layer_begin + bias = activation
    //
    // In this notation activation is equal to the neuron value
    // TODO: Maybe we should work on the semantics here, to be closer to the standard notation?
    //       Rename neurons to activations?
    size_t layer_count = sizeof(nn.connections) / sizeof(LampNNConnection);
    for (int i = 0; i < layer_count; ++i) {
        lamp_mat_multiply_into(nn.connections[i].layer_end->neurons, nn.connections[i].weights,
                               nn.connections[i].layer_begin->neurons);
        assert(nn.connections[i].layer_end->neurons->num_rows == nn.connections[i].bias->num_rows);
        assert(nn.connections[i].layer_end->neurons->num_cols == nn.connections[i].bias->num_cols);
        // TODO: Implement matrix addition?
        for (int j = 0; j < nn.connections[i].layer_end->neurons->num_rows; ++j) {
            for (int k = 0; k < nn.connections[i].layer_end->neurons->num_cols; ++k) {
                LAMP_MAT_ELEMENT_AT(nn.connections[i].layer_end->neurons, j, k) += LAMP_MAT_ELEMENT_AT(
                        nn.connections[i].bias, j, k);
                LAMP_MAT_ELEMENT_AT(nn.connections[i].layer_end->neurons, j, k) = sigmoidf(
                        LAMP_MAT_ELEMENT_AT(nn.connections[i].layer_end->neurons, j, k));
            }
        }
    }
}

LAMP_FLOAT_TYPE loss(LampNN nn, LampMatrix *in, LampMatrix *out) {
    // Loss calculation using mean squared error
    // Loss describes the difference of the calculated value of the nn and the target value out
    assert(in->num_rows == out->num_rows);
    assert(out->num_cols == nn.out.neurons->num_cols);

    LAMP_FLOAT_TYPE loss = 0;
    for (int i = 0; i < in->num_rows; ++i) {
        LAMP_MAT_ELEMENT_AT(nn.input.neurons, 0, 0) = LAMP_MAT_ELEMENT_AT(in, i, 0);
        LAMP_MAT_ELEMENT_AT(nn.input.neurons, 1, 0) = LAMP_MAT_ELEMENT_AT(in, i, 1);

        forward_pass(nn);
        for (int j = 0; j < out->num_cols; ++j) {
            LAMP_FLOAT_TYPE diff = LAMP_MAT_ELEMENT_AT(nn.out.neurons, 0, j) - LAMP_MAT_ELEMENT_AT(out, i, j);
            loss += diff * diff;
        }
    }
    return loss / (LAMP_FLOAT_TYPE) in->num_rows;
}

void finite_diff_gradients_apply(LampNN nn, LampMatrix *in, LampMatrix *out) {
    // Finite difference to approximate derivative
    // We calculate the gradient towards the target value and manipulate weights and bias accordingly

    // TODO: Maybe store the number of connections somewhere so we do not need to calculate it all the time?
    size_t layer_count = sizeof(nn.connections) / sizeof(LampNNConnection);
    LAMP_FLOAT_TYPE init_loss = loss(nn, in, out);
    LAMP_FLOAT_TYPE init_value;

    for (int i = 0; i < layer_count; ++i) {
        for (int j = 0; j < nn.connections[i].weights->num_rows; ++j) {
            for (int k = 0; k < nn.connections[i].weights->num_cols; ++k) {
                init_value = LAMP_MAT_ELEMENT_AT(nn.connections[i].weights, j, k);
                LAMP_MAT_ELEMENT_AT(nn.connections[i].weights, j, k) += FINITE_DIFF_STEP;
                LAMP_FLOAT_TYPE grad_w = (loss(nn, in, out) - init_loss) / FINITE_DIFF_STEP;
                LAMP_MAT_ELEMENT_AT(nn.connections[i].weights, j, k) = init_value;
                LAMP_MAT_ELEMENT_AT(nn.connections[i].weights, j, k) -=
                        LEARNING_RATE * grad_w;
            }
        }

        for (int j = 0; j < nn.connections[i].bias->num_rows; ++j) {
            for (int k = 0; k < nn.connections[i].bias->num_cols; ++k) {
                init_value = LAMP_MAT_ELEMENT_AT(nn.connections[i].bias, j, k);
                LAMP_MAT_ELEMENT_AT(nn.connections[i].bias, j, k) += FINITE_DIFF_STEP;
                LAMP_FLOAT_TYPE grad_b = (loss(nn, in, out) - init_loss) / FINITE_DIFF_STEP;
                LAMP_MAT_ELEMENT_AT(nn.connections[i].bias, j, k) = init_value;
                LAMP_MAT_ELEMENT_AT(nn.connections[i].bias, j, k) -= LEARNING_RATE * grad_b;
            }
        }
    }
}

int main() {
    // Try learning behavior of logic gates - because everybody does this in the beginning ;)

    // TODO: Find good solution to initialize srand()
    srand(time(NULL)); // NOLINT: We know about srand() initialization

    // TODO: Develop more ergonomic way to initialize these matrices
    LampMatrix *input = lamp_mat_alloc(4, NUM_INPUT_NODES);
    LAMP_MAT_ELEMENT_AT(input, 0, 0) = 0;
    LAMP_MAT_ELEMENT_AT(input, 0, 1) = 0;
    LAMP_MAT_ELEMENT_AT(input, 1, 0) = 0;
    LAMP_MAT_ELEMENT_AT(input, 1, 1) = 1;
    LAMP_MAT_ELEMENT_AT(input, 2, 0) = 1;
    LAMP_MAT_ELEMENT_AT(input, 2, 1) = 0;
    LAMP_MAT_ELEMENT_AT(input, 3, 0) = 1;
    LAMP_MAT_ELEMENT_AT(input, 3, 1) = 1;

    // AND-Gate
    LampMatrix *target = lamp_mat_alloc(input->num_rows, 1);
    LAMP_MAT_ELEMENT_AT(target, 0, 0) = 0;
    LAMP_MAT_ELEMENT_AT(target, 1, 0) = 0;
    LAMP_MAT_ELEMENT_AT(target, 2, 0) = 0;
    LAMP_MAT_ELEMENT_AT(target, 3, 0) = 1;

    LampNN nn;
    nn.input.neurons = lamp_mat_alloc(NUM_INPUT_NODES, 1);
    nn.hidden[0].neurons = lamp_mat_alloc(NUM_HIDDEN_NODES, 1);

    nn.connections[0].layer_begin = &nn.input;
    nn.connections[0].layer_end = &nn.hidden[0];

    nn.connections[0].weights = lamp_mat_alloc(nn.connections[0].layer_end->neurons->num_rows,
                                               nn.connections[0].layer_begin->neurons->num_rows);
    nn.connections[0].bias = lamp_mat_alloc(nn.connections[0].layer_end->neurons->num_rows, 1);
    lamp_mat_rand(nn.connections[0].weights);
    lamp_mat_rand(nn.connections[0].bias);

    nn.out.neurons = lamp_mat_alloc(NUM_OUTPUT_NODES, 1);

    nn.connections[1].layer_begin = &nn.hidden[0];
    nn.connections[1].layer_end = &nn.out;

    nn.connections[1].weights = lamp_mat_alloc(nn.connections[1].layer_end->neurons->num_rows,
                                               nn.connections[1].layer_begin->neurons->num_rows);
    nn.connections[1].bias = lamp_mat_alloc(nn.connections[1].layer_end->neurons->num_rows, 1);
    lamp_mat_rand(nn.connections[1].weights);
    lamp_mat_rand(nn.connections[1].bias);

    for (int i = 0; i < 10 * 1000; ++i) {
        finite_diff_gradients_apply(nn, input, target);
        LAMP_FLOAT_TYPE current_loss = loss(nn, input, target);
        if ((i % 100) == 0) {
            printf("Loss[%d]: %f\n", i, current_loss);

//            for (int it = 0; it < input->num_rows; ++it) {
//                LAMP_MAT_ELEMENT_AT(nn.input.neurons, 0, 0) = LAMP_MAT_ELEMENT_AT(input, it, 0);
//                LAMP_MAT_ELEMENT_AT(nn.input.neurons, 1, 0) = LAMP_MAT_ELEMENT_AT(input, it, 1);
//                forward_pass(nn);
//                printf("[%f, %f] -> [%f] (%f)\n",
//                       LAMP_MAT_ELEMENT_AT(input, it, 0),
//                       LAMP_MAT_ELEMENT_AT(input, it, 1),
//                       LAMP_MAT_ELEMENT_AT(nn.out.neurons, 0, 0),
//                       LAMP_MAT_ELEMENT_AT(target, it, 0));
//            }
        }
    }

    for (int i = 0; i < input->num_rows; ++i) {
        LAMP_MAT_ELEMENT_AT(nn.input.neurons, 0, 0) = LAMP_MAT_ELEMENT_AT(input, i, 0);
        LAMP_MAT_ELEMENT_AT(nn.input.neurons, 1, 0) = LAMP_MAT_ELEMENT_AT(input, i, 1);
        forward_pass(nn);
        printf("[%f, %f] -> [%f] (%f)\n",
               LAMP_MAT_ELEMENT_AT(input, i, 0),
               LAMP_MAT_ELEMENT_AT(input, i, 1),
               LAMP_MAT_ELEMENT_AT(nn.out.neurons, 0, 0),
               LAMP_MAT_ELEMENT_AT(target, i, 0));
    }


    lamp_mat_free(input);
    lamp_mat_free(nn.input.neurons);
    lamp_mat_free(nn.hidden[0].neurons);
    lamp_mat_free(nn.out.neurons);
    lamp_mat_free(nn.connections[0].weights);
    lamp_mat_free(nn.connections[0].bias);
    lamp_mat_free(nn.connections[1].weights);
    lamp_mat_free(nn.connections[1].bias);

    return 0;
}
