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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lamp_nn.h"

LampNN *lamp_nn_alloc(const size_t architecture[], size_t layer_count) {
    assert(architecture != NULL);
    assert(layer_count >= 2); // Require at least 1 input and 1 output layer

    LampNN *nn = malloc(sizeof(LampNN));
    assert(nn != NULL);

    nn->layer_count = layer_count;
    nn->connection_count = layer_count - 1; // 2 layers are connected by 1 connection

    nn->layers = malloc(sizeof(LampNNLayer) * nn->layer_count);
    nn->connections = malloc(sizeof(LampNNConnection) * nn->connection_count);

    for (size_t i = 0; i < nn->layer_count; i++) {
        nn->layers[i].activations = lamp_mat_alloc(architecture[i], 1);
    }

    for (size_t j = 0; j < nn->connection_count; ++j) {
        LampNNConnection *conn = &nn->connections[j];
        conn->layer_begin = &nn->layers[j];
        conn->layer_end = &nn->layers[j + 1];
        conn->weights = lamp_mat_alloc(conn->layer_end->activations->num_rows,
                                       conn->layer_begin->activations->num_rows);
        conn->bias = lamp_mat_alloc(nn->layers[j + 1].activations->num_rows, 1);
        lamp_mat_rand(conn->weights);
        lamp_mat_rand(conn->bias);
    }

    return nn;
}

LampNN *lamp_nn_alloc_with(size_t input_nodes, size_t hidden_layers, const size_t hidden_activations[], size_t output_nodes) {
    assert(input_nodes > 0);
    assert(output_nodes > 0);
    assert(hidden_layers == 0 || hidden_activations != NULL);

    // No hidden layers: just input + output (2 total layers)
    if (hidden_layers == 0) {
        size_t arch[] = {input_nodes, output_nodes};
        return lamp_nn_alloc(arch, 2);
    }

    size_t layer_count = 1 + hidden_layers + 1; // input + hidden + output
    size_t *arch = malloc(sizeof(size_t) * layer_count);
    assert(arch != NULL);

    arch[0] = input_nodes;
    for (size_t i = 0; i < hidden_layers; ++i) {
        arch[i + 1] = hidden_activations[i];
    }
    arch[layer_count - 1] = output_nodes;

    LampNN *nn = lamp_nn_alloc(arch, layer_count);
    free(arch);
    return nn;
}

void lamp_nn_free(LampNN *nn) {
    assert(nn != NULL);

    for (size_t i = 0; i < nn->connection_count; i++) {
        lamp_mat_free(nn->connections[i].weights);
        lamp_mat_free(nn->connections[i].bias);
    }

    for (size_t j = 0; j < nn->layer_count; ++j) {
        lamp_mat_free(nn->layers[j].activations);
    }

    free(nn->connections);
    free(nn->layers);
    free(nn);
}

// TODO: Design a way to specify activation function instead of hard coding it here.
#define LAMP_EXP(x) expf(x)

static // Use sigmoid because it is easy and convenient for this test
LAMP_FLOAT_TYPE sigmoidf(LAMP_FLOAT_TYPE x) {
    return 1.0f / (1 + LAMP_EXP(-x));
}

static // Derivative of sigmoidf
LAMP_FLOAT_TYPE d_sigmoidf(LAMP_FLOAT_TYPE x) {
    LAMP_FLOAT_TYPE s = sigmoidf(x);
    return s * (1.0f - s);
}

void lamp_nn_forward(LampNN *nn) {
    assert(nn != NULL);
    // In the forward pass we perform
    // [w.rows, w.cols] * [in.rows, in.cols] + [b] = [a]
    // weights * layer_begin + bias = activation
    // for each layer

    for (size_t i = 0; i < nn->connection_count; ++i) {
        LampNNConnection *conn = &nn->connections[i];
        lamp_mat_multiply_into(conn->layer_end->activations, conn->weights,
                               conn->layer_begin->activations);
        lamp_mat_add(conn->layer_end->activations, conn->bias);
        // TODO: Maybe introduce something like lamp_mat_sigmoid()?
        for (size_t j = 0; j < conn->layer_end->activations->num_rows; ++j) {
            for (size_t k = 0; k < conn->layer_end->activations->num_cols; ++k) {
                LAMP_MAT_ELEMENT_AT(conn->layer_end->activations, j, k) = sigmoidf(
                    LAMP_MAT_ELEMENT_AT(conn->layer_end->activations, j, k));
            }
        }
    }
}

LAMP_FLOAT_TYPE lamp_nn_loss(LampNN *nn, const LampMatrix *input, const LampMatrix *target) {
    assert(nn != NULL && input != NULL && target != NULL);
    assert(input->num_rows == target->num_rows);
    assert(target->num_cols == nn->layers[nn->layer_count - 1].activations->num_rows);

    // Loss calculation using mean squared error
    // Loss describes the difference of the calculated value of the nn and the target value out
    LAMP_FLOAT_TYPE loss = 0;

    for (size_t i = 0; i < input->num_rows; ++i) {
        // For each sample
        for (size_t j = 0; j < input->num_cols; ++j) {
            LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, j, 0)
                    = LAMP_MAT_ELEMENT_AT(input, i, j);
        }

        lamp_nn_forward(nn);

        LampMatrix *output = nn->layers[nn->layer_count - 1].activations;
        for (size_t j = 0; j < target->num_cols; ++j) {
            LAMP_FLOAT_TYPE diff =
                    LAMP_MAT_ELEMENT_AT(output, j, 0) -
                    LAMP_MAT_ELEMENT_AT(target, i, j);
            loss += diff * diff;
        }
    }
    return loss / (LAMP_FLOAT_TYPE) input->num_rows;
}

void lamp_nn_apply_finite_diff_gradients(LampNN *nn, const LampMatrix *input, const LampMatrix *target,
                                         LAMP_FLOAT_TYPE finite_diff_step, LAMP_FLOAT_TYPE learning_rate) {
    assert(nn != NULL && input != NULL && target != NULL);
    assert(fabsf(finite_diff_step) > 1e-6 && fabsf(learning_rate) > 1e-6);

    LAMP_FLOAT_TYPE init_loss = lamp_nn_loss(nn, input, target);
    LAMP_FLOAT_TYPE original_value;

    for (size_t i = 0; i < nn->connection_count; ++i) {
        LampNNConnection *conn = &nn->connections[i];
        LampMatrix *weights = conn->weights;

        for (size_t j = 0; j < weights->num_rows; ++j) {
            for (size_t k = 0; k < weights->num_cols; ++k) {
                original_value = LAMP_MAT_ELEMENT_AT(weights, j, k);
                LAMP_MAT_ELEMENT_AT(weights, j, k) += finite_diff_step;
                LAMP_FLOAT_TYPE grad_w = (lamp_nn_loss(nn, input, target) - init_loss) / finite_diff_step;
                LAMP_MAT_ELEMENT_AT(weights, j, k) = original_value;
                LAMP_MAT_ELEMENT_AT(weights, j, k) -= learning_rate * grad_w;
            }
        }

        LampMatrix *bias = conn->bias;
        for (size_t j = 0; j < bias->num_rows; ++j) {
            for (size_t k = 0; k < bias->num_cols; ++k) {
                original_value = LAMP_MAT_ELEMENT_AT(bias, j, k);
                LAMP_MAT_ELEMENT_AT(bias, j, k) += finite_diff_step;
                LAMP_FLOAT_TYPE grad_b = (lamp_nn_loss(nn, input, target) - init_loss) / finite_diff_step;
                LAMP_MAT_ELEMENT_AT(bias, j, k) = original_value;
                LAMP_MAT_ELEMENT_AT(bias, j, k) -= learning_rate * grad_b;
            }
        }
    }
}

void lamp_nn_backprop(LampNN *nn, const LampMatrix *input, const LampMatrix *target,
                      LAMP_FLOAT_TYPE learning_rate) {
    assert(nn != NULL && input != NULL && target != NULL);
    assert(input->num_rows == target->num_rows);

    size_t num_samples = input->num_rows;

    // Allocate gradient accumulators (zeroed)
    LampMatrix *grad_w[nn->connection_count];
    LampMatrix *grad_b[nn->connection_count];
    for (size_t c = 0; c < nn->connection_count; ++c) {
        grad_w[c] = lamp_mat_alloc(nn->connections[c].weights->num_rows,
                                   nn->connections[c].weights->num_cols);
        lamp_mat_fill_with(grad_w[c], 0.0f);
        grad_b[c] = lamp_mat_alloc(nn->connections[c].bias->num_rows,
                                   nn->connections[c].bias->num_cols);
        lamp_mat_fill_with(grad_b[c], 0.0f);
    }

    // Process each training sample
    for (size_t s = 0; s < num_samples; ++s) {
        // Set input for this sample (row of input matrix → column vector)
        for (size_t j = 0; j < input->num_cols; ++j) {
            LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, j, 0) =
                    LAMP_MAT_ELEMENT_AT(input, s, j);
        }

        lamp_nn_forward(nn);

        // Recompute pre-activations (Z = W * A_prev + B)
        LampMatrix *pre_activations[nn->layer_count];
        for (size_t i = 0; i < nn->layer_count; ++i) {
            pre_activations[i] = lamp_mat_alloc(
                nn->layers[i].activations->num_rows,
                nn->layers[i].activations->num_cols);
            lamp_mat_fill_with(pre_activations[i], 0.0f);
        }
        for (size_t c = 0; c < nn->connection_count; ++c) {
            LampNNConnection *conn = &nn->connections[c];
            lamp_mat_multiply_into(pre_activations[c + 1], conn->weights,
                                   conn->layer_begin->activations);
            lamp_mat_add(pre_activations[c + 1], conn->bias);
        }

        // Compute deltas
        LampMatrix *delta[nn->layer_count];
        for (size_t i = 0; i < nn->layer_count; ++i) {
            delta[i] = lamp_mat_alloc(
                nn->layers[i].activations->num_rows,
                nn->layers[i].activations->num_cols);
            lamp_mat_fill_with(delta[i], 0.0f);
        }

        // Output layer delta: (activation - target) * sigmoid'(pre_activation)
        size_t out_idx = nn->layer_count - 1;
        LampMatrix *output = nn->layers[out_idx].activations;
        for (size_t i = 0; i < output->num_rows; ++i) {
            LAMP_FLOAT_TYPE act = LAMP_MAT_ELEMENT_AT(output, i, 0);
            LAMP_FLOAT_TYPE pre_act = LAMP_MAT_ELEMENT_AT(pre_activations[out_idx], i, 0);
            LAMP_FLOAT_TYPE tgt = LAMP_MAT_ELEMENT_AT(target, s, i); // sample s, output i
            LAMP_MAT_ELEMENT_AT(delta[out_idx], i, 0) = (act - tgt) * d_sigmoidf(pre_act);
        }

        // Hidden layer deltas (backpropagate)
        for (int c = (int) nn->connection_count - 1; c >= 0; --c) {
            if (c > 0) {
                LampMatrix *w_trans = lamp_mat_alloc_transpose(nn->connections[c].weights);
                LampMatrix *delta_in = lamp_mat_alloc_multiply(w_trans, delta[c + 1]);
                for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(delta_in); ++e) {
                    delta_in->elements[e] *= d_sigmoidf(pre_activations[c]->elements[e]);
                }
                lamp_mat_copy_into(delta[c], delta_in);
                lamp_mat_free(w_trans);
                lamp_mat_free(delta_in);
            }
        }

        // Accumulate gradients for this sample
        for (size_t c = 0; c < nn->connection_count; ++c) {
            LampMatrix *a_in = nn->connections[c].layer_begin->activations;
            LampMatrix *d_out = delta[c + 1];

            LampMatrix *a_in_trans = lamp_mat_alloc_transpose(a_in);
            LampMatrix *sample_grad = lamp_mat_alloc_multiply(d_out, a_in_trans);

            lamp_mat_add(grad_w[c], sample_grad);
            lamp_mat_add(grad_b[c], d_out);

            lamp_mat_free(a_in_trans);
            lamp_mat_free(sample_grad);
        }

        for (size_t i = 0; i < nn->layer_count; ++i) {
            lamp_mat_free(pre_activations[i]);
            lamp_mat_free(delta[i]);
        }
    }

    // Apply averaged gradients
    for (size_t c = 0; c < nn->connection_count; ++c) {
        LampNNConnection *conn = &nn->connections[c];
        for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(conn->weights); ++e) {
            conn->weights->elements[e] -=
                    learning_rate * grad_w[c]->elements[e] / (LAMP_FLOAT_TYPE) num_samples;
        }
        for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(conn->bias); ++e) {
            conn->bias->elements[e] -=
                    learning_rate * grad_b[c]->elements[e] / (LAMP_FLOAT_TYPE) num_samples;
        }
        lamp_mat_free(grad_w[c]);
        lamp_mat_free(grad_b[c]);
    }
}

void lamp_nn_print(const LampNN *nn) {
    assert(nn != NULL && nn->layer_count > 0);

    printf("\tinput:\n");
    lamp_mat_print(nn->layers[0].activations);

    for (size_t i = 0; i < nn->connection_count; ++i) {
        LampNNConnection *con = &nn->connections[i];

        printf("\tw%zu\n", i + 1);
        lamp_mat_print(con->weights);
        printf("\tb%zu\n", i + 1);
        lamp_mat_print(con->bias);
        printf("\ta%zu\n", i + 1);
        lamp_mat_print(con->layer_end->activations);
    }
    printf("\n");
}
