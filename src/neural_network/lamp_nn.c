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

#define LAMP_ARENA_ALIGNMENT 8 // sufficient for void*, size_t, float on 64-bit

static size_t nn_arena_capacity(const LampNN *nn);

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
        nn->layers[i].pre_activations = lamp_mat_alloc(architecture[i], 1);
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

        // Pre-allocate gradient accumulators (zero-filled)
        conn->grad_w = lamp_mat_alloc(conn->weights->num_rows, conn->weights->num_cols);
        lamp_mat_fill_with(conn->grad_w, 0.0f);
        conn->grad_b = lamp_mat_alloc(conn->bias->num_rows, conn->bias->num_cols);
        lamp_mat_fill_with(conn->grad_b, 0.0f);

        // Cache weight transpose for backprop
        conn->weight_trans = lamp_mat_alloc_transpose(conn->weights);
    }

    /* Default to sigmoid activation */
    nn->config = &LAMP_ACTIVATION_SIGMOID;

    // Arena for per-sample temporaries
    size_t arena_cap = nn_arena_capacity(nn);
    nn->arena.ptr = malloc(arena_cap);
    assert(nn->arena.ptr != NULL);
    nn->arena.capacity = arena_cap;
    nn->arena.offset = 0;

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

static void *nn_arena_alloc_aligned(LampArena *arena, size_t size, size_t alignment) {
    assert(arena != NULL);
    assert(size > 0 && alignment > 0);

    size_t misaligned = arena->offset % alignment;
    if (misaligned) {
        arena->offset += alignment - misaligned;
    }

    if (arena->offset + size > arena->capacity) {
        assert(0 && "Arena overflow - increase initial capacity");
        return NULL;
    }

    void *p = (char *)arena->ptr + arena->offset;
    arena->offset += size;
    return p;
}

static void *nn_arena_alloc(LampArena *arena, size_t size) {
    return nn_arena_alloc_aligned(arena, size, LAMP_ARENA_ALIGNMENT);
}

static void nn_arena_reset(LampArena *arena) {
    arena->offset = 0;
}

static void nn_arena_free(LampArena *arena) {
    free(arena->ptr);
    arena->ptr = NULL;
    arena->capacity = 0;
    arena->offset = 0;
}

// Compute arena capacity from the network architecture.
// All per-sample buffers are counted, per-batch buffers (grad_w/grad_b) are pre-allocated as fields and not included.
static size_t nn_arena_capacity(const LampNN *nn) {
    size_t cap = 0;

    cap += nn->layer_count * sizeof(LampMatrix *);
    for (size_t i = 0; i < nn->layer_count; ++i) {
        cap += sizeof(LampMatrix);
        cap += nn->layers[i].activations->num_rows * sizeof(LAMP_FLOAT_TYPE);
    }

    for (size_t c = 0; c < nn->connection_count; ++c) {
        LampMatrix *w = nn->connections[c].weights;
        LampMatrix *b = nn->connections[c].bias;

        cap += sizeof(LampMatrix);
        cap += w->num_cols * w->num_rows * sizeof(LAMP_FLOAT_TYPE);

        if (c > 0) {
            cap += sizeof(LampMatrix);
            cap += nn->layers[c].activations->num_rows * sizeof(LAMP_FLOAT_TYPE);
        }

        cap += sizeof(LampMatrix);
        cap += nn->connections[c].layer_begin->activations->num_rows *
               nn->connections[c].layer_begin->activations->num_cols * sizeof(LAMP_FLOAT_TYPE);

        cap += sizeof(LampMatrix);
        cap += b->num_rows * w->num_cols * sizeof(LAMP_FLOAT_TYPE);

        cap += b->num_rows * sizeof(LAMP_FLOAT_TYPE);
    }

    cap = (cap * 5) / 4; // 25% safety margin
    return cap;
}

void lamp_nn_free(LampNN *nn) {
    assert(nn != NULL);

    for (size_t i = 0; i < nn->connection_count; i++) {
        lamp_mat_free(nn->connections[i].weights);
        lamp_mat_free(nn->connections[i].bias);
        lamp_mat_free(nn->connections[i].grad_w);
        lamp_mat_free(nn->connections[i].grad_b);
        lamp_mat_free(nn->connections[i].weight_trans);
    }

    nn_arena_free(&nn->arena);

    for (size_t j = 0; j < nn->layer_count; ++j) {
        lamp_mat_free(nn->layers[j].activations);
        lamp_mat_free(nn->layers[j].pre_activations);
    }

    free(nn->connections);
    free(nn->layers);
    free(nn);
}

#define LAMP_EXP(x) expf(x)

/* Predefined activation configurations */
static LAMP_FLOAT_TYPE sigmoid_activate(LAMP_FLOAT_TYPE x) {
    return 1.0f / (1 + LAMP_EXP(-x));
}

static LAMP_FLOAT_TYPE sigmoid_derivative(LAMP_FLOAT_TYPE pre_act_value) {
    LAMP_FLOAT_TYPE a = sigmoid_activate(pre_act_value);
    return a * (1.0f - a);
}

LampNNActivationConfig LAMP_ACTIVATION_SIGMOID = {
    .activate = sigmoid_activate,
    .derivative = sigmoid_derivative,
    .name = "sigmoid",
    .learning_rate = 1.0f
};

static LAMP_FLOAT_TYPE relu_activate(LAMP_FLOAT_TYPE x) {
    return (x > 0) ? x : 0.0f;
}

static LAMP_FLOAT_TYPE relu_derivative(LAMP_FLOAT_TYPE pre_act_value) {
    return (pre_act_value > 0) ? 1.0f : 0.0f;
}

LampNNActivationConfig LAMP_ACTIVATION_RELU = {
    .activate = relu_activate,
    .derivative = relu_derivative,
    .name = "relu",
    .learning_rate = 0.01f
};

static LAMP_FLOAT_TYPE tanh_activate(LAMP_FLOAT_TYPE x) {
    return tanhf(x);
}

static LAMP_FLOAT_TYPE tanh_derivative(LAMP_FLOAT_TYPE pre_act_value) {
    LAMP_FLOAT_TYPE a = tanhf(pre_act_value);
    return 1.0f - a * a;
}

LampNNActivationConfig LAMP_ACTIVATION_TANH = {
    .activate = tanh_activate,
    .derivative = tanh_derivative,
    .name = "tanh",
    .learning_rate = 1.0f
};

void lamp_nn_set_activation(LampNN *nn, const LampNNActivationConfig *config) {
    assert(nn != NULL && config != NULL);
    nn->config = config;
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
        // Cache pre-activation Z = W*A_prev + B before applying activation
        lamp_mat_copy_into(conn->layer_end->pre_activations, conn->layer_end->activations);

        for (size_t j = 0; j < conn->layer_end->activations->num_rows; ++j) {
            for (size_t k = 0; k < conn->layer_end->activations->num_cols; ++k) {
                LAMP_MAT_ELEMENT_AT(conn->layer_end->activations, j, k) = nn->config->activate(
                    LAMP_MAT_ELEMENT_AT(conn->layer_end->activations, j, k));
            }
        }
    }
}

void lamp_nn_forward_single(LampNN *nn, const LampMatrix *sample) {
    assert(nn != NULL && sample != NULL);
    const size_t input_size = nn->layers[0].activations->num_rows;
    assert(sample->num_rows == input_size);
    assert(sample->num_cols == 1);

    for (size_t i = 0; i < input_size; ++i) {
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, i, 0) =
            LAMP_MAT_ELEMENT_AT(sample, i, 0);
    }

    lamp_nn_forward(nn);
}

const LampMatrix *lamp_nn_get_output(const LampNN *nn) {
    assert(nn != NULL && nn->layer_count > 0);
    return nn->layers[nn->layer_count - 1].activations;
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

// Compute MSE loss using cached activations from the last forward pass.
// The last sample processed was at index (num_samples - 1) in the target matrix.
static LAMP_FLOAT_TYPE lamp_nn_loss_from_cache(const LampNN *nn, const LampMatrix *target, size_t num_samples) {
    assert(nn != NULL && target != NULL);
    assert(num_samples > 0);

    LAMP_FLOAT_TYPE loss = 0.0f;
    LampMatrix *output = nn->layers[nn->layer_count - 1].activations;

    for (size_t j = 0; j < target->num_cols; ++j) {
        LAMP_FLOAT_TYPE diff =
            LAMP_MAT_ELEMENT_AT(output, j, 0) -
            LAMP_MAT_ELEMENT_AT(target, num_samples - 1, j);
        loss += diff * diff;
    }
    return loss;
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
                      LAMP_FLOAT_TYPE learning_rate, LAMP_FLOAT_TYPE *out_loss) {
    assert(nn != NULL && input != NULL && target != NULL);
    assert(input->num_rows == target->num_rows);

    size_t num_samples = input->num_rows;

    // Zero pre-allocated gradient accumulators (must start fresh each backprop)
    for (size_t c = 0; c < nn->connection_count; ++c) {
        lamp_mat_fill_with(nn->connections[c].grad_w, 0.0f);
        lamp_mat_fill_with(nn->connections[c].grad_b, 0.0f);
    }

    // Process each training sample
    LAMP_FLOAT_TYPE total_loss = 0.0f;
    for (size_t s = 0; s < num_samples; ++s) {
        // Set input for this sample (row of input matrix → column vector)
        for (size_t j = 0; j < input->num_cols; ++j) {
            LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, j, 0) =
                    LAMP_MAT_ELEMENT_AT(input, s, j);
        }

        lamp_nn_forward(nn);

        // Pre-activations are cached in forward pass — no redundant matmul needed.
        // Delta computation reads from nn->layers[].pre_activations below.

        // Accumulate loss for this sample (before activation)
        if (out_loss != NULL) {
            LAMP_FLOAT_TYPE sample_loss = 0.0f;
            LampMatrix *output = nn->layers[nn->layer_count - 1].activations;
            for (size_t j = 0; j < target->num_cols; ++j) {
                LAMP_FLOAT_TYPE diff =
                    LAMP_MAT_ELEMENT_AT(output, j, 0) -
                    LAMP_MAT_ELEMENT_AT(target, s, j);
                sample_loss += diff * diff;
            }
            total_loss += sample_loss;
        }

        // Per-sample arena temporaries
        LampMatrix **delta = nn_arena_alloc(&nn->arena, nn->layer_count * sizeof(LampMatrix *));

        for (size_t i = 0; i < nn->layer_count; ++i) {
            delta[i] = nn_arena_alloc(&nn->arena, sizeof(LampMatrix));
            delta[i]->num_rows = nn->layers[i].activations->num_rows;
            delta[i]->num_cols = nn->layers[i].activations->num_cols;
            delta[i]->elements = nn_arena_alloc(&nn->arena,
                nn->layers[i].activations->num_rows * nn->layers[i].activations->num_cols * sizeof(LAMP_FLOAT_TYPE));
            for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(delta[i]); ++e) {
                delta[i]->elements[e] = 0.0f;
            }
        }

        size_t out_idx = nn->layer_count - 1;
        LampMatrix *output = nn->layers[out_idx].activations;
        for (size_t i = 0; i < output->num_rows; ++i) {
            LAMP_FLOAT_TYPE act = LAMP_MAT_ELEMENT_AT(output, i, 0);
            LAMP_FLOAT_TYPE pre_act = LAMP_MAT_ELEMENT_AT(nn->layers[out_idx].pre_activations, i, 0);
            LAMP_FLOAT_TYPE tgt = LAMP_MAT_ELEMENT_AT(target, s, i);
            LAMP_MAT_ELEMENT_AT(delta[out_idx], i, 0) = (act - tgt) * nn->config->derivative(pre_act);
        }

        for (int c = (int) nn->connection_count - 1; c >= 0; --c) {
            if (c > 0) {
                LampMatrix *w_trans = nn->connections[c].weight_trans;

                /* delta_in from arena */
                LampMatrix *delta_in = nn_arena_alloc(&nn->arena, sizeof(LampMatrix));
                delta_in->num_rows = nn->layers[c].activations->num_rows;
                delta_in->num_cols = 1;
                delta_in->elements = nn_arena_alloc(&nn->arena,
                    nn->layers[c].activations->num_rows * sizeof(LAMP_FLOAT_TYPE));

                lamp_mat_multiply_into(delta_in, w_trans, delta[c + 1]);

                for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(delta_in); ++e) {
                    delta_in->elements[e] *= nn->config->derivative(
                        nn->layers[c].pre_activations->elements[e]);
                }
                lamp_mat_copy_into(delta[c], delta_in);
            }
        }

        for (size_t c = 0; c < nn->connection_count; ++c) {
            LampMatrix *a_in = nn->connections[c].layer_begin->activations;
            LampMatrix *d_out = delta[c + 1];

            LampMatrix *a_in_trans = nn_arena_alloc(&nn->arena, sizeof(LampMatrix));
            a_in_trans->num_rows = a_in->num_cols;
            a_in_trans->num_cols = a_in->num_rows;
            a_in_trans->elements = nn_arena_alloc(&nn->arena,
                a_in->num_rows * a_in->num_cols * sizeof(LAMP_FLOAT_TYPE));

            LampMatrix *sample_grad = nn_arena_alloc(&nn->arena, sizeof(LampMatrix));
            sample_grad->num_rows = d_out->num_rows;
            sample_grad->num_cols = a_in_trans->num_cols;
            sample_grad->elements = nn_arena_alloc(&nn->arena,
                d_out->num_rows * a_in_trans->num_cols * sizeof(LAMP_FLOAT_TYPE));

            for (size_t r = 0; r < a_in->num_rows; ++r) {
                for (size_t col = 0; col < a_in->num_cols; ++col) {
                    LAMP_MAT_ELEMENT_AT(a_in_trans, col, r) = LAMP_MAT_ELEMENT_AT(a_in, r, col);
                }
            }

            lamp_mat_multiply_into(sample_grad, d_out, a_in_trans);

            for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(nn->connections[c].grad_w); ++e) {
                nn->connections[c].grad_w->elements[e] += sample_grad->elements[e];
            }

            for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(nn->connections[c].grad_b); ++e) {
                nn->connections[c].grad_b->elements[e] += d_out->elements[e];
            }
        }

        // Reset arena - all per-sample temporaries discarded
        nn_arena_reset(&nn->arena);
    }

    if (out_loss != NULL) {
        *out_loss = total_loss / (LAMP_FLOAT_TYPE) num_samples;
    }

    for (size_t c = 0; c < nn->connection_count; ++c) {
        LampNNConnection *conn = &nn->connections[c];
        for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(conn->weights); ++e) {
            conn->weights->elements[e] -=
                    learning_rate * conn->grad_w->elements[e] / (LAMP_FLOAT_TYPE) num_samples;
        }
        for (size_t e = 0; e < LAMP_MAT_NUM_ELEMENTS(conn->bias); ++e) {
            conn->bias->elements[e] -=
                    learning_rate * conn->grad_b->elements[e] / (LAMP_FLOAT_TYPE) num_samples;
        }
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

void lamp_nn_print_architecture(const LampNN *nn) {
    assert(nn != NULL && nn->layer_count > 0);

    printf("\n=== Neural Network Architecture ===\n");
    printf("Layers: %zu\n", nn->layer_count);
    printf("Connections: %zu\n\n", nn->connection_count);

    for (size_t i = 0; i < nn->layer_count; ++i) {
        printf("  Layer %zu: %zu neurons\n", i + 1, nn->layers[i].activations->num_rows);
    }
    printf("\n");

    for (size_t i = 0; i < nn->connection_count; ++i) {
        LampNNConnection *con = &nn->connections[i];
        printf("  Connection %zu: Layer %zu -> Layer %zu (%zu input neurons, %zu output neurons)\n",
               i + 1, i + 1, i + 2,
               con->layer_begin->activations->num_rows,
               con->layer_end->activations->num_rows);
    }
    printf("\n");
}

void lamp_nn_training_config_init(LampNNTrainingConfig *config) {
    assert(config != NULL);
    config->learning_rate = 0.3f;
    config->lr_decay = 1.0f;
    config->algorithm = LAMP_TRAIN_BACKPROP;
    config->finite_diff_step = 1e-4f;
    config->max_epochs = 0;  // unlimited
    config->target_loss = 0.0f;
}

LampNNTrainingConfig lamp_nn_configure_training(LAMP_FLOAT_TYPE learning_rate,
                                               LAMP_FLOAT_TYPE lr_decay,
                                               LampNNTrainingAlgorithm algorithm) {
    LampNNTrainingConfig config;
    lamp_nn_training_config_init(&config);
    config.learning_rate = learning_rate;
    config.lr_decay = lr_decay;
    config.algorithm = algorithm;
    return config;
}

LAMP_FLOAT_TYPE lamp_nn_train(LampNN *nn, const LampNNTrainingConfig *config,
                              const LampMatrix *input, const LampMatrix *target) {
    assert(nn != NULL && config != NULL && input != NULL && target != NULL);
    assert(input->num_rows == target->num_rows);

    size_t num_samples = input->num_rows;
    size_t epoch = 0;
    LAMP_FLOAT_TYPE loss = INFINITY;
    LAMP_FLOAT_TYPE current_lr = config->learning_rate;

    while (config->max_epochs == 0 || epoch < config->max_epochs) {
        if (loss <= config->target_loss) {
            break;
        }

        assert(config->algorithm == LAMP_TRAIN_BACKPROP || config->algorithm == LAMP_TRAIN_FINITE_DIFF);

        if (config->algorithm == LAMP_TRAIN_BACKPROP) {
            // Accumulate loss across all samples during backprop's forward passes
            lamp_nn_backprop(nn, input, target, current_lr, &loss);
        } else {
            lamp_nn_apply_finite_diff_gradients(nn, input, target,
                                               config->finite_diff_step, current_lr);
            loss = lamp_nn_loss(nn, input, target);
        }

        if ((epoch + 1) % 10000 == 0) {
            printf("  Epoch %4zu | loss: %.6f | lr: %.6f\n",
                   epoch + 1, loss, current_lr);
        }

        if (loss <= config->target_loss) {
            break;
        }

        current_lr *= config->lr_decay;
        epoch++;
    }

    return loss;
}

LAMP_FLOAT_TYPE lamp_nn_train_sample(LampNN *nn, const LampMatrix *sample_row,
                                      const LampMatrix *target_row, LAMP_FLOAT_TYPE learning_rate_override) {
    assert(nn != NULL && sample_row != NULL && target_row != NULL);
    assert(sample_row->num_cols == nn->layers[0].activations->num_rows);
    assert(target_row->num_cols == nn->layers[nn->layer_count - 1].activations->num_rows);
    assert(sample_row->num_rows == 1);
    assert(target_row->num_rows == 1);

    LampMatrix *input_mat = lamp_mat_alloc(1, sample_row->num_cols);
    LampMatrix *target_mat = lamp_mat_alloc(1, target_row->num_cols);

    lamp_mat_copy_into(input_mat, sample_row);
    lamp_mat_copy_into(target_mat, target_row);

    LampNNTrainingConfig config;
    lamp_nn_training_config_init(&config);
    config.learning_rate = learning_rate_override;
    config.max_epochs = 1;  // single epoch

    LAMP_FLOAT_TYPE loss = lamp_nn_train(nn, &config, input_mat, target_mat);

    lamp_mat_free(input_mat);
    lamp_mat_free(target_mat);
    return loss;
}
