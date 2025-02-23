#include <assert.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include "neural_network/lamp_nn.h"

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

int main() {
    // Try learning behavior of logic gates - because everybody does this in the beginning ;)

    // TODO: Find good solution to initialize srand()
    srand(time(NULL)); // NOLINT: We know about srand() initialization

    LAMP_FLOAT_TYPE ins[] = {0, 0,
                             0, 1,
                             1, 0,
                             1, 1};
    LampMatrix *input = lamp_mat_alloc_from_array(4, NUM_INPUT_NODES, ins);

    // AND-Gate
    LAMP_FLOAT_TYPE targs[] = {0, 0, 0, 1};
    LampMatrix *target = lamp_mat_alloc_from_array(input->num_rows, 1, targs);

    size_t architecture[] = {NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES};
    LampNN *nn = lamp_nn_alloc(architecture, sizeof(architecture) / sizeof(architecture[0]));
    for (int i = 0; i < nn->connection_count; ++i) {
        lamp_mat_rand(nn->connections[i].weights);
        lamp_mat_rand(nn->connections[i].bias);
    }

    for (int e = 0; e < 10 * 1000; ++e) {
        lamp_nn_apply_finite_diff_gradients(nn, input, target, FINITE_DIFF_STEP, LEARNING_RATE);
        LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
        printf("Loss %f\n", loss);
    }

    for (int it = 0; it < input->num_rows; ++it) {
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 0, 0) = LAMP_MAT_ELEMENT_AT(input, it, 0);
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 1, 0) = LAMP_MAT_ELEMENT_AT(input, it, 1);
        lamp_nn_forward(nn);
        printf("[%f, %f] -> [%f] (%f)\n",
               LAMP_MAT_ELEMENT_AT(input, it, 0),
               LAMP_MAT_ELEMENT_AT(input, it, 1),
               LAMP_MAT_ELEMENT_AT(nn->layers[nn->layer_count - 1].activations, 0, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 0));
    }

    lamp_nn_free(nn);

    return 0;
}