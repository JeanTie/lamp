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

#define NUMBER_OF_ACTIVATION_FUNCS 3

int main() {
    // Try learning behavior of logic gates - because everybody does this in the beginning ;)

    const LampNNActivationConfig activation_configs[NUMBER_OF_ACTIVATION_FUNCS] = {
        LAMP_ACTIVATION_SIGMOID,
        LAMP_ACTIVATION_RELU,
        LAMP_ACTIVATION_TANH,
    };

    for (int i = 0; i < NUMBER_OF_ACTIVATION_FUNCS; ++i) {
        // TODO: Find good solution to initialize srand()
        srand(time(NULL)); // NOLINT: We know about srand() initialization

        LAMP_FLOAT_TYPE ins[] = {
            0, 0,
            0, 1,
            1, 0,
            1, 1
        };
        LampMatrix *input = lamp_mat_alloc_from_array(4, NUM_INPUT_NODES, ins);

        // AND-Gate
        LAMP_FLOAT_TYPE targs[] = {0, 0, 0, 1};
        LampMatrix *target = lamp_mat_alloc_from_array(input->num_rows, 1, targs);

        size_t hidden[] = {NUM_HIDDEN_NODES};
        LampNN *nn = lamp_nn_alloc_with(NUM_INPUT_NODES, 1, hidden, NUM_OUTPUT_NODES);
        const LampNNActivationConfig *act_config = &activation_configs[i];
        lamp_nn_set_activation(nn, act_config);

        for (int e = 0; e < 10 * 1000; ++e) {
            lamp_nn_backprop(nn, input, target, act_config->learning_rate);
            LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
            // printf("Loss %f\n", loss);
        }

        const char *activation_names[] = {
            "LAMP_ACTIVATION_SIGMOID",
            "LAMP_ACTIVATION_RELU",
            "LAMP_ACTIVATION_TANH",
        };

        printf("Result with %s\n", activation_names[i]);

        for (int it = 0; it < input->num_rows; ++it) {

            float sample_data[] = {
                LAMP_MAT_ELEMENT_AT(input, it, 0),
                LAMP_MAT_ELEMENT_AT(input, it, 1)
            };
            LampMatrix *sample = lamp_mat_alloc_from_array(NUM_INPUT_NODES, 1, sample_data);

            lamp_nn_forward_single(nn, sample);
            const LampMatrix *out = lamp_nn_get_output(nn);

            printf("[%f, %f] -> [%f]    (target: %f)\n",
                   LAMP_MAT_ELEMENT_AT(input, it, 0),
                   LAMP_MAT_ELEMENT_AT(input, it, 1),
                   LAMP_MAT_ELEMENT_AT(out, 0, 0),
                   LAMP_MAT_ELEMENT_AT(target, it, 0));

            lamp_mat_free(sample);
        }

        lamp_nn_free(nn);
    }

    return 0;
}
