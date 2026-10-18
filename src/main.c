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

    LAMP_ACTIVATION_RELU.learning_rate = 0.24f;
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

    // XOR demo with new training API
    printf("\n========================================\n");
    printf("  Neural Network Training Demo (XOR)\n");
    printf("========================================\n");

    srand(42);

    size_t xor_arch[] = {2, 4, 1};
    LampNN *xor_nn = lamp_nn_alloc(xor_arch, sizeof(xor_arch) / sizeof(xor_arch[0]));
    lamp_nn_set_activation(xor_nn, &LAMP_ACTIVATION_SIGMOID);

    lamp_nn_print_architecture(xor_nn);

    float xor_inputs[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };
    LampMatrix *xor_input = lamp_mat_alloc_from_array(4, 2, xor_inputs);

    float xor_targets[] = {
        0, 1, 1, 0
    };
    LampMatrix *xor_target = lamp_mat_alloc_from_array(4, 1, xor_targets);

    LampNNTrainingConfig xor_config = lamp_nn_configure_training(1.0f, 1.0f, LAMP_TRAIN_BACKPROP);
    xor_config.max_epochs = 10000;
    xor_config.target_loss = 0.005f;

    printf("Training for up to %zu epochs (target loss: %f)...\n\n",
           xor_config.max_epochs, xor_config.target_loss);

    LAMP_FLOAT_TYPE final_loss = lamp_nn_train(xor_nn, &xor_config, xor_input, xor_target);

    printf("\nTraining complete. Final loss: %f\n\n", final_loss);

    printf("XOR predictions:\n");
    for (int i = 0; i < 4; ++i) {
        float sample_data[] = {
            LAMP_MAT_ELEMENT_AT(xor_input, i, 0),
            LAMP_MAT_ELEMENT_AT(xor_input, i, 1)
        };
        LampMatrix *sample = lamp_mat_alloc_from_array(2, 1, sample_data);

        lamp_nn_forward_single(xor_nn, sample);
        const LampMatrix *out = lamp_nn_get_output(xor_nn);

        float prediction = (float)LAMP_MAT_ELEMENT_AT(out, 0, 0);
        const char *bit = prediction >= 0.5f ? "1" : "0";
        printf("  [%d, %d] -> %f (predicted: %s, expected: %d)\n",
               (int)sample_data[0], (int)sample_data[1],
               prediction, bit,
               (int)xor_targets[i]);

        lamp_mat_free(sample);
    }

    // Single-sample training demo
    printf("\nSingle-sample training demo (train on [1,1] -> 0 for 100 epochs):\n");
    float test_sample[] = {1.0f, 1.0f};
    float test_target[] = {0.0f};
    LampMatrix *train_in = lamp_mat_alloc_from_array(1, 2, test_sample);
    LampMatrix *train_tgt = lamp_mat_alloc_from_array(1, 1, test_target);
    LampMatrix *forward_in = lamp_mat_alloc_from_array(2, 1, test_sample);

    for (int i = 0; i < 100; ++i) {
        lamp_nn_train_sample(xor_nn, train_in, train_tgt, 0.3f);
        if (i % 25 == 0) {
            lamp_nn_forward_single(xor_nn, forward_in);
            const LampMatrix *result = lamp_nn_get_output(xor_nn);
            printf("  epoch %3d: output = %f\n", i, (float)LAMP_MAT_ELEMENT_AT(result, 0, 0));
        }
    }

    printf("\n");

    lamp_mat_free(xor_input);
    lamp_mat_free(xor_target);
    lamp_mat_free(train_in);
    lamp_mat_free(train_tgt);
    lamp_mat_free(forward_in);
    lamp_nn_free(xor_nn);

    return 0;
}
