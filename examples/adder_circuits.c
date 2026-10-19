//
// Created by Jan Thieme on 23.02.2025.
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/neural_network/lamp_nn.h"

#define LEARNING_RATE 1e-1f
#define FINITE_DIFF_STEP 1e-1f

#define HALF_ADD_INPUTS 2
#define HALF_ADD_HIDDEN 2
#define HALF_ADD_OUT 2

void binary_adder_nbit(size_t num_bits,
                       const size_t architecture[], size_t layer_count,
                       const LampNNActivationConfig *activation,
                       LAMP_FLOAT_TYPE learning_rate,
                       size_t max_epochs,
                       LAMP_FLOAT_TYPE target_loss)
{
    const size_t num_samples = (size_t)pow(4.0, (double)num_bits);
    const size_t input_cols = (size_t)2 * num_bits;
    const size_t target_cols = (size_t)num_bits + 1;

    LAMP_FLOAT_TYPE *input_data = (LAMP_FLOAT_TYPE *)calloc(
        num_samples * input_cols, sizeof(LAMP_FLOAT_TYPE));
    LAMP_FLOAT_TYPE *target_data = (LAMP_FLOAT_TYPE *)calloc(
        num_samples * target_cols, sizeof(LAMP_FLOAT_TYPE));

    const size_t half_range = (size_t)1 << num_bits;  /* 2^num_bits */

    for (size_t i = 0; i < num_samples; ++i) {
        size_t a = i / half_range;
        size_t b = i % half_range;
        size_t sum = a + b;

        // Encode a as num_bits binary (LSB first) into columns 0..num_bits-1
        for (size_t bit = 0; bit < num_bits; ++bit) {
            input_data[i * input_cols + bit] = (LAMP_FLOAT_TYPE)((a >> bit) & 1);
        }

        // Encode b as num_bits binary (LSB first) into columns num_bits..2*num_bits-1
        for (size_t bit = 0; bit < num_bits; ++bit) {
            input_data[i * input_cols + num_bits + bit] = (LAMP_FLOAT_TYPE)((b >> bit) & 1);
        }

        // Write sum bits (LSB first) into target columns 0..num_bits
        for (size_t bit = 0; bit < target_cols; ++bit) {
            target_data[i * target_cols + bit] = (LAMP_FLOAT_TYPE)((sum >> bit) & 1);
        }
    }

    LampMatrix *input = lamp_mat_alloc_from_array(num_samples, input_cols, input_data);
    LampMatrix *target = lamp_mat_alloc_from_array(num_samples, target_cols, target_data);

    LampNN *nn = lamp_nn_alloc(architecture, layer_count);
    lamp_nn_set_activation(nn, activation);

    LampNNTrainingConfig config = lamp_nn_configure_training(learning_rate, 1.0f, LAMP_TRAIN_BACKPROP);
    config.max_epochs = max_epochs;
    config.target_loss = target_loss;

    LAMP_FLOAT_TYPE final_loss = lamp_nn_train(nn, &config, input, target);

    printf("=== Binary Adder %zu-bit ===\n", num_bits);

    size_t correct = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        // Reconstruct original a and b for display
        size_t a = i / half_range;
        size_t b = i % half_range;

        LAMP_FLOAT_TYPE *sample_data = (LAMP_FLOAT_TYPE *)malloc(input_cols * sizeof(LAMP_FLOAT_TYPE));
        for (size_t c = 0; c < input_cols; ++c) {
            sample_data[c] = LAMP_MAT_ELEMENT_AT(input, i, c);
        }
        LampMatrix *sample = lamp_mat_alloc_from_array(input_cols, 1, sample_data);

        lamp_nn_forward_single(nn, sample);
        const LampMatrix *out = lamp_nn_get_output(nn);

        size_t *pred_bits = (size_t *)malloc(target_cols * sizeof(size_t));
        size_t *actual_bits = (size_t *)malloc(target_cols * sizeof(size_t));
        int sample_correct = 1;

        for (size_t bit = 0; bit < target_cols; ++bit) {
            LAMP_FLOAT_TYPE val = (LAMP_FLOAT_TYPE)out->elements[bit];
            pred_bits[bit] = (val >= 0.5f) ? 1 : 0;
            actual_bits[bit] = (size_t)LAMP_MAT_ELEMENT_AT(target, i, bit);

            if (pred_bits[bit] != actual_bits[bit]) {
                sample_correct = 0;
            }
        }

        if (sample_correct) {
            ++correct;
        }

        // Print formatted result
        // Binary string LSB first
        printf("a=");
        for (size_t bit = 0; bit < num_bits; ++bit) {
            printf("%zu", (a >> bit) & 1);
        }
        printf(", b=");
        for (size_t bit = 0; bit < num_bits; ++bit) {
            printf("%zu", (b >> bit) & 1);
        }
        printf(" -> pred=");
        for (size_t bit = 0; bit < target_cols; ++bit) {
            printf("%zu", pred_bits[bit]);
        }
        printf(", actual=");
        for (size_t bit = 0; bit < target_cols; ++bit) {
            printf("%zu", actual_bits[bit]);
        }
        printf(" | acc=%s\n", sample_correct ? "PASS" : "FAIL");

        free(sample_data);
        lamp_mat_free(sample);
        free(pred_bits);
        free(actual_bits);
    }

    double accuracy = (double)correct / (double)num_samples * 100.0;
    printf("Overall accuracy: %.2f%% (%zu/%zu)\n", accuracy, correct, num_samples);
    printf("Final loss: %f\n", final_loss);

    free(input_data);
    free(target_data);
    lamp_mat_free(input);
    lamp_mat_free(target);
    lamp_nn_free(nn);
}

int main() {
    // Try learning behavior of adder curcuits

    srand(time(NULL)); // NOLINT: We know about srand() initialization

    LAMP_FLOAT_TYPE ins_ha[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };
    LampMatrix *input = lamp_mat_alloc_from_array(4, 2, ins_ha);

    const LAMP_FLOAT_TYPE targs_ha[] = {
        0, 0,
        1, 0,
        1, 0,
        0, 1
    };


    size_t half_add_hidden[] = {HALF_ADD_HIDDEN};
    LampNN *nn = lamp_nn_alloc_with(HALF_ADD_INPUTS, 1, half_add_hidden, HALF_ADD_OUT);

    LampMatrix *target = lamp_mat_alloc_from_array(input->num_rows, 2, targs_ha);

    LampNNTrainingConfig config = lamp_nn_configure_training(1.0f, 1.0f, LAMP_TRAIN_BACKPROP);
    config.max_epochs = 10 * 1000;
    config.target_loss = 0.01f;
    LAMP_FLOAT_TYPE final_loss = lamp_nn_train(nn, &config, input, target);
    printf("Half-adder final loss: %f\n", final_loss);

    for (int it = 0; it < input->num_rows; ++it) {
        float sample_data[] = {
            (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(input, it, 0),
            (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(input, it, 1)
        };
        LampMatrix *sample = lamp_mat_alloc_from_array(HALF_ADD_INPUTS, 1, sample_data);

        lamp_nn_forward_single(nn, sample);
        const LampMatrix *out = lamp_nn_get_output(nn);

        printf("[%f, %f] -> [%f, %f] (%f, %f)\n",
               LAMP_MAT_ELEMENT_AT(input, it, 0),
               LAMP_MAT_ELEMENT_AT(input, it, 1),
               (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(out, 0, 0),
               (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(out, 1, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 1));

        lamp_mat_free(sample);
    }
    printf("\n");

    lamp_mat_free(input);
    lamp_mat_free(target);
    lamp_nn_free(nn);

    LAMP_FLOAT_TYPE ins_fa[] = {
        0, 0, 0,
        0, 0, 1,
        0, 1, 0,
        0, 1, 1,
        1, 0, 0,
        1, 0, 1,
        1, 1, 0,
        1, 1, 1
    };
    input = lamp_mat_alloc_from_array(8, 3, ins_fa);

    const LAMP_FLOAT_TYPE targs_fa[] = {
        0, 0,
        1, 0,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        0, 1,
        1, 1
    };

    // NOTE: For this full adder problem we have to change the architecture, since we have to take
    //       more inputs and outputs into account.

    size_t fa_hidden[] = {8, 3};
    nn = lamp_nn_alloc_with(3, 2, fa_hidden, 2);

    target = lamp_mat_alloc_from_array(input->num_rows, 2, targs_fa);

    LampNNTrainingConfig config_fa = lamp_nn_configure_training(1.0f, 1.0f, LAMP_TRAIN_BACKPROP);
    config_fa.max_epochs = 100 * 1000;
    config_fa.target_loss = 0.005f;
    final_loss = lamp_nn_train(nn, &config_fa, input, target);
    printf("Full-adder final loss: %f\n", final_loss);

    for (int it = 0; it < input->num_rows; ++it) {
        float sample_data[] = {
            (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(input, it, 0),
            (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(input, it, 1),
            (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(input, it, 2)
        };
        LampMatrix *sample = lamp_mat_alloc_from_array(3, 1, sample_data);

        lamp_nn_forward_single(nn, sample);
        const LampMatrix *out = lamp_nn_get_output(nn);

        printf("[%f, %f, %f] -> [%f, %f] (%f, %f)\n",
               LAMP_MAT_ELEMENT_AT(input, it, 0),
               LAMP_MAT_ELEMENT_AT(input, it, 1),
               LAMP_MAT_ELEMENT_AT(input, it, 2),
               (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(out, 0, 0),
               (LAMP_FLOAT_TYPE) LAMP_MAT_ELEMENT_AT(out, 1, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 1)
        );

        lamp_mat_free(sample);
    }
    printf("\n");
    lamp_mat_free(input);
    lamp_mat_free(target);
    lamp_nn_free(nn);

    // TODO: Optimize SOMETHING - training larger models on this old/low-end machine is not fun at all.
    // N-bit binary adder test
    size_t num_bits = 2;
    size_t arch_bit[] = {2*num_bits, 8, num_bits + 1};

    binary_adder_nbit(
        num_bits,
        arch_bit,
        sizeof(arch_bit) / sizeof(arch_bit[0]),
        &LAMP_ACTIVATION_SIGMOID,
        1.0f,
        50000,
        0.005f
    );
    printf("\n");

    return 0;
}
