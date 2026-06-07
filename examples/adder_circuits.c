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
    for (int i = 0; i < nn->connection_count; ++i) {
        lamp_mat_rand(nn->connections[i].weights);
        lamp_mat_rand(nn->connections[i].bias);
        lamp_mat_fill_with(nn->connections[i].layer_begin->activations, 0.0f);
        lamp_mat_fill_with(nn->connections[i].layer_end->activations, 0.0f);
    }

    LampMatrix *target = lamp_mat_alloc_from_array(input->num_rows, 2, targs_ha);


    for (int e = 0; e < 10 * 1000; ++e) {
        lamp_nn_backprop(nn, input, target, 1.0f);
        LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
        //        printf("Loss %f\n", loss);
    }

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
    for (int i = 0; i < nn->connection_count; ++i) {
        lamp_mat_rand(nn->connections[i].weights);
        lamp_mat_rand(nn->connections[i].bias);
        lamp_mat_fill_with(nn->connections[i].layer_begin->activations, 0.0f);
        lamp_mat_fill_with(nn->connections[i].layer_end->activations, 0.0f);
    }

    target = lamp_mat_alloc_from_array(input->num_rows, 2, targs_fa);

    for (int e = 0; e < 100 * 1000; ++e) {
        lamp_nn_backprop(nn, input, target, 1.0f);
        if ((e % 10000) == 0) {
            LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
            printf("Loss %f\n", loss);
        }
    }

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

    return 0;
}
