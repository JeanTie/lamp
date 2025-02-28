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

    LAMP_FLOAT_TYPE ins_ha[] = {0, 0,
                                0, 1,
                                1, 0,
                                1, 1};
    LampMatrix *input = lamp_mat_alloc_from_array(4, 2, ins_ha);

    const LAMP_FLOAT_TYPE targs_ha[] = {
            0, 0,
            1, 0,
            1, 0,
            0, 1
    };


    size_t architecture[] = {HALF_ADD_INPUTS, HALF_ADD_HIDDEN, HALF_ADD_OUT};
    LampNN *nn = lamp_nn_alloc(architecture, sizeof(architecture) / sizeof(architecture[0]));
    for (int i = 0; i < nn->connection_count; ++i) {
        lamp_mat_rand(nn->connections[i].weights);
        lamp_mat_rand(nn->connections[i].bias);
        lamp_mat_fill_with(nn->connections[i].layer_begin->activations, 0.0f);
        lamp_mat_fill_with(nn->connections[i].layer_end->activations, 0.0f);
    }

    LampMatrix *target = lamp_mat_alloc_from_array(input->num_rows, 2, targs_ha);


    for (int e = 0; e < 10 * 1000; ++e) {
        lamp_nn_apply_finite_diff_gradients(nn, input, target, FINITE_DIFF_STEP, LEARNING_RATE);
        LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
//        printf("Loss %f\n", loss);
    }

    for (int it = 0; it < input->num_rows; ++it) {
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 0, 0) = LAMP_MAT_ELEMENT_AT(input, it, 0);
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 1, 0) = LAMP_MAT_ELEMENT_AT(input, it, 1);
        lamp_nn_forward(nn);
        printf("[%f, %f] -> [%f, %f] (%f, %f)\n",
               LAMP_MAT_ELEMENT_AT(input, it, 0),
               LAMP_MAT_ELEMENT_AT(input, it, 1),
               LAMP_MAT_ELEMENT_AT(nn->layers[nn->layer_count - 1].activations, 0, 0),
               LAMP_MAT_ELEMENT_AT(nn->layers[nn->layer_count - 1].activations, 1, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 1));
    }
    printf("\n");

    lamp_mat_free(input);
    lamp_mat_free(target);
    lamp_nn_free(nn);

    LAMP_FLOAT_TYPE ins_fa[] = {0, 0, 0,
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

    size_t fa_arch[] = {3, 4, 2};
    nn = lamp_nn_alloc(fa_arch, sizeof(fa_arch) / sizeof(fa_arch[0]));
    for (int i = 0; i < nn->connection_count; ++i) {
        lamp_mat_rand(nn->connections[i].weights);
        lamp_mat_rand(nn->connections[i].bias);
        lamp_mat_fill_with(nn->connections[i].layer_begin->activations, 0.0f);
        lamp_mat_fill_with(nn->connections[i].layer_end->activations, 0.0f);
    }
    lamp_nn_print(nn);

    target = lamp_mat_alloc_from_array(input->num_rows, 2, targs_fa);

    // This got fascinating.
    // I was unable to sufficiently train this network to behave like a full adder.
    // Since I do not have a clue what I am doing - yet - I just toyed around with some values,
    // to see if I can find a working configuration manually. Without any great success. Adding more hidden layers,
    // resulted in the same result (in the best case, often it got worse).
    // On some lucky seeds I was able to train the network to a loss of ~0.375, where it plateaued.
    // Maybe this is a local minimum of the adder? Maybe the approximation with the finite difference method
    // is not good enough? Maybe I am just not smart enough to see the obvious?
    LAMP_FLOAT_TYPE l_rate = 1e-2f;
    LAMP_FLOAT_TYPE fds = 1e-1f;

    for (int e = 0; e < 100 * 1000; ++e) {
        lamp_nn_apply_finite_diff_gradients(nn, input, target, fds, l_rate);

            LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
            printf("Loss %f\n", loss);

    }

    for (int it = 0; it < input->num_rows; ++it) {
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 0, 0) = LAMP_MAT_ELEMENT_AT(input, it, 0);
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 1, 0) = LAMP_MAT_ELEMENT_AT(input, it, 1);
        LAMP_MAT_ELEMENT_AT(nn->layers[0].activations, 2, 0) = LAMP_MAT_ELEMENT_AT(input, it, 2);
        lamp_nn_forward(nn);
        printf("[%f, %f, %f] -> [%f, %f] (%f, %f)\n",
               LAMP_MAT_ELEMENT_AT(input, it, 0),
               LAMP_MAT_ELEMENT_AT(input, it, 1),
               LAMP_MAT_ELEMENT_AT(input, it, 2),
               LAMP_MAT_ELEMENT_AT(nn->layers[nn->layer_count - 1].activations, 0, 0),
               LAMP_MAT_ELEMENT_AT(nn->layers[nn->layer_count - 1].activations, 1, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 0),
               LAMP_MAT_ELEMENT_AT(target, it, 1)
        );
    }
    printf("\n");
    lamp_nn_print(nn);

    return 0;
}
