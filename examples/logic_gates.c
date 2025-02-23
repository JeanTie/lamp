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

#include <assert.h>
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include "../src/neural_network/lamp_nn.h"

// We create a 2x2x1 network
// 2 inputs, 2 hidden nodes and one output
// | i1 | - | h1 |\
// |    | - |    | - | o |
// | i2 | - | h2 |/
#define NUM_INPUT_NODES 2
#define NUM_HIDDEN_NODES 2
#define NUM_OUTPUT_NODES 1

#define LEARNING_RATE 1.0f
#define FINITE_DIFF_STEP 0.5f

#define NUMBER_OF_GATES 6
#define NUMBER_OF_STATES 4

int main() {
    // Try learning behavior of logic gates - it is the 'Hello World!' of neural networks
    // NOTE: The functions of logic gates are relatively easy to approximate.
    //       That allows us to use high learning rate and finite difference step.
    //
    // Running this example will show the output of a network trained to behave like one of six logic gates.
    // As it is known solving the XOR, XNOR gates is what the network struggles with the most.

    srand(time(NULL)); // NOLINT: We know about srand() initialization

    LAMP_FLOAT_TYPE ins[] = {0, 0,
                             0, 1,
                             1, 0,
                             1, 1};
    LampMatrix *input = lamp_mat_alloc_from_array(4, NUM_INPUT_NODES, ins);

    const LAMP_FLOAT_TYPE targs[NUMBER_OF_GATES][NUMBER_OF_STATES] = {
            {0, 0, 0, 1}, // AND
            {1, 1, 1, 0}, // NAND
            {0, 1, 1, 1}, // OR
            {1, 0, 0, 0}, // NOR
            {0, 1, 1, 0}, // XOR
            {1, 0, 0, 1}, // XNOR
    };

    const char *gate_descriptions[NUMBER_OF_GATES]  = {
            "AND",
            "NAND",
            "OR",
            "NOR",
            "XOR",
            "XNOR"
    };


    size_t architecture[] = {NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES};
    LampNN *nn = lamp_nn_alloc(architecture, sizeof(architecture) / sizeof(architecture[0]));

    for (int i = 0; i < NUMBER_OF_GATES; ++i) {
        LampMatrix *target = lamp_mat_alloc_from_array(input->num_rows, 1, targs[i]);
        for (int j = 0; j < nn->connection_count; ++j) {
            lamp_mat_rand(nn->connections[j].weights);
            lamp_mat_rand(nn->connections[j].bias);
        }

        for (int e = 0; e < 10 * 1000; ++e) {
            lamp_nn_apply_finite_diff_gradients(nn, input, target, FINITE_DIFF_STEP, LEARNING_RATE);
            LAMP_FLOAT_TYPE loss = lamp_nn_loss(nn, input, target);
//        printf("Loss %f\n", loss);
        }

        printf("%s:\n", gate_descriptions[i]);
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
        printf("\n");
        lamp_mat_free(target);
    }

    lamp_nn_free(nn);

    return 0;
}