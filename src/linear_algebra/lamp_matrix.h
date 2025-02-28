//
// Created by Jan Thieme on 16.02.2025.
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

#ifndef LAMP_LAMP_MATRIX_H
#define LAMP_LAMP_MATRIX_H

#include <stddef.h>
#include <stdbool.h>

#define LAMP_FLOAT_TYPE float // To make it easier to use double if we want to
#define LAMP_FABS(f) fabsf(f) // Absolute function differs for double and float

typedef struct {
    size_t num_rows;
    size_t num_cols;
    LAMP_FLOAT_TYPE *elements;
} LampMatrix;

#define LAMP_MAT_NUM_ELEMENTS(p_M) (p_M->num_rows * p_M->num_cols)
#define LAMP_MAT_ELEMENT_IDX(p_M, row, col) ((row * p_M->num_cols) + col)
#define LAMP_MAT_ELEMENT_AT(p_M, row, col) (p_M->elements[LAMP_MAT_ELEMENT_IDX(p_M, row, col)])

LampMatrix *lamp_mat_alloc(size_t rows, size_t cols);

void lamp_mat_free(LampMatrix *mat);

void lamp_mat_fill_with(LampMatrix *mat, LAMP_FLOAT_TYPE filler);

void lamp_mat_rand(LampMatrix *mat);

LampMatrix *lamp_mat_alloc_identity(size_t size);

LampMatrix *lamp_mat_alloc_from_array(size_t rows, size_t cols, const LAMP_FLOAT_TYPE *content);

bool lamp_matrix_equal_dimensions(const LampMatrix *m1, const LampMatrix *m2);

bool lamp_matrix_equal(const LampMatrix *m1, const LampMatrix *m2);

// ATTENTION: Users must assure the dst and src matrices have the same dimensions
// TODO: Maybe we find a nicer API, that clarifies the dimension requirements?
void lamp_mat_copy_into(LampMatrix *dst, const LampMatrix *src);

LampMatrix *lamp_mat_alloc_copy(const LampMatrix *m_to_copy);

// ATTENTION: Users must assure the dst, m1 and m2 matrices are aligned properly.
//            m1.n_cols == m2.n_rows && (dst.n_rows == m1.n_rows && dst.n_cols == m2.n_cols)
void lamp_mat_multiply_into(LampMatrix *dst, const LampMatrix *m1, const LampMatrix *m2);

LampMatrix *lamp_mat_alloc_multiply(const LampMatrix *m1, const LampMatrix *m2);

// Add src to dst
// ATTENTION: Matrix addition requires matrices of equal dimension
void lamp_mat_add(LampMatrix *dst, const LampMatrix *src);

LampMatrix *lamp_mat_alloc_sum(const LampMatrix *src1, const LampMatrix *src2);

LampMatrix *lamp_mat_transpose(const LampMatrix *m);

void lamp_mat_print(const LampMatrix *m);

#endif //LAMP_LAMP_MATRIX_H
