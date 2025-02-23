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

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lamp_matrix.h"

// Get a pseudo random floating point value between 0.0 and 1.0 inclusive
static LAMP_FLOAT_TYPE rand_f_normalized(void) {
    return (LAMP_FLOAT_TYPE) rand() / // NOLINT(cert-msc30-c, cert-msc50-cpp) suppress warning about limited randomness
           (LAMP_FLOAT_TYPE) RAND_MAX;
}

LampMatrix *lamp_mat_alloc(size_t rows, size_t cols) {
    assert(rows >= 1 && cols >= 1);

    // TODO: Propagate memory allocation error instead of asserting here
    LampMatrix *mat = malloc(sizeof(LampMatrix));
    assert(mat != NULL);
    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->elements = malloc(sizeof(LAMP_FLOAT_TYPE) * rows * cols);
    assert(mat->elements != NULL);

    return mat;
}

void lamp_mat_free(LampMatrix *mat) {
    assert(mat != NULL);
    free(mat->elements);
    free(mat);
}

void lamp_mat_fill_with(LampMatrix *mat, LAMP_FLOAT_TYPE filler) {
    assert(mat != NULL);

    for (size_t i = 0; i < LAMP_MAT_NUM_ELEMENTS(mat); ++i) {
        mat->elements[i] = filler;
    }
}

void lamp_mat_rand(LampMatrix *mat) {
    for (size_t i = 0; i < LAMP_MAT_NUM_ELEMENTS(mat); ++i) {
        mat->elements[i] = rand_f_normalized();
    }
}

LampMatrix *lamp_mat_alloc_identity(size_t size) {
    assert(size >= 1);
    LampMatrix *mi = lamp_mat_alloc(size, size);
    for (size_t i = 0; i < LAMP_MAT_NUM_ELEMENTS(mi); ++i) {
        size_t row = i / mi->num_rows;
        size_t col = i % mi->num_rows;

        if (row == col) {
            mi->elements[i] = 1.0f;
        } else {
            mi->elements[i] = 0.0f;
        }
    }

    return mi;
}

// Allocate matrix of specified size with content of a flattened 1D array
LampMatrix *lamp_mat_alloc_from_array(size_t rows, size_t cols, const LAMP_FLOAT_TYPE *content) {
    LampMatrix *mat = lamp_mat_alloc(rows, cols);
    for (size_t i = 0; i < mat->num_rows; ++i) {
        for (size_t j = 0; j < mat->num_cols; ++j) {
            LAMP_MAT_ELEMENT_AT(mat, i, j) = content[i * mat->num_cols + j];
        }
    }
    return mat;
}

bool lamp_matrix_equal_dimensions(const LampMatrix *m1, const LampMatrix *m2) {
    return ((m1->num_rows == m2->num_rows) && (m1->num_cols == m2->num_cols));
}

bool lamp_matrix_equal(const LampMatrix *m1, const LampMatrix *m2) {
    assert(m1 != NULL && m2 != NULL);

    if (!lamp_matrix_equal_dimensions(m1, m2)) {
        // Matrix dimensions not identical
        return false;
    }

    // For comparing the elements we unfortunately have to deal with floating point shenanigans,
    // so we assume a "reasonable" tolerable difference of the values.
    const LAMP_FLOAT_TYPE tolerance = 0.000001f;
    for (size_t i = 0; i < m1->num_rows; ++i) {
        for (size_t j = 0; j < m1->num_cols; ++j) {
            if (LAMP_FABS(LAMP_MAT_ELEMENT_AT(m1, i, j) - LAMP_MAT_ELEMENT_AT(m2, i, j)) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

void lamp_mat_copy_into(LampMatrix *dst, const LampMatrix *src) {
    assert(dst != NULL && src != NULL);
    assert(lamp_matrix_equal_dimensions(dst, src));
    memcpy(dst->elements, src->elements, LAMP_MAT_NUM_ELEMENTS(src) * sizeof(LAMP_FLOAT_TYPE));
}

LampMatrix *lamp_mat_alloc_copy(const LampMatrix *m_to_copy) {
    assert(m_to_copy != NULL);
    LampMatrix *mc = lamp_mat_alloc(m_to_copy->num_rows, m_to_copy->num_cols);

    lamp_mat_copy_into(mc, m_to_copy);
    return mc;
}

void lamp_mat_multiply_into(LampMatrix *dst, const LampMatrix *m1, const LampMatrix *m2) {
    assert(m1->num_cols == m2->num_rows);
    assert((dst->num_rows == m1->num_rows) && (dst->num_cols == m2->num_cols));

    lamp_mat_fill_with(dst, 0.0f);

    // It is probably also not very efficient and the easiest to read, but at least for me, it is clear
    // how the matrix multiplication is executed.
    for (size_t i = 0; i < dst->num_rows; ++i) {
        for (size_t j = 0; j < dst->num_cols; ++j) {
            for (size_t k = 0; k < m1->num_cols; ++k) {
                LAMP_FLOAT_TYPE elem1 = LAMP_MAT_ELEMENT_AT(m1, i, k);
                LAMP_FLOAT_TYPE elem2 = LAMP_MAT_ELEMENT_AT(m2, k, j);
                dst->elements[LAMP_MAT_ELEMENT_IDX(dst, i, j)] += elem1 * elem2;
            }
        }
    }
}

LampMatrix *lamp_mat_alloc_multiply(const LampMatrix *m1, const LampMatrix *m2) {
    assert(m1 != NULL && m2 != NULL);

    LampMatrix *dst = lamp_mat_alloc(m1->num_rows, m2->num_cols);
    lamp_mat_multiply_into(dst, m1, m2);
    return dst;

}

void lamp_mat_add(LampMatrix *dst, const LampMatrix *src) {
    assert(dst != NULL);
    assert(src != NULL);
    assert(lamp_matrix_equal_dimensions(dst, src));

    for (size_t i = 0; i < dst->num_rows; ++i) {
        for (size_t j = 0; j < dst->num_cols; ++j) {
            LAMP_MAT_ELEMENT_AT(dst, i, j) += LAMP_MAT_ELEMENT_AT(src, i, j);
        }
    }
}

LampMatrix *lamp_mat_alloc_sum(const LampMatrix *src1, const LampMatrix *src2) {
    assert(src1 != NULL);
    assert(src2 != NULL);

    LampMatrix *sum = lamp_mat_alloc_copy(src1);
    lamp_mat_add(sum, src2);
    return sum;
}

LampMatrix *lamp_mat_transpose(const LampMatrix *m) {
    assert(m != NULL);
    LampMatrix *mt = lamp_mat_alloc(m->num_cols, m->num_rows);

    for (size_t i = 0; i < m->num_rows; ++i) {
        for (size_t j = 0; j < m->num_cols; ++j) {
            LAMP_MAT_ELEMENT_AT(mt, j, i) = LAMP_MAT_ELEMENT_AT(m, i, j);
        }
    }

    return mt;
}