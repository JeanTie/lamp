#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include "../src/linear_algebra/lamp_matrix.h"

#define LAMP_TEST_FAILED 0x00
#define LAMP_TEST_PASSED 0x01

typedef struct {
    bool (*test_function)(void);

    char *desc;
} LampTest;

bool test_matrix_fill(void) {
    LampMatrix *mat = lamp_mat_alloc(2, 2);
    LAMP_FLOAT_TYPE filler = 0.0f;
    lamp_mat_fill_with(mat, filler);

    bool filled_correctly = LAMP_TEST_PASSED;
    for (size_t i = 0; i < LAMP_MAT_NUM_ELEMENTS(mat); ++i) {
        if (mat->elements[i] != filler) {
            filled_correctly = LAMP_TEST_FAILED;
        }
    }

    lamp_mat_free(mat);
    return filled_correctly;
}

bool test_matrix_randomize(void) {
    LampMatrix *mat = lamp_mat_alloc(2, 2);
    LAMP_FLOAT_TYPE def_val = 0.0f;
    lamp_mat_fill_with(mat, def_val);

    lamp_mat_rand(mat);

    bool randomized = LAMP_TEST_FAILED;
    for (size_t i = 0; i < LAMP_MAT_NUM_ELEMENTS(mat); ++i) {
        if (mat->elements[i] != def_val) {
            randomized = LAMP_TEST_PASSED;
        }
    }

    lamp_mat_free(mat);
    return randomized;
}

bool test_matrix_equals(void) {
    const size_t rows = 2;
    const size_t cols = 2;
    LampMatrix *m1 = lamp_mat_alloc(rows, cols);
    LampMatrix *m2 = lamp_mat_alloc(rows, cols);

    lamp_mat_fill_with(m1, 0.0f);
    lamp_mat_fill_with(m2, 0.0f);

    if (!lamp_matrix_equal(m1, m2)) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        return LAMP_TEST_FAILED;
    }

    lamp_mat_rand(m1);
    if (lamp_matrix_equal(m1, m2)) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        return LAMP_TEST_FAILED;
    }
    lamp_mat_free(m1);

    LampMatrix *uneq_mat = lamp_mat_alloc(rows + 1, cols);
    if (lamp_matrix_equal(uneq_mat, m2)) {
        lamp_mat_free(uneq_mat);
        lamp_mat_free(m2);
        return LAMP_TEST_FAILED;
    }

    lamp_mat_free(uneq_mat);
    lamp_mat_free(m2);
    return LAMP_TEST_PASSED;
}

bool test_matrix_copies(void) {
    LampMatrix *m1 = lamp_mat_alloc(2, 2);
    lamp_mat_fill_with(m1, 0.0f);
    LampMatrix *m2 = lamp_mat_alloc_copy(m1);

    if (!lamp_matrix_equal(m1, m2)) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        return LAMP_TEST_FAILED;
    }

    lamp_mat_fill_with(m2, 1.0f);
    lamp_mat_copy_into(m1, m2);

    if (!lamp_matrix_equal(m1, m2)) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        return LAMP_TEST_FAILED;
    }

    lamp_mat_free(m1);
    lamp_mat_free(m2);
    return LAMP_TEST_PASSED;
}

bool test_matrix_multiplication(void) {
    // Calculate a simple example that can be verified quickly by humans
    // [3, 2, 1]
    // [1, 0, 2] M1
    //
    // [1, 2]
    // [0, 1]
    // [4, 0] M2
    //
    // [7, 8]
    // [9, 2] MR
    LampMatrix *m1 = lamp_mat_alloc(2, 3);
    LAMP_MAT_ELEMENT_AT(m1, 0, 0) = 3;
    LAMP_MAT_ELEMENT_AT(m1, 0, 1) = 2;
    LAMP_MAT_ELEMENT_AT(m1, 0, 2) = 1;
    LAMP_MAT_ELEMENT_AT(m1, 1, 0) = 1;
    LAMP_MAT_ELEMENT_AT(m1, 1, 1) = 0;
    LAMP_MAT_ELEMENT_AT(m1, 1, 2) = 2;

    LampMatrix *m2 = lamp_mat_alloc(3, 2);
    LAMP_MAT_ELEMENT_AT(m2, 0, 0) = 1;
    LAMP_MAT_ELEMENT_AT(m2, 0, 1) = 2;
    LAMP_MAT_ELEMENT_AT(m2, 1, 0) = 0;
    LAMP_MAT_ELEMENT_AT(m2, 1, 1) = 1;
    LAMP_MAT_ELEMENT_AT(m2, 2, 0) = 4;
    LAMP_MAT_ELEMENT_AT(m2, 2, 1) = 0;

    LampMatrix *mr = lamp_mat_alloc(m1->num_rows, m2->num_cols);
    LAMP_MAT_ELEMENT_AT(mr, 0, 0) = 7;
    LAMP_MAT_ELEMENT_AT(mr, 0, 1) = 8;
    LAMP_MAT_ELEMENT_AT(mr, 1, 0) = 9;
    LAMP_MAT_ELEMENT_AT(mr, 1, 1) = 2;

    LampMatrix *mt = lamp_mat_alloc_multiply(m1, m2);
    if (!lamp_matrix_equal(mt, mr)) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        lamp_mat_free(mr);
        lamp_mat_free(mt);
        return LAMP_TEST_FAILED;
    }

    lamp_mat_fill_with(mt, 0.0f);
    lamp_mat_multiply_into(mt, m1, m2);

    if (!lamp_matrix_equal(mt, mr)) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        lamp_mat_free(mr);
        lamp_mat_free(mt);
        return LAMP_TEST_FAILED;
    }
    lamp_mat_free(m1);
    lamp_mat_free(m2);
    lamp_mat_free(mr);
    lamp_mat_free(mt);
    return LAMP_TEST_PASSED;
}

bool test_matrix_allocation(void) {
    LampMatrix *m = lamp_mat_alloc_identity(2);
    if (m == NULL) {
        return LAMP_TEST_FAILED;
    }

    LampMatrix *mi = lamp_mat_alloc(2, 2);
    LAMP_MAT_ELEMENT_AT(mi, 0, 0) = 1;
    LAMP_MAT_ELEMENT_AT(mi, 0, 1) = 0;
    LAMP_MAT_ELEMENT_AT(mi, 1, 0) = 0;
    LAMP_MAT_ELEMENT_AT(mi, 1, 1) = 1;

    if (!lamp_matrix_equal(m, mi)) {
        lamp_mat_free(m);
        lamp_mat_free(mi);
        return LAMP_TEST_FAILED;
    }
    lamp_mat_free(m);
    lamp_mat_free(mi);
    return LAMP_TEST_PASSED;
}

bool test_matrix_transpose(void) {
    LampMatrix *m1 = lamp_mat_alloc(3, 1);
    LAMP_MAT_ELEMENT_AT(m1, 0, 0) = 0;
    LAMP_MAT_ELEMENT_AT(m1, 1, 0) = 1;
    LAMP_MAT_ELEMENT_AT(m1, 2, 0) = 2;

    LampMatrix *m2 = lamp_mat_transpose(m1);
    if ((m1->num_rows != m2->num_cols) || (m1->num_cols != m2->num_rows) ||
        (LAMP_MAT_ELEMENT_AT(m1, 0, 0) != LAMP_MAT_ELEMENT_AT(m2, 0, 0)) ||
        (LAMP_MAT_ELEMENT_AT(m1, 1, 0) != LAMP_MAT_ELEMENT_AT(m2, 0, 1)) ||
        (LAMP_MAT_ELEMENT_AT(m1, 2, 0) != LAMP_MAT_ELEMENT_AT(m2, 0, 2))) {
        lamp_mat_free(m1);
        lamp_mat_free(m2);
        return LAMP_TEST_FAILED;
    }
    return LAMP_TEST_PASSED;
}

static void show_result(bool success, char *test_name) {
    printf("TEST: %s \t%s\n", test_name, success == LAMP_TEST_PASSED ? "SUCCESS" : "FAILED");
}

static bool run_test(const LampTest *test) {
    assert(test != NULL);
    return test->test_function();
}

int main() {
    printf("LAMP Tests\n");

    LampTest test_mat_fill = {test_matrix_fill, "Matrix fill"};
    show_result(run_test(&test_mat_fill), test_mat_fill.desc);

    LampTest test_mat_rand = {test_matrix_randomize, "Matrix randomize"};
    show_result(run_test(&test_mat_rand), test_mat_rand.desc);

    LampTest test_mat_equal = {test_matrix_equals, "Matrix equals"};
    show_result(run_test(&test_mat_equal), test_mat_equal.desc);

    LampTest test_mat_copy = {test_matrix_copies, "Matrix copies"};
    show_result(run_test(&test_mat_copy), test_mat_copy.desc);

    LampTest test_mat_mult = {test_matrix_multiplication, "Matrix mult"};
    show_result(run_test(&test_mat_mult), test_mat_mult.desc);

    LampTest test_mat_alloc = {test_matrix_allocation, "Matrix alloc"};
    show_result(run_test(&test_mat_alloc), test_mat_alloc.desc);

    LampTest test_mat_transpose = {test_matrix_transpose, "Matrix transpose"};
    show_result(run_test(&test_mat_transpose), test_mat_transpose.desc);

    return 0;
}
