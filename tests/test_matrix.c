#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "matrix.h"

void test_create_and_zeros(void) {
    Matrix *m = create_matrix(3, 4);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 4);
    assert(m->data != NULL);

    matrix_fill_zeros(m);
    
    for (size_t i = 0; i < 12; i++) {
        assert(m->data[i] == 0.0);
    }

    matrix_free(m);
    printf("  [OK] test_create_and_zeros\n");
}

void test_free_null(void) {
    matrix_free(NULL);
    printf("  [OK] test_free_null\n");
}

void test_null_inputs(void) {
    RNG rng;
    rng_seed(&rng, 1);

    matrix_fill_zeros(NULL);
    matrix_identity(NULL);
    matrix_random(NULL, &rng, 0.0, 1.0);
    print_matrix(NULL, (char*)"null");

    Matrix *m = create_matrix(2, 2);
    matrix_fill_zeros(m);
    matrix_random(m, NULL, 0.0, 1.0);
    for (size_t i = 0; i < 4; i++) {
        assert(m->data[i] == 0.0);
    }
    matrix_free(m);

    printf("  [OK] test_null_inputs\n");
}

void test_identity(void) {
    Matrix *m = create_matrix(4, 4);
    matrix_identity(m);

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            size_t idx = i * m->cols + j;
            
            if (i == j) {
                assert(m->data[idx] == 1.0);
            } else {
                assert(m->data[idx] == 0.0);
            }
        }
    }

    matrix_free(m);
    printf("  [OK] test_identity\n");
}

void test_identity_non_square(void) {
    size_t shapes[][2] = {{4, 2}, {2, 4}, {1, 5}, {5, 1}, {3, 7}, {7, 3}};

    for (size_t s = 0; s < 6; s++) {
        size_t rows = shapes[s][0], cols = shapes[s][1];
        Matrix *m = create_matrix(rows, cols);
        matrix_identity(m);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assert(m->data[i * cols + j] == expected);
            }
        }

        matrix_free(m);
    }

    printf("  [OK] test_identity_non_square\n");
}

void test_random(void) {
    RNG rng1, rng2;
    rng_seed(&rng1, 10);
    rng_seed(&rng2, 10);

    Matrix *a = create_matrix(6, 9);
    Matrix *b = create_matrix(6, 9);
    matrix_random(a, &rng1, -3.0, 4.0);
    matrix_random(b, &rng2, -3.0, 4.0);

    for (size_t i = 0; i < 54; i++) {
        assert(a->data[i] >= -3.0 && a->data[i] <= 4.0);
    }
    assert(matrix_are_equal(a, b, 0.0, 0.0) == true);

    matrix_random(b, &rng2, -3.0, 4.0);
    assert(matrix_are_equal(a, b, 0.0, 0.0) == false);

    matrix_free(a);
    matrix_free(b);
    printf("  [OK] test_random\n");
}

void test_equality_and_tolerances(void) {
    Matrix *a = create_matrix(2, 2);
    Matrix *b = create_matrix(2, 2);
    
    matrix_fill_zeros(a);
    matrix_fill_zeros(b);

    assert(matrix_are_equal(a, b, 1e-9, 1e-9) == true);
    assert(matrix_are_equal(a, b, 0.0, 0.0) == true);

    a->data[0] = 1.0;
    assert(matrix_are_equal(a, b, 1e-9, 1e-9) == false);

    b->data[0] = 1.0 + 1e-10; 
    assert(matrix_are_equal(a, b, 1e-9, 0.0) == true);
    assert(matrix_are_equal(a, b, 0.0, 0.0) == false);

    a->data[3] = 1000000.0;
    b->data[3] = 1000000.0001; 
    assert(matrix_are_equal(a, b, 0.0, 1e-9) == true);
    assert(matrix_are_equal(a, b, 0.0, 1e-12) == false);
    assert(matrix_are_equal(a, b, 1e-9, 1e-9) == true);

    matrix_free(a);
    matrix_free(b);
    printf("  [OK] test_equality_and_tolerances\n");
}

void test_equality_symmetry(void) {
    Matrix *a = create_matrix(1, 2);
    Matrix *b = create_matrix(1, 2);

    a->data[0] = 5.0;
    a->data[1] = -3.0;
    b->data[0] = 5.0 + 1e-6;
    b->data[1] = -3.0;

    assert(matrix_are_equal(a, b, 1e-5, 0.0) == matrix_are_equal(b, a, 1e-5, 0.0));
    assert(matrix_are_equal(a, b, 1e-7, 0.0) == matrix_are_equal(b, a, 1e-7, 0.0));
    assert(matrix_are_equal(a, b, 1e-7, 0.0) == false);

    matrix_free(a);
    matrix_free(b);
    printf("  [OK] test_equality_symmetry\n");
}

void test_equality_invalid(void) {
    Matrix *a = create_matrix(2, 3);
    Matrix *b = create_matrix(3, 2);
    Matrix *c = create_matrix(2, 3);
    matrix_fill_zeros(a);
    matrix_fill_zeros(b);
    matrix_fill_zeros(c);

    assert(matrix_are_equal(a, b, 1.0, 1.0) == false);
    assert(matrix_are_equal(a, NULL, 1.0, 1.0) == false);
    assert(matrix_are_equal(NULL, a, 1.0, 1.0) == false);
    assert(matrix_are_equal(a, c, 0.0, 0.0) == true);

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
    printf("  [OK] test_equality_invalid\n");
}

void test_equality_nan(void) {
    Matrix *a = create_matrix(1, 2);
    Matrix *b = create_matrix(1, 2);
    matrix_fill_zeros(a);
    matrix_fill_zeros(b);

    a->data[0] = NAN;
    assert(matrix_are_equal(a, b, 1e9, 1e9) == false);
    b->data[0] = NAN;
    assert(matrix_are_equal(a, b, 1e9, 1e9) == false);

    matrix_free(a);
    matrix_free(b);
    printf("  [OK] test_equality_nan\n");
}

int main(void) {
    printf("=== Running Matrix tests ===\n");
    test_create_and_zeros();
    test_free_null();
    test_null_inputs();
    test_identity();
    test_identity_non_square();
    test_random();
    test_equality_and_tolerances();
    test_equality_symmetry();
    test_equality_invalid();
    test_equality_nan();
    printf("=== All Matrix tests passed ===\n\n");
    
    return 0;
}
