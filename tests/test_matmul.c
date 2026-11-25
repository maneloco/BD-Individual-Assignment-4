#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "matmul.h"
#include "rng.h"

static Matrix* reference_mult(const Matrix *A, const Matrix *B) {
    Matrix *C = create_matrix(A->rows, B->cols);
    assert(C != NULL);

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            long double sum = 0.0L;
            for (size_t k = 0; k < A->cols; k++) {
                sum += (long double)A->data[i * A->cols + k] * (long double)B->data[k * B->cols + j];
            }
            C->data[i * B->cols + j] = (double)sum;
        }
    }
    return C;
}

static void test_known_result_2x3(void) {
    Matrix *A = create_matrix(2, 3);
    Matrix *B = create_matrix(3, 2);

    A->data[0] = 1.0; A->data[1] = 2.0; A->data[2] = 3.0;
    A->data[3] = 4.0; A->data[4] = 5.0; A->data[5] = 6.0;

    B->data[0] = 7.0;  B->data[1] = 8.0;
    B->data[2] = 9.0;  B->data[3] = 1.0;
    B->data[4] = 2.0;  B->data[5] = 3.0;

    Matrix *C = matrix_mult_ikj(A, B);
    assert(C != NULL);

    assert(C->data[0] == 31.0);
    assert(C->data[1] == 19.0);
    assert(C->data[2] == 85.0);
    assert(C->data[3] == 55.0);

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
    printf("  [OK] test_known_result_2x3\n");
}

static void test_identity(void) {
    Matrix *A = create_matrix(4, 4);
    Matrix *I = create_matrix(4, 4);
    RNG rng;
    rng_seed(&rng, 123);
    matrix_random(A, &rng, -10.0, 10.0);
    matrix_identity(I);

    Matrix *C = matrix_mult_ikj(A, I);
    assert(C != NULL);
    assert(matrix_are_equal(A, C, 1e-9, 1e-9));

    matrix_free(A);
    matrix_free(I);
    matrix_free(C);
    printf("  [OK] test_identity\n");
}

static void test_zeros(void) {
    Matrix *A = create_matrix(3, 3);
    Matrix *Z = create_matrix(3, 3);
    RNG rng;
    rng_seed(&rng, 456);
    matrix_random(A, &rng, -10.0, 10.0);
    matrix_fill_zeros(Z);

    Matrix *C = matrix_mult_ikj(A, Z);
    assert(C != NULL);
    assert(matrix_are_equal(C, Z, 1e-9, 1e-9));

    matrix_free(A);
    matrix_free(Z);
    matrix_free(C);
    printf("  [OK] test_zeros\n");
}

static void test_random_against_reference(void) {
    RNG rng;
    rng_seed(&rng, 789);

    size_t shapes[][3] = {
        {1, 1, 1},
        {5, 5, 5},
        {4, 10, 7},
        {13, 3, 17},
        {32, 32, 32}
    };

    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        size_t m = shapes[s][0];
        size_t k = shapes[s][1];
        size_t n = shapes[s][2];

        Matrix *A = create_matrix(m, k);
        Matrix *B = create_matrix(k, n);
        matrix_random(A, &rng, -5.0, 5.0);
        matrix_random(B, &rng, -5.0, 5.0);

        Matrix *C_actual = matrix_mult_ikj(A, B);
        Matrix *C_ref = reference_mult(A, B);

        assert(matrix_are_equal(C_actual, C_ref, 1e-9, 1e-9));

        matrix_free(A);
        matrix_free(B);
        matrix_free(C_actual);
        matrix_free(C_ref);
    }
    printf("  [OK] test_random_against_reference\n");
}

static void test_invalid_inputs(void) {
    Matrix *A = create_matrix(2, 3);
    Matrix *B = create_matrix(4, 2);

    assert(matrix_mult_ikj(A, B) == NULL);
    assert(matrix_mult_ikj(NULL, B) == NULL);
    assert(matrix_mult_ikj(A, NULL) == NULL);

    matrix_free(A);
    matrix_free(B);
    printf("  [OK] test_invalid_inputs\n");
}

int main(void) {
    printf("=== Running sequential matmul tests ===\n");
    test_known_result_2x3();
    test_identity();
    test_zeros();
    test_random_against_reference();
    test_invalid_inputs();
    printf("=== All sequential matmul tests passed ===\n\n");
    return 0;
}
