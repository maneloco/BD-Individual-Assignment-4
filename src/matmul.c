#include "matmul.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

Matrix* matrix_mult_ikj(const Matrix *A, const Matrix *B) {
    if (A == NULL || A->data == NULL || B == NULL || B->data == NULL) {
        printf("Please, use valid matrices\n");
        return NULL;
    }

    if (A->cols != B->rows) {
        printf("Cannot mult this matrices. A: %zuX%zu, B: %zuX%zu\n", A->rows, A->cols, B->rows, B->cols);
        return NULL;
    }

    size_t a_rows = A->rows, a_cols = A->cols, b_cols = B->cols;
    
    Matrix *C = create_matrix(a_rows, b_cols);
    if (C == NULL) {
        return NULL;
    }
    matrix_fill_zeros(C);

    for (size_t i = 0; i < a_rows; i++) {
        for (size_t k = 0; k < a_cols; k++) {
            double a_value = A->data[i * a_cols + k];

            for (size_t j = 0; j < b_cols; j++) {
                C->data[i * b_cols + j] += a_value * B->data[k * b_cols + j];
            }
        }
    }

    return C;
}
