#include "matrix.h"
#include "rng.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Matrix* create_matrix(size_t rows, size_t cols) {
    Matrix *mat = malloc(sizeof(Matrix));
    if (mat == NULL) {
        printf("The matrix could not be initialized.\n");
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;

    mat->data = (double *)malloc(rows * cols * sizeof(double));
    if (mat->data == NULL) {
        free(mat);
        printf("The matrix could not be initialized.\n");
        return NULL;
    }
    return mat;
}

void matrix_free(Matrix *matrix) {
    if (matrix == NULL) {
        return;
    }
    free(matrix->data);
    free(matrix);
}

void print_matrix(Matrix *matrix, char *name) {
    if (matrix == NULL || matrix->data == NULL) {
        printf("It is not a valid matrix.\n");
        return;
    }

    size_t rows = matrix->rows, cols = matrix->cols;
    printf("Matrix %s: %zuX%zu\n", name, rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%f ", matrix->data[i*cols + j]);
        }
        printf("\n");
    }
}

void matrix_fill_zeros(Matrix *matrix) {
    if (matrix == NULL || matrix->data == NULL) {
        printf("The matrix is not initialized.\n");
        return;
    }

    for (size_t i = 0; i < matrix->cols * matrix->rows; i++) {
        matrix->data[i] = 0;
    }
} 

void matrix_identity(Matrix *matrix) {
    if (matrix == NULL || matrix->data == NULL) {
        printf("The matrix is not initialized.\n");
        return;
    }
    
    matrix_fill_zeros(matrix);
    size_t diagonal = matrix->rows < matrix->cols ? matrix->rows : matrix->cols;
    for (size_t i = 0; i < diagonal; i++) {
        matrix->data[i * matrix->cols + i] = 1;
    }
}

void matrix_random(Matrix *matrix, RNG *rng, double min, double max) {
    if (matrix == NULL || matrix->data == NULL || rng == NULL) {
        printf("Error with the input.\n");
        return;
    }

    for (size_t i = 0; i < matrix->cols * matrix->rows; i++) {
        matrix->data[i] = rng_next_range(rng, min, max);
    }
}

bool matrix_are_equal(const Matrix *a, const Matrix *b, double tol_abs, double tol_rel) {
    if (a == NULL || a->data == NULL || b == NULL || b->data == NULL) {
        printf("The matrix must be created.\n");
        return false;
    }

    if (a->rows != b->rows || a->cols != b->cols) {
        printf("The matrices don't have the same dimension.\n");
        return false;
    }
    
    double local_max, diff;
    for (size_t i = 0; i < a->cols * a->rows; i++) {
        if (fabs(a->data[i]) > fabs(b->data[i])) {
            local_max = fabs(a->data[i]);
        } else {
            local_max = fabs(b->data[i]);
        }

        diff = fabs(a->data[i] - b->data[i]);
        
        if (!(diff <= tol_abs + tol_rel * local_max)) {
            return false;
        }
    }

    return true;
}

