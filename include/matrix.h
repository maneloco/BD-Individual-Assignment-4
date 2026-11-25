#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "rng.h"


typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Matrix;

Matrix* create_matrix(size_t rows, size_t cols);
void matrix_free(Matrix *matrix);
void print_matrix(Matrix *matrix, char *name);

void matrix_fill_zeros(Matrix *matrix);
void matrix_identity(Matrix *matrix);
void matrix_random(Matrix *matrix, RNG *rng, double min, double max);

bool matrix_are_equal(const Matrix *a, const Matrix *b, double tol_abs, double tol_rel);

#endif
