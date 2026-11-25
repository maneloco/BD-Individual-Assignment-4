#ifndef PARALLEL_MATMUL_H
#define PARALLEL_MATMUL_H

#include "matrix.h"
#include "thread.h"
#include <stddef.h>

typedef struct {
    const Matrix *A;
    const Matrix *B;
    Matrix *C;
    size_t start_row;
    size_t end_row;
} MatmulSliceTask;

void matmul_ikj_worker(void *arg);
Matrix* matrix_mult_ikj_parallel(ThreadPool *pool, const Matrix *A, const Matrix *B);

#endif
