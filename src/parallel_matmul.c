#include "parallel_matmul.h"
#include "matrix.h"
#include "thread.h"

#include <stdio.h>
#include <stdlib.h>

void matmul_ikj_worker(void *arg) {
    if (arg == NULL) return;

    MatmulSliceTask *task = (MatmulSliceTask *)arg;

    const Matrix *A = task->A;
    const Matrix *B = task->B;
    Matrix *C = task->C;

    size_t a_cols = A->cols;
    size_t b_cols = B->cols;

    const double * restrict a_data = A->data;
    const double * restrict b_data = B->data;
    double * restrict c_data = C->data;

    for (size_t i = task->start_row; i < task->end_row; i++) {
        for (size_t k = 0; k < a_cols; k++) {
            double a_value = a_data[i * a_cols + k];

            for (size_t j = 0; j < b_cols; j++) {
                c_data[i * b_cols + j] += a_value * b_data[k * b_cols + j];
            }
        }
    }
}

Matrix* matrix_mult_ikj_parallel(ThreadPool *pool, const Matrix *A, const Matrix *B) {
    if (pool == NULL || A == NULL || A->data == NULL || B == NULL || B->data == NULL) {
        printf("Error: Invalid inputs passed to matrix_mult_ikj_parallel\n");
        return NULL;
    }

    if (A->cols != B->rows) {
        printf("Error: Matrix dimension mismatch. A: %zuX%zu, B: %zuX%zu\n",
               A->rows, A->cols, B->rows, B->cols);
        return NULL;
    }

    size_t a_rows = A->rows;
    size_t b_cols = B->cols;

    Matrix *C = create_matrix(a_rows, b_cols);
    if (C == NULL) {
        return NULL;
    }
    matrix_fill_zeros(C);

    size_t num_threads = pool->num_threads;

    MatmulSliceTask *tasks = (MatmulSliceTask *)malloc(sizeof(MatmulSliceTask) * num_threads);
    if (tasks == NULL) {
        matrix_free(C);
        return NULL;
    }

    size_t base_rows = a_rows / num_threads;
    size_t remainder = a_rows % num_threads;
    size_t current_row = 0;

    for (size_t t = 0; t < num_threads; t++) {
        size_t rows_for_thread = base_rows + (t < remainder ? 1 : 0);

        tasks[t].A = A;
        tasks[t].B = B;
        tasks[t].C = C;
        tasks[t].start_row = current_row;
        tasks[t].end_row = current_row + rows_for_thread;

        current_row += rows_for_thread;

        if (tasks[t].start_row < tasks[t].end_row) {
            threadpool_add(pool, matmul_ikj_worker, &tasks[t]);
        }
    }

    threadpool_wait(pool);
    free(tasks);

    return C;
}
