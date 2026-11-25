#ifndef MPI_MATMUL_H
#define MPI_MATMUL_H

#include <stddef.h>
#include <mpi.h>
#include "matrix.h"
#include "thread.h"

typedef struct {
    double distribute_sec;
    double compute_sec;
    double compute_max_sec;
    double compute_avg_sec;
    double gather_sec;
    double end_to_end_sec;
    size_t bytes_scattered_a;
    size_t bytes_broadcast_b;
    size_t bytes_gathered_c;
} MpiMatmulMetrics;

Matrix* matrix_mult_ikj_mpi(const Matrix *matrix_a,
                            const Matrix *matrix_b,
                            ThreadPool *thread_pool,
                            MPI_Comm communicator,
                            MpiMatmulMetrics *metrics);

#endif
