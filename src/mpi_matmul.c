#include "mpi_matmul.h"
#include "matmul.h"
#include "parallel_matmul.h"

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *send_counts_a;
    int *displacements_a;
    int *recv_counts_c;
    int *displacements_c;
    size_t local_rows;
} RowPartition;

static bool all_ranks_succeeded(bool local_success, MPI_Comm communicator) {
    int local_flag = local_success ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MIN, communicator);
    return global_flag == 1;
}

static bool broadcast_and_validate_dimensions(const Matrix *matrix_a,
                                              const Matrix *matrix_b,
                                              int rank,
                                              MPI_Comm communicator,
                                              size_t *a_rows,
                                              size_t *a_cols,
                                              size_t *b_cols) {
    unsigned long long metadata[4] = {0, 0, 0, 0};

    if (rank == 0) {
        bool valid_pointers = (matrix_a != NULL && matrix_a->data != NULL &&
                               matrix_b != NULL && matrix_b->data != NULL);
        if (!valid_pointers) {
            printf("Error: Invalid inputs passed to matrix_mult_ikj_mpi\n");
        } else if (matrix_a->cols != matrix_b->rows) {
            printf("Error: Matrix dimension mismatch. A: %zuX%zu, B: %zuX%zu\n",
                   matrix_a->rows, matrix_a->cols, matrix_b->rows, matrix_b->cols);
        } else if (matrix_a->rows * matrix_a->cols > (size_t)INT_MAX ||
                   matrix_b->rows * matrix_b->cols > (size_t)INT_MAX ||
                   matrix_a->rows * matrix_b->cols > (size_t)INT_MAX) {
            printf("Error: Matrix exceeds MPI 32-bit element count limit\n");
        } else {
            metadata[0] = 1;
            metadata[1] = (unsigned long long)matrix_a->rows;
            metadata[2] = (unsigned long long)matrix_a->cols;
            metadata[3] = (unsigned long long)matrix_b->cols;
        }
    }

    MPI_Bcast(metadata, 4, MPI_UNSIGNED_LONG_LONG, 0, communicator);
    if (metadata[0] == 0) {
        return false;
    }

    *a_rows = (size_t)metadata[1];
    *a_cols = (size_t)metadata[2];
    *b_cols = (size_t)metadata[3];
    return true;
}

static void free_row_partition(RowPartition *partition) {
    free(partition->send_counts_a);
    free(partition->displacements_a);
    free(partition->recv_counts_c);
    free(partition->displacements_c);
    partition->send_counts_a = NULL;
    partition->displacements_a = NULL;
    partition->recv_counts_c = NULL;
    partition->displacements_c = NULL;
    partition->local_rows = 0;
}

static bool create_row_partition(size_t a_rows,
                                 size_t a_cols,
                                 size_t b_cols,
                                 int rank,
                                 int num_processes,
                                 MPI_Comm communicator,
                                 RowPartition *partition) {
    size_t proc_count = (size_t)num_processes;

    partition->send_counts_a = (int *)malloc(proc_count * sizeof(int));
    partition->displacements_a = (int *)malloc(proc_count * sizeof(int));
    partition->recv_counts_c = (int *)malloc(proc_count * sizeof(int));
    partition->displacements_c = (int *)malloc(proc_count * sizeof(int));
    partition->local_rows = 0;

    bool allocated = (partition->send_counts_a != NULL &&
                      partition->displacements_a != NULL &&
                      partition->recv_counts_c != NULL &&
                      partition->displacements_c != NULL);

    if (!all_ranks_succeeded(allocated, communicator)) {
        free_row_partition(partition);
        return false;
    }

    size_t base_rows = a_rows / proc_count;
    size_t remainder_rows = a_rows % proc_count;
    int offset_a = 0;
    int offset_c = 0;

    for (int proc = 0; proc < num_processes; proc++) {
        size_t rows_for_proc = base_rows + ((size_t)proc < remainder_rows ? 1 : 0);

        partition->send_counts_a[proc] = (int)(rows_for_proc * a_cols);
        partition->displacements_a[proc] = offset_a;
        offset_a += partition->send_counts_a[proc];

        partition->recv_counts_c[proc] = (int)(rows_for_proc * b_cols);
        partition->displacements_c[proc] = offset_c;
        offset_c += partition->recv_counts_c[proc];
    }

    partition->local_rows = base_rows + ((size_t)rank < remainder_rows ? 1 : 0);
    return true;
}

static Matrix* compute_local_product(const Matrix *local_a,
                                     const Matrix *matrix_b,
                                     ThreadPool *thread_pool) {
    if (local_a == NULL) {
        return NULL;
    }
    if (thread_pool != NULL) {
        return matrix_mult_ikj_parallel(thread_pool, local_a, matrix_b);
    }
    return matrix_mult_ikj(local_a, matrix_b);
}

static void collect_and_record_metrics(MpiMatmulMetrics *metrics,
                                       int rank,
                                       int num_processes,
                                       size_t a_rows,
                                       size_t a_cols,
                                       size_t b_cols,
                                       size_t local_rows,
                                       double local_compute_sec,
                                       double time_before_dist,
                                       double time_after_dist,
                                       double time_after_compute,
                                       double time_after_gather,
                                       MPI_Comm communicator) {
    double max_compute_sec = 0.0;
    double sum_compute_sec = 0.0;

    MPI_Reduce(&local_compute_sec, &max_compute_sec, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    MPI_Reduce(&local_compute_sec, &sum_compute_sec, 1, MPI_DOUBLE, MPI_SUM, 0, communicator);

    if (rank == 0 && metrics != NULL) {
        size_t remote_rows = a_rows - local_rows;
        metrics->distribute_sec = time_after_dist - time_before_dist;
        metrics->compute_sec = time_after_compute - time_after_dist;
        metrics->compute_max_sec = max_compute_sec;
        metrics->compute_avg_sec = sum_compute_sec / (double)num_processes;
        metrics->gather_sec = time_after_gather - time_after_compute;
        metrics->end_to_end_sec = time_after_gather - time_before_dist;
        metrics->bytes_scattered_a = remote_rows * a_cols * sizeof(double);
        metrics->bytes_broadcast_b = a_cols * b_cols * sizeof(double) * (size_t)(num_processes - 1);
        metrics->bytes_gathered_c = remote_rows * b_cols * sizeof(double);
    }
}

Matrix* matrix_mult_ikj_mpi(const Matrix *matrix_a,
                            const Matrix *matrix_b,
                            ThreadPool *thread_pool,
                            MPI_Comm communicator,
                            MpiMatmulMetrics *metrics) {
    int rank = 0;
    int num_processes = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &num_processes);

    size_t a_rows = 0;
    size_t a_cols = 0;
    size_t b_cols = 0;
    if (!broadcast_and_validate_dimensions(matrix_a, matrix_b, rank, communicator,
                                           &a_rows, &a_cols, &b_cols)) {
        return NULL;
    }

    RowPartition partition;
    if (!create_row_partition(a_rows, a_cols, b_cols, rank, num_processes,
                              communicator, &partition)) {
        return NULL;
    }

    Matrix *local_a = (partition.local_rows > 0) ? create_matrix(partition.local_rows, a_cols) : NULL;
    Matrix *received_b = (rank == 0) ? NULL : create_matrix(a_cols, b_cols);
    Matrix *result_c = (rank == 0) ? create_matrix(a_rows, b_cols) : NULL;

    bool buffers_ready = ((partition.local_rows == 0 || local_a != NULL) &&
                          (rank == 0 || received_b != NULL) &&
                          (rank != 0 || result_c != NULL));

    if (!all_ranks_succeeded(buffers_ready, communicator)) {
        matrix_free(local_a);
        matrix_free(received_b);
        matrix_free(result_c);
        free_row_partition(&partition);
        return NULL;
    }

    const Matrix *active_b = (rank == 0) ? matrix_b : received_b;
    double *broadcast_b_buffer = (rank == 0) ? matrix_b->data : received_b->data;
    double *local_a_buffer = (local_a != NULL) ? local_a->data : NULL;
    const double *root_a_buffer = (rank == 0) ? matrix_a->data : NULL;
    double *root_c_buffer = (rank == 0) ? result_c->data : NULL;

    MPI_Barrier(communicator);
    double time_before_dist = MPI_Wtime();

    MPI_Scatterv(root_a_buffer,
                 partition.send_counts_a,
                 partition.displacements_a,
                 MPI_DOUBLE,
                 local_a_buffer,
                 partition.send_counts_a[rank],
                 MPI_DOUBLE,
                 0,
                 communicator);

    MPI_Bcast(broadcast_b_buffer, (int)(a_cols * b_cols), MPI_DOUBLE, 0, communicator);

    MPI_Barrier(communicator);
    double time_after_dist = MPI_Wtime();

    double local_compute_start = MPI_Wtime();
    Matrix *local_c = compute_local_product(local_a, active_b, thread_pool);
    double local_compute_sec = MPI_Wtime() - local_compute_start;

    MPI_Barrier(communicator);
    double time_after_compute = MPI_Wtime();

    double *local_c_buffer = (local_c != NULL) ? local_c->data : NULL;

    MPI_Gatherv(local_c_buffer,
                partition.recv_counts_c[rank],
                MPI_DOUBLE,
                root_c_buffer,
                partition.recv_counts_c,
                partition.displacements_c,
                MPI_DOUBLE,
                0,
                communicator);

    MPI_Barrier(communicator);
    double time_after_gather = MPI_Wtime();

    collect_and_record_metrics(metrics,
                               rank,
                               num_processes,
                               a_rows,
                               a_cols,
                               b_cols,
                               partition.local_rows,
                               local_compute_sec,
                               time_before_dist,
                               time_after_dist,
                               time_after_compute,
                               time_after_gather,
                               communicator);

    matrix_free(local_a);
    matrix_free(received_b);
    matrix_free(local_c);
    free_row_partition(&partition);

    return result_c;
}
