#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "matrix.h"
#include "matmul.h"
#include "mpi_matmul.h"
#include "rng.h"
#include "thread.h"

static void verify_mpi_result(const Matrix *mpi_result,
                              const Matrix *expected,
                              int rank,
                              const char *test_name) {
    if (rank == 0) {
        assert(mpi_result != NULL);
        if (!matrix_are_equal(mpi_result, expected, 1e-9, 1e-9)) {
            printf("  [FAIL] %s\n", test_name);
            assert(false);
        }
    } else {
        assert(mpi_result == NULL);
    }
}

static void test_mpi_known_result_2x3(int rank) {
    Matrix *matrix_a = NULL;
    Matrix *matrix_b = NULL;
    Matrix *expected = NULL;

    if (rank == 0) {
        matrix_a = create_matrix(2, 3);
        matrix_b = create_matrix(3, 2);
        expected = create_matrix(2, 2);

        double a_values[] = {1, 2, 3, 4, 5, 6};
        double b_values[] = {7, 8, 9, 10, 11, 12};
        double expected_values[] = {58, 64, 139, 154};

        for (size_t i = 0; i < 6; i++) {
            matrix_a->data[i] = a_values[i];
            matrix_b->data[i] = b_values[i];
        }
        for (size_t i = 0; i < 4; i++) {
            expected->data[i] = expected_values[i];
        }
    }

    Matrix *pure_result = matrix_mult_ikj_mpi(matrix_a, matrix_b, NULL, MPI_COMM_WORLD, NULL);
    verify_mpi_result(pure_result, expected, rank, "test_mpi_known_result_2x3 (pure)");
    matrix_free(pure_result);

    ThreadPool *pool = threadpool_create(2, 2);
    Matrix *hybrid_result = matrix_mult_ikj_mpi(matrix_a, matrix_b, pool, MPI_COMM_WORLD, NULL);
    verify_mpi_result(hybrid_result, expected, rank, "test_mpi_known_result_2x3 (hybrid)");
    matrix_free(hybrid_result);
    threadpool_destroy(pool);

    matrix_free(matrix_a);
    matrix_free(matrix_b);
    matrix_free(expected);

    if (rank == 0) {
        printf("  [OK] test_mpi_known_result_2x3\n");
    }
}

static void test_mpi_identity(int rank) {
    const size_t n = 64;
    Matrix *matrix_a = NULL;
    Matrix *identity = NULL;

    if (rank == 0) {
        RNG rng;
        rng_seed(&rng, 12345);
        matrix_a = create_matrix(n, n);
        identity = create_matrix(n, n);
        matrix_random(matrix_a, &rng, -5.0, 5.0);
        matrix_identity(identity);
    }

    Matrix *pure_result = matrix_mult_ikj_mpi(matrix_a, identity, NULL, MPI_COMM_WORLD, NULL);
    verify_mpi_result(pure_result, matrix_a, rank, "test_mpi_identity (pure)");
    matrix_free(pure_result);

    ThreadPool *pool = threadpool_create(4, 4);
    Matrix *hybrid_result = matrix_mult_ikj_mpi(matrix_a, identity, pool, MPI_COMM_WORLD, NULL);
    verify_mpi_result(hybrid_result, matrix_a, rank, "test_mpi_identity (hybrid)");
    matrix_free(hybrid_result);
    threadpool_destroy(pool);

    matrix_free(matrix_a);
    matrix_free(identity);

    if (rank == 0) {
        printf("  [OK] test_mpi_identity\n");
    }
}

static void test_mpi_zero(int rank) {
    const size_t n = 32;
    Matrix *matrix_a = NULL;
    Matrix *zeros = NULL;
    Matrix *expected = NULL;

    if (rank == 0) {
        RNG rng;
        rng_seed(&rng, 54321);
        matrix_a = create_matrix(n, n);
        zeros = create_matrix(n, n);
        expected = create_matrix(n, n);
        matrix_random(matrix_a, &rng, -10.0, 10.0);
        matrix_fill_zeros(zeros);
        matrix_fill_zeros(expected);
    }

    Matrix *result = matrix_mult_ikj_mpi(matrix_a, zeros, NULL, MPI_COMM_WORLD, NULL);
    verify_mpi_result(result, expected, rank, "test_mpi_zero");
    matrix_free(result);

    matrix_free(matrix_a);
    matrix_free(zeros);
    matrix_free(expected);

    if (rank == 0) {
        printf("  [OK] test_mpi_zero\n");
    }
}

static void test_mpi_vs_sequential_equivalence(int rank, int num_processes) {
    size_t test_sizes[] = {1, 3, 17, 32, 63, 128, 256};
    size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    size_t thread_configs[] = {0, 2, 4};
    size_t num_thread_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);

    RNG rng;
    rng_seed(&rng, 99999);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = test_sizes[s];
        Matrix *matrix_a = NULL;
        Matrix *matrix_b = NULL;
        Matrix *seq_result = NULL;

        if (rank == 0) {
            matrix_a = create_matrix(n, n);
            matrix_b = create_matrix(n, n);
            matrix_random(matrix_a, &rng, -2.0, 2.0);
            matrix_random(matrix_b, &rng, -2.0, 2.0);
            seq_result = matrix_mult_ikj(matrix_a, matrix_b);
            assert(seq_result != NULL);
        }

        for (size_t t = 0; t < num_thread_configs; t++) {
            size_t threads = thread_configs[t];
            ThreadPool *pool = (threads > 0) ? threadpool_create(threads, threads) : NULL;

            MpiMatmulMetrics metrics;
            Matrix *mpi_result = matrix_mult_ikj_mpi(matrix_a, matrix_b, pool, MPI_COMM_WORLD, &metrics);
            verify_mpi_result(mpi_result, seq_result, rank, "test_mpi_vs_sequential_equivalence");

            if (rank == 0) {
                assert(metrics.end_to_end_sec >= 0.0);
                assert(metrics.distribute_sec >= 0.0);
                assert(metrics.compute_sec >= 0.0);
                assert(metrics.gather_sec >= 0.0);
                size_t expected_bcast = n * n * sizeof(double) * (size_t)(num_processes - 1);
                assert(metrics.bytes_broadcast_b == expected_bcast);
            }

            matrix_free(mpi_result);
            if (pool != NULL) {
                threadpool_destroy(pool);
            }
        }

        matrix_free(matrix_a);
        matrix_free(matrix_b);
        matrix_free(seq_result);
    }

    if (rank == 0) {
        printf("  [OK] test_mpi_vs_sequential_equivalence\n");
    }
}

static void test_mpi_dimension_mismatch(int rank) {
    Matrix *matrix_a = (rank == 0) ? create_matrix(4, 5) : NULL;
    Matrix *matrix_b = (rank == 0) ? create_matrix(6, 4) : NULL;

    Matrix *mismatch_result = matrix_mult_ikj_mpi(matrix_a, matrix_b, NULL, MPI_COMM_WORLD, NULL);
    assert(mismatch_result == NULL);

    Matrix *null_result = matrix_mult_ikj_mpi(NULL, matrix_b, NULL, MPI_COMM_WORLD, NULL);
    assert(null_result == NULL);

    matrix_free(matrix_a);
    matrix_free(matrix_b);

    if (rank == 0) {
        printf("  [OK] test_mpi_dimension_mismatch\n");
    }
}

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int num_processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (rank == 0) {
        printf("==================================================\n");
        printf("  Running Distributed MPI Unit Tests (P = %d)\n", num_processes);
        printf("==================================================\n");
    }

    test_mpi_known_result_2x3(rank);
    test_mpi_identity(rank);
    test_mpi_zero(rank);
    test_mpi_vs_sequential_equivalence(rank, num_processes);
    test_mpi_dimension_mismatch(rank);

    if (rank == 0) {
        printf("\n===> ALL MPI TESTS PASSED SUCCESSFULLY! <===\n");
    }

    MPI_Finalize();
    return 0;
}
