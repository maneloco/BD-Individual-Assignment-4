#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <pthread.h>

#include "matrix.h"
#include "matmul.h"
#include "rng.h"
#include "thread.h"
#include "parallel_matmul.h"

typedef struct {
    int *array;
    int index;
    int value;
} SimpleTaskArg;

static void task_write_value(void *arg) {
    SimpleTaskArg *data = (SimpleTaskArg *)arg;
    data->array[data->index] = data->value;
}

static void test_pool_basic_lifecycle(void) {
    const size_t num_threads = 4;
    const size_t num_tasks = 16;

    ThreadPool *pool = threadpool_create(num_threads, num_tasks);
    assert(pool != NULL);

    int results[16] = {0};
    SimpleTaskArg args[16];

    for (size_t i = 0; i < num_tasks; i++) {
        args[i].array = results;
        args[i].index = (int)i;
        args[i].value = (int)(i * 10);
        bool ok = threadpool_add(pool, task_write_value, &args[i]);
        assert(ok == true);
    }

    threadpool_wait(pool);

    for (size_t i = 0; i < num_tasks; i++) {
        assert(results[i] == (int)(i * 10));
    }

    threadpool_destroy(pool);
    printf("  [OK] test_pool_basic_lifecycle\n");
}

typedef struct {
    pthread_mutex_t *lock;
    long *counter;
} CounterTaskArg;

static void task_increment_counter(void *arg) {
    CounterTaskArg *data = (CounterTaskArg *)arg;
    pthread_mutex_lock(data->lock);
    (*(data->counter))++;
    pthread_mutex_unlock(data->lock);
}

static void test_pool_high_volume_tasks(void) {
    const size_t num_threads = 4;
    const size_t total_tasks = 500;
    const size_t queue_capacity = 100;

    ThreadPool *pool = threadpool_create(num_threads, queue_capacity);
    assert(pool != NULL);

    pthread_mutex_t counter_lock;
    pthread_mutex_init(&counter_lock, NULL);
    long shared_counter = 0;

    CounterTaskArg arg = {
        .lock = &counter_lock,
        .counter = &shared_counter
    };

    size_t submitted = 0;
    while (submitted < total_tasks) {
        size_t batch = (total_tasks - submitted > 50) ? 50 : (total_tasks - submitted);
        for (size_t i = 0; i < batch; i++) {
            bool ok = threadpool_add(pool, task_increment_counter, &arg);
            assert(ok == true);
        }
        submitted += batch;
        threadpool_wait(pool);
    }

    assert(shared_counter == (long)total_tasks);

    pthread_mutex_destroy(&counter_lock);
    threadpool_destroy(pool);
    printf("  [OK] test_pool_high_volume_tasks\n");
}

static void test_parallel_known_result_2x3(void) {
    Matrix *A = create_matrix(2, 3);
    Matrix *B = create_matrix(3, 2);
    double a_val[] = {1, 2, 3, 4, 5, 6};
    double b_val[] = {7, 8, 9, 10, 11, 12};
    for (size_t i = 0; i < 6; i++) {
        A->data[i] = a_val[i];
        B->data[i] = b_val[i];
    }

    Matrix *expected = create_matrix(2, 2);
    double expected_val[] = {58, 64, 139, 154};
    for (size_t i = 0; i < 4; i++) {
        expected->data[i] = expected_val[i];
    }

    size_t thread_counts[] = {1, 2, 3, 4};
    for (size_t t = 0; t < 4; t++) {
        size_t threads = thread_counts[t];
        ThreadPool *pool = threadpool_create(threads, threads);

        Matrix *C = matrix_mult_ikj_parallel(pool, A, B);
        assert(C != NULL);
        assert(matrix_are_equal(C, expected, 1e-9, 1e-9));

        matrix_free(C);
        threadpool_destroy(pool);
    }

    matrix_free(A);
    matrix_free(B);
    matrix_free(expected);
    printf("  [OK] test_parallel_known_result_2x3\n");
}

static void test_parallel_identity(void) {
    const size_t n = 64;
    RNG rng;
    rng_seed(&rng, 12345);

    Matrix *A = create_matrix(n, n);
    Matrix *I = create_matrix(n, n);
    matrix_random(A, &rng, -5.0, 5.0);
    matrix_identity(I);

    size_t thread_counts[] = {1, 2, 4, 7, 8};
    for (size_t t = 0; t < 5; t++) {
        size_t threads = thread_counts[t];
        ThreadPool *pool = threadpool_create(threads, threads);

        Matrix *C = matrix_mult_ikj_parallel(pool, A, I);
        assert(C != NULL);
        assert(matrix_are_equal(C, A, 1e-9, 1e-9));

        matrix_free(C);
        threadpool_destroy(pool);
    }

    matrix_free(A);
    matrix_free(I);
    printf("  [OK] test_parallel_identity\n");
}

static void test_parallel_zero(void) {
    const size_t n = 32;
    RNG rng;
    rng_seed(&rng, 54321);

    Matrix *A = create_matrix(n, n);
    Matrix *Z = create_matrix(n, n);
    Matrix *expected = create_matrix(n, n);
    matrix_random(A, &rng, -10.0, 10.0);
    matrix_fill_zeros(Z);
    matrix_fill_zeros(expected);

    ThreadPool *pool = threadpool_create(4, 4);
    Matrix *C = matrix_mult_ikj_parallel(pool, A, Z);
    assert(C != NULL);
    assert(matrix_are_equal(C, expected, 1e-9, 1e-9));

    matrix_free(A);
    matrix_free(Z);
    matrix_free(expected);
    matrix_free(C);
    threadpool_destroy(pool);
    printf("  [OK] test_parallel_zero\n");
}

static void test_parallel_vs_sequential_equivalence(void) {
    RNG rng;
    rng_seed(&rng, 99999);

    size_t test_sizes[] = {8, 17, 32, 63, 128, 256};
    size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    size_t test_threads[] = {1, 2, 3, 4, 8, 16};
    size_t num_thread_configs = sizeof(test_threads) / sizeof(test_threads[0]);

    for (size_t s = 0; s < num_sizes; s++) {
        size_t n = test_sizes[s];

        Matrix *A = create_matrix(n, n);
        Matrix *B = create_matrix(n, n);
        matrix_random(A, &rng, -2.0, 2.0);
        matrix_random(B, &rng, -2.0, 2.0);

        Matrix *C_seq = matrix_mult_ikj(A, B);
        assert(C_seq != NULL);

        for (size_t t = 0; t < num_thread_configs; t++) {
            size_t threads = test_threads[t];
            ThreadPool *pool = threadpool_create(threads, threads);

            Matrix *C_par = matrix_mult_ikj_parallel(pool, A, B);
            assert(C_par != NULL);

            if (!matrix_are_equal(C_seq, C_par, 1e-9, 1e-9)) {
                printf("  [FAIL] Equivalence failed for n=%zu with threads=%zu!\n", n, threads);
                assert(false);
            }

            matrix_free(C_par);
            threadpool_destroy(pool);
        }

        matrix_free(A);
        matrix_free(B);
        matrix_free(C_seq);
    }

    printf("  [OK] test_parallel_vs_sequential_equivalence\n");
}

static void test_parallel_dimension_mismatch(void) {
    Matrix *A = create_matrix(4, 5);
    Matrix *B = create_matrix(6, 4);

    ThreadPool *pool = threadpool_create(2, 2);

    Matrix *C = matrix_mult_ikj_parallel(pool, A, B);
    assert(C == NULL);

    Matrix *C_null = matrix_mult_ikj_parallel(pool, NULL, B);
    assert(C_null == NULL);

    Matrix *C_nopool = matrix_mult_ikj_parallel(NULL, A, B);
    assert(C_nopool == NULL);

    matrix_free(A);
    matrix_free(B);
    threadpool_destroy(pool);
    printf("  [OK] test_parallel_dimension_mismatch\n");
}

int main(void) {
    printf("==================================================\n");
    printf("  Running Parallel & Thread Pool Unit Tests       \n");
    printf("==================================================\n");

    printf("\n[1] Thread Pool Tests:\n");
    test_pool_basic_lifecycle();
    test_pool_high_volume_tasks();

    printf("\n[2] Parallel Matrix Multiplication Correctness:\n");
    test_parallel_known_result_2x3();
    test_parallel_identity();
    test_parallel_zero();
    test_parallel_vs_sequential_equivalence();
    test_parallel_dimension_mismatch();

    printf("\n===> ALL TESTS PASSED SUCCESSFULLY! <===\n");
    return 0;
}
