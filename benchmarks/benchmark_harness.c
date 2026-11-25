#define _POSIX_C_SOURCE 199309L

#include "benchmark_harness.h"
#include "matmul.h"
#include "parallel_matmul.h"
#include "thread.h"
#include "rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

Timer timer_start(void) {
    Timer t;
    clock_gettime(CLOCK_MONOTONIC, &t.start_time);
    return t;
}

double timer_stop(Timer *timer) {
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    return (double)(end_time.tv_sec - timer->start_time.tv_sec) +
           (double)(end_time.tv_nsec - timer->start_time.tv_nsec) * 1e-9;
}

long memory_peak_rss_kb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

double matrix_checksum(const Matrix *m) {
    if (m == NULL || m->data == NULL) return 0.0;
    double sum = 0.0;
    size_t total = m->rows * m->cols;
    for (size_t i = 0; i < total; i++) {
        sum += m->data[i];
    }
    return sum;
}

FILE* benchmark_create_csv(const char *dir, const char *filename) {
    mkdir(dir, 0755);

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s", dir, filename);

    FILE *file = fopen(filepath, "w");
    if (file == NULL) {
        printf("Error: Cannot create output CSV file: %s\n", filepath);
        return NULL;
    }

    fprintf(file, "category,method,n,threads,repetition,time_sec,peak_rss_kb,checksum\n");
    fflush(file);
    return file;
}

void benchmark_close_csv(FILE *file) {
    if (file != NULL) {
        fclose(file);
    }
}

void benchmark_trial(FILE *output_file, const char *method, size_t n, size_t threads, int rep, uint64_t seed) {
    RNG rng;
    rng_seed(&rng, seed + (uint64_t)(rep >= 0 ? rep : 9999));

    Matrix *A = create_matrix(n, n);
    Matrix *B = create_matrix(n, n);
    matrix_random(A, &rng, -2.0, 2.0);
    matrix_random(B, &rng, -2.0, 2.0);

    Matrix *C = NULL;
    double elapsed = 0.0;

    if (strcmp(method, "seq_ikj") == 0) {
        Timer timer = timer_start();
        C = matrix_mult_ikj(A, B);
        elapsed = timer_stop(&timer);
    } else if (strcmp(method, "par_ikj") == 0) {
        ThreadPool *pool = threadpool_create(threads, threads);
        Timer timer = timer_start();
        C = matrix_mult_ikj_parallel(pool, A, B);
        elapsed = timer_stop(&timer);
        threadpool_destroy(pool);
    }

    double checksum = matrix_checksum(C);
    long peak_rss = memory_peak_rss_kb();

    if (output_file != NULL && rep >= 0) {
        fprintf(output_file, "parallel,%s,%zu,%zu,%d,%.9f,%ld,%.4f\n",
                method, n, threads, rep, elapsed, peak_rss, checksum);
        fflush(output_file);
    }

    const char *rep_label = (rep < 0) ? "WARMUP" : "REP";
    printf("  [%s] %-8s n=%-4zu threads=%-2zu %s=%-2d time=%.6fs rss=%-6ldKB\n",
           method, (rep < 0 ? "warmup" : "trial"), n, threads, rep_label, rep, elapsed, peak_rss);

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

void run_isolated_trial(FILE *output_file, const char *method, size_t n, size_t threads, int rep, uint64_t seed) {
    pid_t pid = fork();
    if (pid == 0) {
        benchmark_trial(output_file, method, n, threads, rep, seed);
        exit(0);
    }
    waitpid(pid, NULL, 0);
}
