#ifndef BENCHMARK_HARNESS_H
#define BENCHMARK_HARNESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include "matrix.h"

typedef struct {
    struct timespec start_time;
} Timer;

typedef struct {
    long rank0_rss_kb;
    long worker_max_rss_kb;
    long total_rss_kb;
} ClusterMemoryMetrics;

Timer timer_start(void);
double timer_stop(Timer *timer);

long memory_peak_rss_kb(void);
double matrix_checksum(const Matrix *matrix);

FILE* benchmark_open_csv(const char *dir, const char *filename, bool append_mode);
void benchmark_close_csv(FILE *file);

void run_isolated_local_trial(FILE *output_file,
                              const char *experiment,
                              const char *method,
                              size_t n,
                              size_t threads,
                              int rep,
                              uint64_t seed);

void run_mpi_trial(FILE *output_file,
                   const char *experiment,
                   const char *method,
                   size_t n,
                   size_t threads_per_proc,
                   int rep,
                   uint64_t seed,
                   MPI_Comm communicator);

#endif
