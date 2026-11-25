#ifndef BENCHMARK_HARNESS_H
#define BENCHMARK_HARNESS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "matrix.h"

typedef struct {
    struct timespec start_time;
} Timer;

Timer timer_start(void);
double timer_stop(Timer *timer);

long memory_peak_rss_kb(void);
double matrix_checksum(const Matrix *m);

FILE* benchmark_create_csv(const char *dir, const char *filename);
void benchmark_close_csv(FILE *file);

void benchmark_trial(FILE *output_file, const char *method, size_t n, size_t threads, int rep, uint64_t seed);
void run_isolated_trial(FILE *output_file, const char *method, size_t n, size_t threads, int rep, uint64_t seed);

#endif
