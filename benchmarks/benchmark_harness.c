#define _POSIX_C_SOURCE 199309L

#include "benchmark_harness.h"
#include "matmul.h"
#include "mpi_matmul.h"
#include "parallel_matmul.h"
#include "rng.h"
#include "thread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

Timer timer_start(void) {
    Timer timer;
    clock_gettime(CLOCK_MONOTONIC, &timer.start_time);
    return timer;
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

double matrix_checksum(const Matrix *matrix) {
    if (matrix == NULL || matrix->data == NULL) {
        return 0.0;
    }
    double sum = 0.0;
    size_t total = matrix->rows * matrix->cols;
    for (size_t i = 0; i < total; i++) {
        sum += matrix->data[i];
    }
    return sum;
}

FILE* benchmark_open_csv(const char *dir, const char *filename, bool append_mode) {
    mkdir(dir, 0755);

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s", dir, filename);

    bool file_exists = (access(filepath, F_OK) == 0);
    const char *mode = (append_mode && file_exists) ? "a" : "w";

    FILE *file = fopen(filepath, mode);
    if (file == NULL) {
        printf("Error: Cannot open output CSV file: %s\n", filepath);
        return NULL;
    }

    if (!append_mode || !file_exists) {
        fprintf(file,
                "experiment,method,n,processes,threads_per_proc,total_workers,repetition,"
                "end_to_end_sec,distribute_sec,compute_sec,compute_max_sec,compute_avg_sec,"
                "gather_sec,rank0_rss_kb,worker_max_rss_kb,total_rss_kb,bytes_transferred,checksum\n");
        fflush(file);
    }

    return file;
}

void benchmark_close_csv(FILE *file) {
    if (file != NULL) {
        fclose(file);
    }
}

static uint64_t trial_seed(uint64_t base_seed, int rep) {
    return base_seed + (uint64_t)(rep >= 0 ? rep : 9999);
}

static void execute_local_trial(FILE *output_file,
                                const char *experiment,
                                const char *method,
                                size_t n,
                                size_t threads,
                                int rep,
                                uint64_t seed) {
    RNG rng;
    rng_seed(&rng, trial_seed(seed, rep));

    Matrix *matrix_a = create_matrix(n, n);
    Matrix *matrix_b = create_matrix(n, n);
    matrix_random(matrix_a, &rng, -2.0, 2.0);
    matrix_random(matrix_b, &rng, -2.0, 2.0);

    Matrix *result_c = NULL;
    double elapsed_sec = 0.0;

    if (strcmp(method, "seq_ikj") == 0) {
        Timer timer = timer_start();
        result_c = matrix_mult_ikj(matrix_a, matrix_b);
        elapsed_sec = timer_stop(&timer);
    } else if (strcmp(method, "par_ikj") == 0) {
        ThreadPool *pool = threadpool_create(threads, threads);
        Timer timer = timer_start();
        result_c = matrix_mult_ikj_parallel(pool, matrix_a, matrix_b);
        elapsed_sec = timer_stop(&timer);
        threadpool_destroy(pool);
    }

    double checksum = matrix_checksum(result_c);
    long peak_rss_kb = memory_peak_rss_kb();

    if (output_file != NULL && rep >= 0) {
        fprintf(output_file,
                "%s,%s,%zu,1,%zu,%zu,%d,%.9f,0.000000000,%.9f,%.9f,%.9f,0.000000000,%ld,%ld,%ld,0,%.4f\n",
                experiment, method, n, threads, threads, rep,
                elapsed_sec, elapsed_sec, elapsed_sec, elapsed_sec,
                peak_rss_kb, peak_rss_kb, peak_rss_kb, checksum);
        fflush(output_file);
    }

    const char *rep_label = (rep < 0) ? "WARMUP" : "REP";
    printf("  [%-10s] n=%-4zu P=1  threads=%-2zu %s=%-2d total=%.6fs rss=%-6ldKB\n",
           method, n, threads, rep_label, rep, elapsed_sec, peak_rss_kb);

    matrix_free(matrix_a);
    matrix_free(matrix_b);
    matrix_free(result_c);
}

void run_isolated_local_trial(FILE *output_file,
                              const char *experiment,
                              const char *method,
                              size_t n,
                              size_t threads,
                              int rep,
                              uint64_t seed) {
    pid_t pid = fork();
    if (pid == 0) {
        execute_local_trial(output_file, experiment, method, n, threads, rep, seed);
        exit(0);
    }
    waitpid(pid, NULL, 0);
}

static ClusterMemoryMetrics collect_cluster_memory(int rank, int num_processes, MPI_Comm communicator) {
    long local_rss_kb = memory_peak_rss_kb();
    long *all_rss_kb = NULL;

    if (rank == 0) {
        all_rss_kb = (long *)malloc((size_t)num_processes * sizeof(long));
    }

    MPI_Gather(&local_rss_kb, 1, MPI_LONG, all_rss_kb, 1, MPI_LONG, 0, communicator);

    ClusterMemoryMetrics memory_metrics = {0, 0, 0};
    if (rank == 0 && all_rss_kb != NULL) {
        memory_metrics.rank0_rss_kb = all_rss_kb[0];
        memory_metrics.worker_max_rss_kb = (num_processes > 1) ? all_rss_kb[1] : all_rss_kb[0];
        for (int proc = 0; proc < num_processes; proc++) {
            memory_metrics.total_rss_kb += all_rss_kb[proc];
            if (proc >= 1 && all_rss_kb[proc] > memory_metrics.worker_max_rss_kb) {
                memory_metrics.worker_max_rss_kb = all_rss_kb[proc];
            }
        }
        free(all_rss_kb);
    }

    return memory_metrics;
}

void run_mpi_trial(FILE *output_file,
                   const char *experiment,
                   const char *method,
                   size_t n,
                   size_t threads_per_proc,
                   int rep,
                   uint64_t seed,
                   MPI_Comm communicator) {
    int rank = 0;
    int num_processes = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &num_processes);

    Matrix *matrix_a = NULL;
    Matrix *matrix_b = NULL;

    if (rank == 0) {
        RNG rng;
        rng_seed(&rng, trial_seed(seed, rep));
        matrix_a = create_matrix(n, n);
        matrix_b = create_matrix(n, n);
        matrix_random(matrix_a, &rng, -2.0, 2.0);
        matrix_random(matrix_b, &rng, -2.0, 2.0);
    }

    ThreadPool *pool = (threads_per_proc > 1)
                       ? threadpool_create(threads_per_proc, threads_per_proc)
                       : NULL;

    MpiMatmulMetrics metrics = {0};
    Matrix *result_c = matrix_mult_ikj_mpi(matrix_a, matrix_b, pool, communicator, &metrics);

    if (pool != NULL) {
        threadpool_destroy(pool);
    }

    ClusterMemoryMetrics memory_metrics = collect_cluster_memory(rank, num_processes, communicator);

    if (rank == 0) {
        double checksum = matrix_checksum(result_c);
        size_t total_bytes = metrics.bytes_scattered_a +
                             metrics.bytes_broadcast_b +
                             metrics.bytes_gathered_c;
        size_t total_workers = (size_t)num_processes * threads_per_proc;

        if (output_file != NULL && rep >= 0) {
            fprintf(output_file,
                    "%s,%s,%zu,%d,%zu,%zu,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%ld,%ld,%ld,%zu,%.4f\n",
                    experiment,
                    method,
                    n,
                    num_processes,
                    threads_per_proc,
                    total_workers,
                    rep,
                    metrics.end_to_end_sec,
                    metrics.distribute_sec,
                    metrics.compute_sec,
                    metrics.compute_max_sec,
                    metrics.compute_avg_sec,
                    metrics.gather_sec,
                    memory_metrics.rank0_rss_kb,
                    memory_metrics.worker_max_rss_kb,
                    memory_metrics.total_rss_kb,
                    total_bytes,
                    checksum);
            fflush(output_file);
        }

        const char *rep_label = (rep < 0) ? "WARMUP" : "REP";
        printf("  [%-10s] n=%-4zu P=%-2d threads=%-2zu %s=%-2d total=%.6fs (dist=%.4fs comp=%.4fs gath=%.4fs) rss0=%-6ldKB\n",
               method,
               n,
               num_processes,
               threads_per_proc,
               rep_label,
               rep,
               metrics.end_to_end_sec,
               metrics.distribute_sec,
               metrics.compute_sec,
               metrics.gather_sec,
               memory_metrics.rank0_rss_kb);
    }

    matrix_free(matrix_a);
    matrix_free(matrix_b);
    matrix_free(result_c);
}
