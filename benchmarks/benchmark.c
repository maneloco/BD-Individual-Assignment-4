#include "benchmark_harness.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

static const size_t BENCHMARK_SIZES[] = {
    64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
};
static const size_t NUM_BENCHMARK_SIZES = sizeof(BENCHMARK_SIZES) / sizeof(BENCHMARK_SIZES[0]);

static void run_local_baselines(FILE *csv_file,
                                int repetitions,
                                uint64_t base_seed,
                                int max_size,
                                size_t hybrid_threads) {
    printf("\n=== Running Local Baselines (seq_ikj and par_ikj) ===\n");

    for (size_t s = 0; s < NUM_BENCHMARK_SIZES; s++) {
        size_t n = BENCHMARK_SIZES[s];
        if (max_size > 0 && (int)n > max_size) {
            break;
        }

        printf(">>> BASELINE MATRIX SIZE n = %zu <<<\n", n);

        run_isolated_local_trial(csv_file, "baseline", "seq_ikj", n, 1, -1, base_seed);
        for (int rep = 0; rep < repetitions; rep++) {
            run_isolated_local_trial(csv_file, "baseline", "seq_ikj", n, 1, rep, base_seed);
        }

        run_isolated_local_trial(csv_file, "baseline", "par_ikj", n, 1, -1, base_seed);
        for (int rep = 0; rep < repetitions; rep++) {
            run_isolated_local_trial(csv_file, "baseline", "par_ikj", n, 1, rep, base_seed);
        }

        if (hybrid_threads > 1) {
            run_isolated_local_trial(csv_file, "baseline", "par_ikj", n, hybrid_threads, -1, base_seed);
            for (int rep = 0; rep < repetitions; rep++) {
                run_isolated_local_trial(csv_file, "baseline", "par_ikj", n, hybrid_threads, rep, base_seed);
            }
        }
    }
}

static void run_mpi_strong_scaling(FILE *csv_file,
                                   int repetitions,
                                   uint64_t base_seed,
                                   int max_size,
                                   size_t hybrid_threads,
                                   int rank,
                                   int num_processes) {
    if (rank == 0) {
        printf("\n=== Running MPI Strong Scaling (P = %d processes) ===\n", num_processes);
    }

    for (size_t s = 0; s < NUM_BENCHMARK_SIZES; s++) {
        size_t n = BENCHMARK_SIZES[s];
        if (max_size > 0 && (int)n > max_size) {
            break;
        }

        if (rank == 0) {
            printf(">>> MPI STRONG SCALING n = %zu (P = %d) <<<\n", n, num_processes);
        }

        run_mpi_trial(csv_file, "strong_scaling", "mpi_ikj", n, 1, -1, base_seed, MPI_COMM_WORLD);
        for (int rep = 0; rep < repetitions; rep++) {
            run_mpi_trial(csv_file, "strong_scaling", "mpi_ikj", n, 1, rep, base_seed, MPI_COMM_WORLD);
        }

        if (hybrid_threads > 1) {
            run_mpi_trial(csv_file, "strong_scaling", "mpi_hybrid", n, hybrid_threads, -1, base_seed, MPI_COMM_WORLD);
            for (int rep = 0; rep < repetitions; rep++) {
                run_mpi_trial(csv_file, "strong_scaling", "mpi_hybrid", n, hybrid_threads, rep, base_seed, MPI_COMM_WORLD);
            }
        }
    }
}

static size_t compute_weak_scaling_size(size_t base_n, int num_processes) {
    double scaled = (double)base_n * cbrt((double)num_processes);
    return (size_t)llround(scaled);
}

static void run_mpi_weak_scaling(FILE *csv_file,
                                 int repetitions,
                                 uint64_t base_seed,
                                 size_t base_n,
                                 size_t hybrid_threads,
                                 int rank,
                                 int num_processes) {
    size_t scaled_n = compute_weak_scaling_size(base_n, num_processes);

    if (rank == 0) {
        printf("\n=== Running MPI Weak Scaling (P = %d, base_n = %zu -> n = %zu) ===\n",
               num_processes, base_n, scaled_n);
    }

    run_mpi_trial(csv_file, "weak_scaling", "mpi_ikj", scaled_n, 1, -1, base_seed, MPI_COMM_WORLD);
    for (int rep = 0; rep < repetitions; rep++) {
        run_mpi_trial(csv_file, "weak_scaling", "mpi_ikj", scaled_n, 1, rep, base_seed, MPI_COMM_WORLD);
    }

    if (hybrid_threads > 1) {
        run_mpi_trial(csv_file, "weak_scaling", "mpi_hybrid", scaled_n, hybrid_threads, -1, base_seed, MPI_COMM_WORLD);
        for (int rep = 0; rep < repetitions; rep++) {
            run_mpi_trial(csv_file, "weak_scaling", "mpi_hybrid", scaled_n, hybrid_threads, rep, base_seed, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv) {
    const char *output_filename = "distributed_benchmark.csv";
    int repetitions = 5;
    uint64_t seed = 42;
    int max_size = 4096;
    size_t hybrid_threads = 2;
    size_t weak_base_n = 1024;
    bool run_baseline = false;
    bool run_mpi = false;
    bool run_weak = false;
    bool append_csv = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--baseline") == 0) {
            run_baseline = true;
        } else if (strcmp(argv[i], "--mpi") == 0) {
            run_mpi = true;
        } else if (strcmp(argv[i], "--weak") == 0) {
            run_weak = true;
        } else if (strcmp(argv[i], "--append") == 0) {
            append_csv = true;
        } else if (strcmp(argv[i], "--quick") == 0 || strcmp(argv[i], "quick") == 0) {
            max_size = 512;
            weak_base_n = 256;
            repetitions = 3;
        } else if (strcmp(argv[i], "--medium") == 0 || strcmp(argv[i], "medium") == 0) {
            max_size = 2048;
            weak_base_n = 512;
        } else if (strncmp(argv[i], "--threads=", 10) == 0) {
            hybrid_threads = (size_t)atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--output=", 9) == 0) {
            output_filename = argv[i] + 9;
        } else if (argv[i][0] != '-') {
            output_filename = argv[i];
        }
    }

    if (!run_baseline && !run_mpi && !run_weak) {
        run_mpi = true;
        run_weak = true;
    }

    if (run_baseline && !run_mpi && !run_weak) {
        FILE *csv_file = benchmark_open_csv("results", output_filename, append_csv);
        if (csv_file == NULL) {
            return 1;
        }
        run_local_baselines(csv_file, repetitions, seed, max_size, hybrid_threads);
        benchmark_close_csv(csv_file);
        return 0;
    }

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int num_processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    FILE *csv_file = NULL;
    if (rank == 0) {
        csv_file = benchmark_open_csv("results", output_filename, append_csv);
        if (csv_file == NULL) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    if (run_mpi) {
        run_mpi_strong_scaling(csv_file, repetitions, seed, max_size, hybrid_threads, rank, num_processes);
    }

    if (run_weak) {
        run_mpi_weak_scaling(csv_file, repetitions, seed, weak_base_n, hybrid_threads, rank, num_processes);
    }

    if (rank == 0) {
        benchmark_close_csv(csv_file);
    }

    MPI_Finalize();
    return 0;
}
