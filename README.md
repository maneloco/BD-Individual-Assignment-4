# Distributed Matrix Multiplication (Assignment 4)
**Big Data**  
*Grado en Ciencia e Ingeniería de Datos — Universidad de Las Palmas de Gran Canaria (ULPGC)*  
**Author:** Manuel Muñoz Roca  

---

## 1. Project Overview

This project implements and evaluates distributed matrix multiplication in C using **OpenMPI** (`mpi_matmul.c`) across a 4-node Google Cloud cluster. It also combines OpenMPI with my POSIX thread pool from Assignment 3 (`mpi_hybrid`) to use both multiple machines and multiple threads per machine, and compares them against the sequential (`seq_ikj`) and shared-memory parallel (`par_ikj`) baselines.

### Implementations Compared
- **Sequential Baseline (`seq_ikj`):** Loop-reordered `ikj` multiplication from Assignment 2 (1 node, 1 thread).
- **Shared-Memory Parallel (`par_ikj`):** POSIX thread pool row-slicing from Assignment 3 (1 node, 2 threads).
- **Pure MPI (`mpi_ikj`):** 1D row-partitioned distributed multiplication using `MPI_Scatterv`, `MPI_Bcast`, and `MPI_Gatherv` (1 to 4 nodes, 1 thread per node).
- **Hybrid MPI + Pthreads (`mpi_hybrid`):** Same MPI row distribution across 1 to 4 nodes, using 2 POSIX threads inside each node.

---

## 2. Cluster Environment (Google Cloud Platform)

All tests were executed on a 4-node cluster in Google Cloud (`europe-southwest1-a`):

| Component | Specification |
| :--- | :--- |
| **Cluster Nodes** | 4 $\times$ GCP `e2-highcpu-2` VMs (`mpi-node-0` to `mpi-node-3`) |
| **CPU (per Node)** | Intel(R) Xeon(R) CPU @ 2.20 GHz (1 physical core, 2 threads) |
| **Total Cluster Compute** | 4 physical cores, 8 vCPUs across 4 machines |
| **Caches (per Node)** | L1d: 32 KiB, L2: 256 KiB, L3: 55 MiB |
| **Memory** | 1.9 GiB RAM per node (7.6 GiB total, 0 swap) |
| **OS & Kernel** | Ubuntu 24.04.5 LTS (Kernel `7.0.0-1017-gcp x86_64`) |
| **Compiler & MPI** | GCC 13.3.0 (`-std=c11 -O3 -march=native`), OpenMPI 4.1.6 |
| **Python** | Python 3.12+ (`numpy`, `pandas`, `matplotlib`) |

---

## 3. Project Structure

```text
BD_Assignment4/
├── Makefile                    # Build rules for gcc and mpicc
├── README.md                   # Project documentation and instructions
├── requirements.txt            # Python libraries for plotting
├── run_pipeline.sh             # Automated test, benchmark, and plot script
├── report.tex                  # LaTeX report source
├── report.pdf                  # Compiled PDF report
├── benchmarks/
│   ├── benchmark.c             # Baselines, strong scaling, and weak scaling runner
│   ├── benchmark_harness.c     # Timers, RSS tracking, and CSV output
│   └── benchmark_harness.h
├── include/
│   ├── matmul.h                # Sequential ikj baseline
│   ├── matrix.h                # Matrix struct and helpers
│   ├── mpi_matmul.h            # Distributed MPI multiplication
│   ├── parallel_matmul.h       # Shared-memory pthreads multiplication
│   ├── rng.h                   # Deterministic 64-bit LCG random generator
│   └── thread.h                # POSIX thread pool
├── src/
│   ├── matmul.c
│   ├── matrix.c
│   ├── mpi_matmul.c
│   ├── parallel_matmul.c
│   ├── rng.c
│   └── thread.c
├── tests/
│   ├── test_matmul.c
│   ├── test_matrix.c
│   ├── test_mpi.c              # MPI correctness tests (P = 1, 2, 3, 4)
│   ├── test_parallel.c
│   └── test_rng.c
├── tools/
│   └── plot_results.py         # Generates the 6 figures from CSV
└── results/
    ├── distributed_benchmark.csv
    ├── environment_info.txt
    └── figures/
        ├── strong_scaling_speedup.png
        ├── strong_scaling_efficiency.png
        ├── execution_time_vs_n.png
        ├── communication_breakdown.png
        ├── memory_scaling.png
        └── weak_scaling.png
```

---

## 4. How to Build and Run

### Build Commands
```bash
make               # Compiles bin/benchmark
make performance   # Compiles bin/benchmark with -O3 -march=native
make run_tests     # Runs all local and MPI unit tests
make clean         # Removes build/ and bin/
```
*(Note: `-ffast-math` is omitted on purpose to keep strict IEEE 754 floating-point results and identical checksums across all methods.)*

### Running on a Multi-Node Cluster
1. **On each worker VM (`mpi-node-1..3`):**
   ```bash
   ./run_pipeline.sh worker
   ```
2. **On the coordinator VM (`mpi-node-0`, with `~/hosts.txt` listing the node IPs):**
   ```bash
   MPIRUN="mpirun --oversubscribe" ./run_pipeline.sh
   ```

### Other Useful Modes
```bash
./run_pipeline.sh plots    # Regenerate figures from results/distributed_benchmark.csv
./run_pipeline.sh quick    # Fast test run (n <= 512)
./run_pipeline.sh medium   # Medium run (n <= 2048)
```

### Compiling the Report
```bash
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

---

## 5. Main Results

1. **Network Overhead on Small Matrices ($n \le 256$):** Sending data over the network (`Scatterv`, `Bcast`, `Gatherv`) takes 70%–81% of the total time for small matrices, making 4 nodes slower than 1 sequential core ($0.35\times$ at $n = 128$, breaking even at $n = 256$ with $0.98\times$).
2. **Strong Scaling on Large Matrices ($n = 4096$):** At $n = 4096$, computation represents 90.3% of total time. Pure MPI (`mpi_ikj`) across 4 nodes achieves a **$3.50\times$ speedup** ($87.5\%$ efficiency), dropping runtime from 58.77s to 16.80s.
3. **Hybrid MPI + Pthreads:** Using 4 nodes with 2 threads per node (`mpi_hybrid`, 8 vCPUs total) reaches a **$4.89\times$ speedup** at $n = 4096$, reducing runtime to **12.02s** (compared to 58.77s sequential and 40.26s for 1-node `par_ikj`).
4. **Memory Usage:** At $n = 4096$ with 4 nodes, worker node memory drops to **207.9 MB** (vs. 384.8 MB on a single node), while Rank 0 uses **464.6 MB** and total cluster memory is **1088.0 MB** because matrix $B$ is broadcast to every node.
5. **Weak Scaling:** Keeping work per node constant ($n = 1024, 1290, 1477, 1625$ for $P = 1, 2, 3, 4$) keeps local computation time stable (~0.46s–0.59s in hybrid mode), while total time rises slowly from 0.48s to 0.72s due to larger network transfers (up to 95 MB).
