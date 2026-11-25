#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-all}"
HOSTFILE="${HOSTFILE:-$HOME/hosts.txt}"
THREADS_PER_NODE="${THREADS_PER_NODE:-2}"
CSV_OUTPUT="distributed_benchmark.csv"

echo "=========================================================="
echo "  Distributed Matrix Multiplication Pipeline (Assign. 4)  "
echo "=========================================================="

if [ "$MODE" = "worker" ]; then
    echo "[Worker Setup] Compiling project in performance mode (-O3 -march=native)..."
    make performance
    echo "===> Worker node is compiled and ready for Rank 0 mpirun calls."
    exit 0
fi

ensure_python_env() {
    if [ -d ".venv" ]; then
        PY_BIN=".venv/bin/python"
    elif [ -d "../BD_Assignment3/.venv" ]; then
        PY_BIN="../BD_Assignment3/.venv/bin/python"
    elif [ -d "../BD_Assignment2/.venv" ]; then
        PY_BIN="../BD_Assignment2/.venv/bin/python"
    else
        echo "Creating Python virtual environment in .venv..."
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt > /dev/null
        PY_BIN=".venv/bin/python"
    fi
}

if [ "$MODE" = "plots" ]; then
    ensure_python_env
    echo "[1/1] Generating figures from results/$CSV_OUTPUT..."
    "$PY_BIN" tools/plot_results.py "$CSV_OUTPUT"
    echo "===> Figures successfully generated in results/figures/"
    exit 0
fi

echo "[1/5] Running local and MPI unit tests..."
make clean > /dev/null
make run_tests

echo "[2/5] Compiling project in performance mode (-O3 -march=native)..."
make performance > /dev/null

MPI_HOST_ARGS=""
if [ -f "$HOSTFILE" ]; then
    echo "      Detected cluster hostfile: $HOSTFILE"
    MPI_HOST_ARGS="--hostfile $HOSTFILE --map-by node"
    WORKER_IPS=$(awk '{print $1}' "$HOSTFILE")
    for ip in $WORKER_IPS; do
        echo "      Syncing and compiling on node $ip..."
        ssh -o StrictHostKeyChecking=no "$ip" "cd $ROOT_DIR && make performance > /dev/null"
    done
else
    echo "      No $HOSTFILE found; running MPI processes locally (--oversubscribe)."
    MPI_HOST_ARGS="--oversubscribe"
fi

BENCH_FLAGS="--threads=$THREADS_PER_NODE --output=$CSV_OUTPUT"
if [ "$MODE" = "quick" ]; then
    BENCH_FLAGS="$BENCH_FLAGS --quick"
elif [ "$MODE" = "medium" ]; then
    BENCH_FLAGS="$BENCH_FLAGS --medium"
fi

mkdir -p results

echo "[3/5] Running Single-Node Baselines (seq_ikj and par_ikj) on Rank 0..."
./bin/benchmark --baseline $BENCH_FLAGS

echo "[4/5] Running Distributed MPI Strong & Weak Scaling (P = 1, 2, 3, 4)..."
for P in 1 2 3 4; do
    echo "----------------------------------------------------------"
    echo "  Executing MPI cluster run with P = $P nodes"
    echo "----------------------------------------------------------"
    mpirun $MPI_HOST_ARGS -np "$P" ./bin/benchmark --mpi --weak --append $BENCH_FLAGS
done

echo "[5/5] Checking Python environment for figure generation..."
if python3 -m venv --help > /dev/null 2>&1; then
    ensure_python_env
    "$PY_BIN" tools/plot_results.py "$CSV_OUTPUT"
else
    echo "      Python venv not installed on this node; skip plotting here."
    echo "      Download results/$CSV_OUTPUT to your laptop and run: ./run_pipeline.sh plots"
fi

echo "=========================================================="
echo "  Pipeline completed successfully!"
echo "  - Raw CSV results: results/$CSV_OUTPUT"
echo "  - Generated figures: results/figures/"
echo "=========================================================="
