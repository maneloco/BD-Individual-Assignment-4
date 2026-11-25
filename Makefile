CC ?= gcc
MPICC ?= mpicc
MPIRUN ?= mpirun
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2 -Iinclude
LDFLAGS ?= -lpthread -lm

SRC_DIR := src
INC_DIR := include
TEST_DIR := tests
BUILD_DIR := build
BIN_DIR := bin

MPI_SRC := $(SRC_DIR)/mpi_matmul.c
MPI_OBJ := $(BUILD_DIR)/mpi_matmul.o

CORE_SRCS := $(filter-out $(MPI_SRC), $(wildcard $(SRC_DIR)/*.c))
CORE_OBJS := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(CORE_SRCS))

MPI_TEST_SRC := $(TEST_DIR)/test_mpi.c
MPI_TEST_BIN := $(BIN_DIR)/test_mpi

LOCAL_TEST_SRCS := $(filter-out $(MPI_TEST_SRC), $(wildcard $(TEST_DIR)/*.c))
LOCAL_TEST_BINS := $(patsubst $(TEST_DIR)/%.c, $(BIN_DIR)/%, $(LOCAL_TEST_SRCS))

.PHONY: all clean tests run_tests run_mpi_tests help

all: tests

$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

$(MPI_OBJ): $(MPI_SRC) | $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(MPI_TEST_BIN): $(MPI_TEST_SRC) $(MPI_OBJ) $(CORE_OBJS) | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< $(MPI_OBJ) $(CORE_OBJS) -o $@ $(LDFLAGS)
	@echo "==> MPI test built successfully: $@"

$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.c $(CORE_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(CORE_OBJS) -o $@ $(LDFLAGS)
	@echo "==> Test built successfully: $@"

tests: $(LOCAL_TEST_BINS) $(MPI_TEST_BIN)

run_mpi_tests: $(MPI_TEST_BIN)
	@echo "\n=== EXECUTING MPI TEST SUITE ==="
	@for p in 1 2 3 4; do \
		echo "\n[MPI TEST] Executing $(MPI_TEST_BIN) with -np $$p..."; \
		$(MPIRUN) -np $$p $(MPI_TEST_BIN) || exit 1; \
	done
	@echo "\n===> All MPI tests passed successfully.\n"

run_tests: tests
	@echo "\n=== EXECUTING LOCAL TEST SUITE ==="
	@for t in $(LOCAL_TEST_BINS); do \
		echo "\n[TEST] Executing $$t..."; \
		$$t || exit 1; \
	done
	@$(MAKE) run_mpi_tests

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "==> Clean complete."

help:
	@echo "Available commands:"
	@echo "  make               - Compiles all unit tests"
	@echo "  make tests         - Compiles local and MPI unit tests"
	@echo "  make run_tests     - Runs all local and MPI unit tests (P = 1, 2, 3, 4)"
	@echo "  make run_mpi_tests - Runs MPI unit tests only (P = 1, 2, 3, 4)"
	@echo "  make clean         - Erases all objects and binaries"
