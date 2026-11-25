#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "rng.h"

void test_determinism(void) {
    RNG rng1, rng2;
    rng_seed(&rng1, 12345);
    rng_seed(&rng2, 12345);

    for (int i = 0; i < 100; i++) {
        uint64_t a = rng_next_u64(&rng1);
        uint64_t b = rng_next_u64(&rng2);
        assert(a == b);

        double c = rng_next_double(&rng1);
        double d = rng_next_double(&rng2);
        assert(c == d);
    }

    printf("  [OK] test_determinism\n");
}

void test_different_seeds(void) {
    RNG rng1, rng2;
    rng_seed(&rng1, 1);
    rng_seed(&rng2, 2);

    int differences = 0;
    for (int i = 0; i < 100; i++) {
        uint64_t a = rng_next_u64(&rng1);
        uint64_t b = rng_next_u64(&rng2);
        if (a != b) {
            differences++;
        }
    }
    assert(differences == 100);

    printf("  [OK] test_different_seeds\n");
}

void test_zero_seed(void) {
    RNG zero, forty_two;
    rng_seed(&zero, 0);
    rng_seed(&forty_two, 42);

    assert(zero.state == 42);
    for (int i = 0; i < 10; i++) {
        uint64_t a = rng_next_u64(&zero);
        uint64_t b = rng_next_u64(&forty_two);
        assert(a == b);
    }

    printf("  [OK] test_zero_seed\n");
}

void test_known_sequence(void) {
    RNG rng;
    rng_seed(&rng, 7);

    uint64_t state = 7;
    for (int i = 0; i < 50; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint64_t value = rng_next_u64(&rng);
        assert(value == state);
    }

    printf("  [OK] test_known_sequence\n");
}

void test_double_limits(void) {
    RNG rng;
    rng_seed(&rng, 2024);

    double sum = 0.0;
    int iterations = 100000;
    for (int i = 0; i < iterations; i++) {
        double value = rng_next_double(&rng);
        assert(value >= 0.0 && value < 1.0);
        sum += value;
    }

    double mean = sum / iterations;
    assert(fabs(mean - 0.5) < 0.01);

    printf("  [OK] test_double_limits (mean: %.4f)\n", mean);
}

void test_range_limits(void) {
    RNG rng;
    rng_seed(&rng, 42);

    double min = -5.0;
    double max = 10.0;

    for (int i = 0; i < 10000; i++) {
        double val = rng_next_range(&rng, min, max);
        assert(val >= min && val <= max);
    }

    printf("  [OK] test_range_limits\n");
}

void test_range_degenerate(void) {
    RNG rng;
    rng_seed(&rng, 5);

    for (int i = 0; i < 100; i++) {
        double val = rng_next_range(&rng, 3.5, 3.5);
        assert(val == 3.5);
    }

    printf("  [OK] test_range_degenerate\n");
}

int main(void) {
    printf("=== Running PRNG tests ===\n");
    test_determinism();
    test_different_seeds();
    test_zero_seed();
    test_known_sequence();
    test_double_limits();
    test_range_limits();
    test_range_degenerate();
    printf("=== All PRNG tests passed ===\n\n");
    
    return 0;
}
