#include "rng.h"

void rng_seed(RNG *rng, uint64_t seed) {
    if (seed == 0) {
        rng->state = 42;
        return;
    }
    rng->state = seed;
}

uint64_t rng_next_u64(RNG *rng) {
    uint64_t A = 6364136223846793005ULL;
    uint64_t B = 1442695040888963407ULL;

    rng->state = rng->state * A + B;
    return rng->state;
}

double rng_next_double(RNG *rng) {
    uint64_t val53 = rng_next_u64(rng) >> 11;
    return (double)val53 / 9007199254740992.0;
}

double rng_next_range(RNG *rng, double min, double max) {
    return min + rng_next_double(rng) * (max - min);
}
