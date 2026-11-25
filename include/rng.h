#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint64_t state;
} RNG;

void rng_seed(RNG *rng, uint64_t seed);

uint64_t rng_next_u64(RNG *rng);
double rng_next_double(RNG *rng);
double rng_next_range(RNG *rng, double min, double max);

#endif
