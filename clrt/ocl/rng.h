/*!
 * \file rng.h
 * \brief Simple random number generation for OpenCL.
 */

#ifndef RNG_H
#define RNG_H

typedef uint2 seed_value_t;
/*!
 * \brief Floating-point random number generation.  
 *
 * This implementation is not a particularly high quality PRNG,
 * but it is fast and works well enough when strong randomness is not necessary.
 *
 * Based on code from SmallPTGPU by Tom Flanagan (http://github.com/Knio/SmallptGPU).
 *
 * \param seed0 32 bits of seed data, advanced to the next value in the random sequence with each call to this function
 * \param seed1 32 bits of seed data, advanced to the next value in the random sequence with each call to this function
 */
float frand(seed_value_t *seed) {
    /*
     * Advance the seeds for the next number in the sequence.
     * Based on Marsaglia Multiply-With-Carry as described on
     * http://en.wikipedia.org/wiki/Random_number_generation
     */
    seed->x = 36969 * ((seed->x) & 65535) + ((seed->x) >> 16);
    seed->y = 18000 * ((seed->y) & 65535) + ((seed->y) >> 16);

    /*
     * The random number based on the seeds...
     */
    uint bits = ((seed->x) << 16) + (seed->y);
    /*
     * Convert to a float in the range 0-1.
     */
    bits = (bits & 0x007fffff) | 0x40000000;
    return (as_float(bits) - 2.0f) / 2.0f;
}

#endif /* RNG_H */
