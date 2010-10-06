/*!
 * \file rng.h
 * \brief Simple random number generation for OpenCL.
 */

#ifndef RNG_H
#define RNG_H

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
static float frand(uint *seed0, uint *seed1) {
  /*
   * Advance the seeds for the next number in the sequence.
   * Based on Marsaglia Multiply-With-Carry as described on
   * http://en.wikipedia.org/wiki/Random_number_generation
   */
 *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
 *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);
 /*
  * Union to convert uint bit-string to float.
  */
 union {
  float fValue;
  uint bits;
 } converter;
 /*
  * The random number based on the seeds...
  */
 converter.bits = ((*seed0) << 16) + (*seed1);
 /*
  * Convert to a float in the range 0-1.
  */
 converter.bits = ((converter.bits) & 0x007fffff) | 0x40000000;
 return (converter.fValue - 2.0f) / 2.0f;
}


#endif /* RNG_H */
