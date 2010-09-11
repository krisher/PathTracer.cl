
#ifndef TIMING_H
#define TIMING_H

#include <time.h>

inline double timeElapsed(clock_t const since)
{
  return (clock()-since) / (double)CLOCKS_PER_SEC;
} 

#endif /* TIMING_H */
