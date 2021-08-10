#ifndef _RAND_H_
#define _RAND_H_

#include <Pebbles/Common.h>

INLINE uint32_t rand15(uint32_t* seed) {
  *seed = (*seed * 1664525 + 1013904223) & 0x7fffffff;
  return *seed >> 16;
}

#endif
