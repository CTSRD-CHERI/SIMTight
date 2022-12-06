#ifndef _FASTZERO_H_
#define _FASTZERO_H_

#include <Config.h>
#include <Pebbles/Common.h>
#include <Pebbles/UART/IO.h>
#include <Pebbles/Instrs/FastZeroing.h>
#include <Pebbles/CSRs/Sim.h>
#include <Pebbles/CSRs/UART.h>

inline void fastZero(void* base, uint32_t numBytes)
{
  assert((numBytes & (DRAMBeatBytes-1)) == 0,
      "fastZero: num bytes must be a multiple of beat size in bytes");
  uint32_t numBeats = numBytes >> DRAMBeatLogBytes;
  if (numBeats == 0) return;
  uint32_t beatAddr = (uint32_t) base;
  assert((beatAddr & (DRAMBeatBytes-1)) == 0,
      "fastZero: address must be a multiple of beat size in bytes");
  pebblesFastZero(beatAddr, numBeats-1);
  pebblesFence();
}

#endif
