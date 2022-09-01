#ifndef _MEMORY_MAP_H_
#define _MEMORY_MAP_H_

#include <Config.h>

// DRAM size in bytes
#define DRAM_SIZE \
  (1ull << (DRAMAddrWidth + DRAMBeatLogBytes))
#define DRAM_SIZE_LINK \
  (1 << (DRAMAddrWidth + DRAMBeatLogBytes))

// Sum of stack sizes for all SIMT threads
#define SIMT_STACKS_SIZE \
  (1 << (SIMTLogLanes + SIMTLogWarps + SIMTLogBytesPerStack))

// Sum of sizes of banked SRAMs
#define BANKED_SRAMS_SIZE \
  (1 << (SIMTLogLanes + SIMTLogWordsPerSRAMBank+2))

// Base of data memory (after instruction memory)
#define DMEM_BASE (MemBase + (4 << CPUInstrMemLogWords))

// SIMT local memory is toward the end of DRAM, before SIMT thread stacks
#define LOCAL_MEM_BASE (DRAM_SIZE - SIMT_STACKS_SIZE - BANKED_SRAMS_SIZE)
#define LOCAL_MEM_BASE_LINK \
  (DRAM_SIZE_LINK - SIMT_STACKS_SIZE - BANKED_SRAMS_SIZE)

// Base of CPU stack (growing down) is before SIMT local memory
#define STACK_BASE LOCAL_MEM_BASE_LINK

#endif
