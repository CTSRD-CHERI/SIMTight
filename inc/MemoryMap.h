#ifndef _MEMORY_MAP_H_
#define _MEMORY_MAP_H_

#include <Config.h>

NOTE("DRAM size in bytes")
#define DRAM_SIZE \
  (1ull << (DRAMAddrWidth + DRAMBeatLogBytes))
#define DRAM_SIZE_LINK \
  (1 << (DRAMAddrWidth + DRAMBeatLogBytes))

NOTE("Base of data memory (after instruction memory)")
#define DMEM_BASE (MemBase + (4 << CPUInstrMemLogWords))

NOTE("Sum of stack sizes for all SIMT threads")
NOTE("These stacks appear at the end of DRAM")
#define SIMT_STACKS_SIZE \
  (1 << (SIMTLogLanes + SIMTLogWarps + SIMTLogBytesPerStack))

NOTE("Sum of sizes of banked SRAMs")
#define BANKED_SRAMS_SIZE \
  (1 << (SIMTLogSRAMBanks + SIMTLogWordsPerSRAMBank+2))

NOTE("SIMT local memory is toward the end of DRAM, before the SIMT stacks")
#define LOCAL_MEM_BASE (DRAM_SIZE - SIMT_STACKS_SIZE - BANKED_SRAMS_SIZE)
#define LOCAL_MEM_BASE_LINK \
  (DRAM_SIZE_LINK - SIMT_STACKS_SIZE - BANKED_SRAMS_SIZE)

NOTE("Size of SIMT register spill region")
NOTE("We have space for int & cap meta-data reg files here")
#define REG_SPILL_SIZE (2 << (SIMTLogLanes + SIMTLogWarps + 5 + 2))

NOTE("The SIMT register spill region appears before the SRAM banks")
#define REG_SPILL_BASE (LOCAL_MEM_BASE_LINK - REG_SPILL_SIZE)

NOTE("Base of CPU stack (growing down) is before SIMT reg spill region")
#define STACK_BASE REG_SPILL_BASE

#endif
