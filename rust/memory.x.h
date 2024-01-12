#include <Config.h>
#include <MemoryMap.h>

// Size of instruction memory, excluding boot loader
#define IMEM_LENGTH ((4 << CPUInstrMemLogWords) - MaxBootImageBytes)

MEMORY
{
  IMEM    : ORIGIN = MemBase+MaxBootImageBytes, LENGTH = IMEM_LENGTH
  GLOBALS : ORIGIN = DMEM_BASE, LENGTH = 1 << 28
  STACK   : ORIGIN = STACK_BASE-CPUStackSize, LENGTH = CPUStackSize - 1024
}

REGION_ALIAS("REGION_TEXT", IMEM);
REGION_ALIAS("REGION_RODATA", GLOBALS);
REGION_ALIAS("REGION_DATA", GLOBALS);
REGION_ALIAS("REGION_BSS", GLOBALS);
REGION_ALIAS("REGION_HEAP", GLOBALS);
REGION_ALIAS("REGION_STACK", STACK);

_hart_stack_size = CPUStackSize - 2048;
_heap_size = 1 << 27;
