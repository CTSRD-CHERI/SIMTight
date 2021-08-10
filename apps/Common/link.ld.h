#include <Config.h>
#include <MemoryMap.h>

OUTPUT_ARCH( "riscv" )

// Size of instruction memory, excluding boot loader
#define IMEM_LENGTH ((4 << CPUInstrMemLogWords) - MaxBootImageBytes)

MEMORY
{
  /* Define max length of boot loader */
  instrs : ORIGIN = MemBase+MaxBootImageBytes, LENGTH = IMEM_LENGTH
  globals : ORIGIN = DMEM_BASE, LENGTH = 1 << 20
}

SECTIONS
{
  .text   : { *.o(.text*) }             > instrs
  .bss    : { *.o(.bss*) }              > globals = 0
  .rodata : { *.o(.rodata*) }           > globals
  .sdata  : { *.o(.sdata*) }            > globals
  .data   : { *.o(.data*) }             > globals
  .captable : { *.o(.captable*) }       > globals
  __cap_relocs : { *.o(__cap_relocs*) } > globals
  .eh_frame_hdr : ONLY_IF_RW { KEEP (*(.eh_frame_hdr))
                                     *(.eh_frame_hdr.*) } > globals
  .eh_frame : ONLY_IF_RW { KEEP (*(.eh_frame)) *(.eh_frame.*) } > globals
}
