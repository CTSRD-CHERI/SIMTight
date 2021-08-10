#include <Config.h>

OUTPUT_ARCH( "riscv" )

MEMORY
{
  /* Define max length of boot loader */
  boot : ORIGIN = MemBase, LENGTH = MaxBootImageBytes
}

SECTIONS
{
  /* Instruction memory */
  /* (No data sections allowed in boot loader) */
  .text : { *.o(.text*) } > boot
}
