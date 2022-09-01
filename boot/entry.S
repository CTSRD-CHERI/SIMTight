#include <Config.h>
#include <MemoryMap.h>

.global _start
_start:

# Set stack pointer
#if EnableCHERI
  # Constrain the bounds/permssions of the stack pointer
  cspecialr csp, ddc
  li x1, STACK_BASE - CPUStackSize
  csetaddr csp, csp, x1
  li x1, CPUStackSize
  csetbounds csp, csp, x1
  li x1, STACK_BASE-8
  csetaddr csp, csp, x1
#else
  li sp, STACK_BASE-8
#endif

# Jump to main
#if EnableCHERI
  cllc c1, main
  cjalr cra, c1
#else
  jal main
#endif

# Terminate simulator
csrw 0x801, zero

# Loop forever
j .
