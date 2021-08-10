#include <Config.h>
#include <MemoryMap.h>

.global _start
_start:

# Set stack pointer
#if EnableCHERI
  # TODO: Constrain the bounds/permssions of the stack pointer
  cspecialr csp, ddc
  li x1, STACK_BASE
  csetaddr csp, csp, x1
#else
  li sp, STACK_BASE
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
