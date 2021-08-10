#*****************************************************************************
# cjalr.S
#-----------------------------------------------------------------------------
#
# Test cjalr instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

test_2:
  li  TESTNUM, 2

  # Load capability target
  cllc c1, target_2

  # Capability jump
  cjalr c2, c1
linkaddr_2:
  j fail

target_2:
  # Check link address
  cgetaddr x2, c2
  cllc c3, linkaddr_2
  cgetaddr x3, c3
  bne x2, x3, fail

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
