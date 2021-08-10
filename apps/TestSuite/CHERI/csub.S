#*****************************************************************************
# csub.S
#-----------------------------------------------------------------------------
#
# Test cssub instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0
  cspecialrw c2, ddc, c0

  # Set address of c1
  li x3, 0x00008000
  csetaddr c1, c1, x3

  # Set address of c2
  li x3, 0x00009000
  csetaddr c2, c2, x3

  # Subtract caps
  TEST_CASE( 2, x3, 0x00001000, csub x3, c2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
