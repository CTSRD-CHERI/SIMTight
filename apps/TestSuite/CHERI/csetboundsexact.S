#*****************************************************************************
# csetbounds.S
#-----------------------------------------------------------------------------
#
# Test csetbounds instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Set address
  li x2, 0x00008000
  csetaddr c1, c1, x2

  # Set bounds
  li x2, 0x00001000
  csetboundsexact c1, c1, x2

  # Check that the base is correct
  TEST_CASE( 2, x2, 0x00008000, cgetbase x2, c1; );

  # Check that the length is correct
  TEST_CASE( 3, x2, 0x00001000, cgetlen x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
