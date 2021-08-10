#*****************************************************************************
# ccleartag.S
#-----------------------------------------------------------------------------
#
# Test ccleartag instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Check that tag bit is set
  TEST_CASE( 2, x2, 0x00000001, cgettag x2, c1; );

  # Clear the tag bit
  ccleartag c1, c1

  # Check that tag bit is clear
  TEST_CASE( 3, x2, 0x00000000, cgettag x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
