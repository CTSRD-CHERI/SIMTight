#*****************************************************************************
# csetflags.S
#-----------------------------------------------------------------------------
#
# Test csetflags instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Set a flag
  li x2, 1
  csetflags c1, c1, x2

  # Check that the flag is set
  TEST_CASE( 2, x2, 0x00000001, cgetflags x2, c1; );

  # Clear flag
  csetflags c1, c1, x0

  # Check that flag is clear
  TEST_CASE( 3, x2, 0, cgetflags x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
