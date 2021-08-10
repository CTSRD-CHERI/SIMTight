#*****************************************************************************
# cgetperm.S
#-----------------------------------------------------------------------------
#
# Test cgetperm instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Check that all permissions are available
  TEST_CASE( 2, x2, 0x00000fff, cgetperm x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
