#*****************************************************************************
# candperm.S
#-----------------------------------------------------------------------------
#
# Test candperm instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Disable a few permissions
  li x2, 0x80a
  candperm c1, c1, x2

  # Check that correct permissions are available
  TEST_CASE( 2, x2, 0x0000080a, cgetperm x2, c1; );

  # Disable a permission we don't have
  li x2, 0x81a
  candperm c1, c1, x2

  # Check that correct permissions are available
  TEST_CASE( 3, x2, 0x0000080a, cgetperm x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
