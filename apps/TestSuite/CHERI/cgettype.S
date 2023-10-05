#*****************************************************************************
# cgettype.S
#-----------------------------------------------------------------------------
#
# Test cgettype instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Check that cap is unsealed
  TEST_CASE( 2, x2, 0xffffffff, cgettype x2, c1; );

  # Setup null cap
  add x1, x0, 0

  # Check that cap is unsealed
  TEST_CASE( 3, x2, 0xffffffff, cgettype x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
