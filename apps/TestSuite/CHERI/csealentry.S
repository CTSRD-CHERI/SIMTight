#*****************************************************************************
# csealentry.S
#-----------------------------------------------------------------------------
#
# Test csealentry instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Seal cap as sentry
  csealentry c1, c1

  # Check that cap is sealed
  TEST_CASE( 2, x2, 0xfffffffe, cgettype x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
