#*****************************************************************************
# cincoffset.S
#-----------------------------------------------------------------------------
#
# Test cincoffset instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Set offset
  li x2, 0x00008000
  csetoffset c1, c1, x2

  # Increment offset using immediate
  cincoffsetimm c1, c1, 0x10

  # Check that address is correct
  TEST_CASE( 2, x2, 0x00008010, cgetoffset x2, c1; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
