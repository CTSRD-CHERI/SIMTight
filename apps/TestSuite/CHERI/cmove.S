#*****************************************************************************
# cmove.S
#-----------------------------------------------------------------------------
#
# Test cmove instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Read almighty cap
  cspecialrw c1, ddc, c0

  # Initialise c2
  cmove c2, c0

  # Check that c2 is not tagged
  TEST_CASE( 2, x3, 0x00000000, cgettag x3, c2; );

  # Move cap
  cmove c2, c1

  # Check that c2 is tagged
  TEST_CASE( 3, x3, 0x00000001, cgettag x3, c2; );

  # Move cap
  cmove c2, c0

  # Check that c2 is not tagged
  TEST_CASE( 4, x3, 0x00000000, cgettag x3, c2; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
