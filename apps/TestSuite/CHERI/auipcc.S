# See LICENSE for license details.

#*****************************************************************************
# auipcc.S
#-----------------------------------------------------------------------------
#
# Test auipcc instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  TEST_CASE(2, a0, 4096, \
    foo: auipcc ca0, 1; \
    cllc ca1, foo; \
    csub a0, ca0, ca1; \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
