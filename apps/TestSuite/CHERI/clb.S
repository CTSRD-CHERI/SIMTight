# See LICENSE for license details.

#*****************************************************************************
# clb.S
#-----------------------------------------------------------------------------
#
# Test clb instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Basic tests
  #-------------------------------------------------------------

  TEST_CLD_OP( 4, clb, 0xfffffffffffffff0, 2, tdat );
  TEST_CLD_OP( 5, clb, 0x000000000000000f, 3, tdat );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .byte 0xff
tdat2:  .byte 0x00
tdat3:  .byte 0xf0
tdat4:  .byte 0x0f

RVTEST_DATA_END
