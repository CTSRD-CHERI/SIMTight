#*****************************************************************************
# csc.S
#-----------------------------------------------------------------------------
#
# Test csc instruction.
#

#include "riscv_test.h"
#include "test_macros.h"
#include <Config.h>

RVTEST_RV32U
RVTEST_CODE_BEGIN

  # Use only a single thread for this test
  #ifdef _TEST_SIMT_
    csrrw t3, 0xf14, zero
    bnez t3, pass
  #endif

  # Set stack pointer to test data
  cllc csp, tdat

  # Push stack pointer onto stack
  csc csp, 0(csp)
  # Read it back
  clc c5, 0(csp)

  # Check that tag bit is set
  TEST_CASE( 2, x3, 0x00000001, cgettag x3, c5; );

  # Check that address is correct
  li TESTNUM, 3
  cgetaddr x3, c5
  cgetaddr x4, csp
  bne x3, x4, fail;

  # Write to capability sub-word
  csw x0, 0(csp)
  # Read it back
  clc c5, 0(csp)

  # Check that tag bit is clear
  TEST_CASE( 4, x3, 0x00000000, cgettag x3, c5; );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .byte 0x0
tdat2:  .byte 0x0
tdat3:  .byte 0x0
tdat4:  .byte 0x0
tdat5:  .byte 0x0
tdat6:  .byte 0x0
tdat7:  .byte 0x0
tdat8:  .byte 0x0

RVTEST_DATA_END
