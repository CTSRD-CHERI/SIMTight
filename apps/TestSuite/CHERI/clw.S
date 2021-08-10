# See LICENSE for license details.

#*****************************************************************************
# clw.S
#-----------------------------------------------------------------------------
#
# Test clw instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Basic tests
  #-------------------------------------------------------------

  TEST_CLD_OP( 2, clw, 0x0000000000ff00ff, 0,  tdat );
  TEST_CLD_OP( 3, clw, 0xffffffffff00ff00, 4,  tdat );
  TEST_CLD_OP( 4, clw, 0x000000000ff00ff0, 8,  tdat );
  TEST_CLD_OP( 5, clw, 0xfffffffff00ff00f, 12, tdat );

  # Test with negative offset

  TEST_CLD_OP( 6, clw, 0x0000000000ff00ff, -12, tdat4 );
  TEST_CLD_OP( 7, clw, 0xffffffffff00ff00, -8,  tdat4 );
  TEST_CLD_OP( 8, clw, 0x000000000ff00ff0, -4,  tdat4 );
  TEST_CLD_OP( 9, clw, 0xfffffffff00ff00f, 0,   tdat4 );

  # Test with a negative base

  TEST_CASE( 10, x3, 0x0000000000ff00ff, \
    cllc c1, tdat; \
    cincoffset c1, c1, -32; \
    clw x3, 32(c1); \
  )

  # Test with unaligned base

  TEST_CASE( 11, x3, 0xffffffffff00ff00, \
    cllc c1, tdat; \
    cincoffset c1, c1, -3; \
    clw x3, 7(c1); \
  )

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_CLD_DEST_BYPASS( 12, 0, clw, 0x000000000ff00ff0, 4, tdat2 );
  TEST_CLD_DEST_BYPASS( 13, 1, clw, 0xfffffffff00ff00f, 4, tdat3 );
  TEST_CLD_DEST_BYPASS( 14, 2, clw, 0xffffffffff00ff00, 4, tdat1 );

  TEST_CLD_SRC1_BYPASS( 15, 0, clw, 0x000000000ff00ff0, 4, tdat2 );
  TEST_CLD_SRC1_BYPASS( 16, 1, clw, 0xfffffffff00ff00f, 4, tdat3 );
  TEST_CLD_SRC1_BYPASS( 17, 2, clw, 0xffffffffff00ff00, 4, tdat1 );

  #-------------------------------------------------------------
  # Test write-after-write hazard
  #-------------------------------------------------------------

  TEST_CASE( 18, x2, 2, \
    cllc  c3, tdat; \
    clw  x2, 0(c3); \
    li  x2, 2; \
  )

  TEST_CASE( 19, x2, 2, \
    cllc  c3, tdat; \
    clw  x2, 0(c3); \
    nop; \
    li  x2, 2; \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .word 0x00ff00ff
tdat2:  .word 0xff00ff00
tdat3:  .word 0x0ff00ff0
tdat4:  .word 0xf00ff00f

RVTEST_DATA_END
