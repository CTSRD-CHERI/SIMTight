# See LICENSE for license details.

#*****************************************************************************
# clh.S
#-----------------------------------------------------------------------------
#
# Test clh instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Basic tests
  #-------------------------------------------------------------

  TEST_CLD_OP( 2, clh, 0x00000000000000ff, 0, tdat );
  TEST_CLD_OP( 3, clh, 0xffffffffffffff00, 2, tdat );
  TEST_CLD_OP( 4, clh, 0x0000000000000ff0, 4, tdat );
  TEST_CLD_OP( 5, clh, 0xfffffffffffff00f, 6, tdat );

  # Test with negative offset

  TEST_CLD_OP( 6, clh, 0x00000000000000ff, -6, tdat4 );
  TEST_CLD_OP( 7, clh, 0xffffffffffffff00, -4, tdat4 );
  TEST_CLD_OP( 8, clh, 0x0000000000000ff0, -2, tdat4 );
  TEST_CLD_OP( 9, clh, 0xfffffffffffff00f,  0, tdat4 );

  # Test with a negative base

  TEST_CASE( 10, x3, 0x00000000000000ff, \
    cllc c1, tdat; \
    cincoffset c1, c1, -32; \
    clh x3, 32(c1); \
  )

  # Test with unaligned base

  TEST_CASE( 11, x3, 0xffffffffffffff00, \
    cllc c1, tdat; \
    cincoffset c1, c1, -5; \
    clh x3, 7(c1); \
  )

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_CLD_DEST_BYPASS( 12, 0, clh, 0x0000000000000ff0, 2, tdat2 );
  TEST_CLD_DEST_BYPASS( 13, 1, clh, 0xfffffffffffff00f, 2, tdat3 );
  TEST_CLD_DEST_BYPASS( 14, 2, clh, 0xffffffffffffff00, 2, tdat1 );

  TEST_CLD_SRC1_BYPASS( 15, 0, clh, 0x0000000000000ff0, 2, tdat2 );
  TEST_CLD_SRC1_BYPASS( 16, 1, clh, 0xfffffffffffff00f, 2, tdat3 );
  TEST_CLD_SRC1_BYPASS( 17, 2, clh, 0xffffffffffffff00, 2, tdat1 );

  #-------------------------------------------------------------
  # Test write-after-write hazard
  #-------------------------------------------------------------

  TEST_CASE( 18, x2, 2, \
    cllc c3, tdat; \
    clh  x2, 0(c3); \
    li  x2, 2; \
  )

  TEST_CASE( 19, x2, 2, \
    cllc c3, tdat; \
    clh  x2, 0(c3); \
    nop; \
    li  x2, 2; \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .half 0x00ff
tdat2:  .half 0xff00
tdat3:  .half 0x0ff0
tdat4:  .half 0xf00f

RVTEST_DATA_END
