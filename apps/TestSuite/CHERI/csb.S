# See LICENSE for license details.

#*****************************************************************************
# csb.S
#-----------------------------------------------------------------------------
#
# Test csb instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32U
RVTEST_CODE_BEGIN

# Use only a single thread for this test
#ifdef _TEST_SIMT_
  csrrw t3, 0xf14, zero
  bnez t3, pass
#endif

  #-------------------------------------------------------------
  # Basic tests
  #-------------------------------------------------------------

  TEST_CST_OP( 2, clb, csb, 0xffffffffffffffaa, 0, tdat );
  TEST_CST_OP( 3, clb, csb, 0x0000000000000000, 1, tdat );
  TEST_CST_OP( 4, clh, csb, 0xffffffffffffefa0, 2, tdat );
  TEST_CST_OP( 5, clb, csb, 0x000000000000000a, 3, tdat );

  # Test with negative offset

  TEST_CST_OP( 6, clb, csb, 0xffffffffffffffaa, -3, tdat8 );
  TEST_CST_OP( 7, clb, csb, 0x0000000000000000, -2, tdat8 );
  TEST_CST_OP( 8, clb, csb, 0xffffffffffffffa0, -1, tdat8 );
  TEST_CST_OP( 9, clb, csb, 0x000000000000000a, 0,  tdat8 );

  # Test with a negative base

  TEST_CASE( 10, x3, 0x78, \
    cllc c1, tdat9; \
    li  x2, 0x12345678; \
    cincoffset c4, c1, -32; \
    csb x2, 32(c4); \
    clb x3, 0(c1); \
  )

  # Test with unaligned base

  TEST_CASE( 11, x3, 0xffffffffffffff98, \
    cllc c1, tdat9; \
    li  x2, 0x00003098; \
    cincoffset c1, c1, -6; \
    csb x2, 7(c1); \
    cllc c4, tdat10; \
    clb x3, 0(c4); \
  )

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_CST_SRC12_BYPASS( 12, 0, 0, clb, csb, 0xffffffffffffffdd, 0, tdat );
  TEST_CST_SRC12_BYPASS( 13, 0, 1, clb, csb, 0xffffffffffffffcd, 1, tdat );
  TEST_CST_SRC12_BYPASS( 14, 0, 2, clb, csb, 0xffffffffffffffcc, 2, tdat );
  TEST_CST_SRC12_BYPASS( 15, 1, 0, clb, csb, 0xffffffffffffffbc, 3, tdat );
  TEST_CST_SRC12_BYPASS( 16, 1, 1, clb, csb, 0xffffffffffffffbb, 4, tdat );
  TEST_CST_SRC12_BYPASS( 17, 2, 0, clb, csb, 0xffffffffffffffab, 5, tdat );

  TEST_CST_SRC21_BYPASS( 18, 0, 0, clb, csb, 0x33, 0, tdat );
  TEST_CST_SRC21_BYPASS( 19, 0, 1, clb, csb, 0x23, 1, tdat );
  TEST_CST_SRC21_BYPASS( 20, 0, 2, clb, csb, 0x22, 2, tdat );
  TEST_CST_SRC21_BYPASS( 21, 1, 0, clb, csb, 0x12, 3, tdat );
  TEST_CST_SRC21_BYPASS( 22, 1, 1, clb, csb, 0x11, 4, tdat );
  TEST_CST_SRC21_BYPASS( 23, 2, 0, clb, csb, 0x01, 5, tdat );

  li a0, 0xef
  cllc c1, tdat
  csb a0, 3(c1)

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .byte 0xef
tdat2:  .byte 0xef
tdat3:  .byte 0xef
tdat4:  .byte 0xef
tdat5:  .byte 0xef
tdat6:  .byte 0xef
tdat7:  .byte 0xef
tdat8:  .byte 0xef
tdat9:  .byte 0xef
tdat10: .byte 0xef

RVTEST_DATA_END
