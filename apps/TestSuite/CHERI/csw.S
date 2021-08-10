# See LICENSE for license details.

#*****************************************************************************
# csw.S
#-----------------------------------------------------------------------------
#
# Test csw instruction.
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

  TEST_CST_OP( 2, clw, csw, 0x0000000000aa00aa, 0,  tdat );
  TEST_CST_OP( 3, clw, csw, 0xffffffffaa00aa00, 4,  tdat );
  TEST_CST_OP( 4, clw, csw, 0x000000000aa00aa0, 8,  tdat );
  TEST_CST_OP( 5, clw, csw, 0xffffffffa00aa00a, 12, tdat );

  # Test with negative offset

  TEST_CST_OP( 6, clw, csw, 0x0000000000aa00aa, -12, tdat8 );
  TEST_CST_OP( 7, clw, csw, 0xffffffffaa00aa00, -8,  tdat8 );
  TEST_CST_OP( 8, clw, csw, 0x000000000aa00aa0, -4,  tdat8 );
  TEST_CST_OP( 9, clw, csw, 0xffffffffa00aa00a, 0,   tdat8 );

  # Test with a negative base

  TEST_CASE( 10, x3, 0x12345678, \
    cllc  c1, tdat9; \
    li  x2, 0x12345678; \
    cincoffset c4, c1, -32; \
    csw x2, 32(c4); \
    clw x3, 0(c1); \
  )

  # Test with unaligned base

  TEST_CASE( 11, x3, 0x58213098, \
    cllc  c1, tdat9; \
    li  x2, 0x58213098; \
    cincoffset c1, c1, -3; \
    csw x2, 7(c1); \
    cllc c4, tdat10; \
    clw x3, 0(c4); \
  )

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_CST_SRC12_BYPASS( 12, 0, 0, clw, csw, 0xffffffffaabbccdd, 0,  tdat );
  TEST_CST_SRC12_BYPASS( 13, 0, 1, clw, csw, 0xffffffffdaabbccd, 4,  tdat );
  TEST_CST_SRC12_BYPASS( 14, 0, 2, clw, csw, 0xffffffffddaabbcc, 8,  tdat );
  TEST_CST_SRC12_BYPASS( 15, 1, 0, clw, csw, 0xffffffffcddaabbc, 12, tdat );
  TEST_CST_SRC12_BYPASS( 16, 1, 1, clw, csw, 0xffffffffccddaabb, 16, tdat );
  TEST_CST_SRC12_BYPASS( 17, 2, 0, clw, csw, 0xffffffffbccddaab, 20, tdat );

  TEST_CST_SRC21_BYPASS( 18, 0, 0, clw, csw, 0x00112233, 0,  tdat );
  TEST_CST_SRC21_BYPASS( 19, 0, 1, clw, csw, 0x30011223, 4,  tdat );
  TEST_CST_SRC21_BYPASS( 20, 0, 2, clw, csw, 0x33001122, 8,  tdat );
  TEST_CST_SRC21_BYPASS( 21, 1, 0, clw, csw, 0x23300112, 12, tdat );
  TEST_CST_SRC21_BYPASS( 22, 1, 1, clw, csw, 0x22330011, 16, tdat );
  TEST_CST_SRC21_BYPASS( 23, 2, 0, clw, csw, 0x12233001, 20, tdat );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .word 0xdeadbeef
tdat2:  .word 0xdeadbeef
tdat3:  .word 0xdeadbeef
tdat4:  .word 0xdeadbeef
tdat5:  .word 0xdeadbeef
tdat6:  .word 0xdeadbeef
tdat7:  .word 0xdeadbeef
tdat8:  .word 0xdeadbeef
tdat9:  .word 0xdeadbeef
tdat10: .word 0xdeadbeef

RVTEST_DATA_END
