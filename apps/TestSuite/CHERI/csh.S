# See LICENSE for license details.

#*****************************************************************************
# csh.S
#-----------------------------------------------------------------------------
#
# Test csh instruction.
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

  TEST_CST_OP( 2, clh, csh, 0x00000000000000aa, 0, tdat );
  TEST_CST_OP( 3, clh, csh, 0xffffffffffffaa00, 2, tdat );
  TEST_CST_OP( 4, clw, csh, 0xffffffffbeef0aa0, 4, tdat );
  TEST_CST_OP( 5, clh, csh, 0xffffffffffffa00a, 6, tdat );

  # Test with negative offset

  TEST_CST_OP( 6, clh, csh, 0x00000000000000aa, -6, tdat8 );
  TEST_CST_OP( 7, clh, csh, 0xffffffffffffaa00, -4, tdat8 );
  TEST_CST_OP( 8, clh, csh, 0x0000000000000aa0, -2, tdat8 );
  TEST_CST_OP( 9, clh, csh, 0xffffffffffffa00a, 0,  tdat8 );

  # Test with a negative base

  TEST_CASE( 10, x3, 0x5678, \
    cllc  c1, tdat9; \
    li  x2, 0x12345678; \
    cincoffset c4, c1, -32; \
    csh x2, 32(c4); \
    clh x3, 0(c1); \
  )

  # Test with unaligned base

  TEST_CASE( 11, x3, 0x3098, \
    cllc  c1, tdat9; \
    li  x2, 0x00003098; \
    cincoffset c1, c1, -5; \
    csh x2, 7(c1); \
    cllc c4, tdat10; \
    clh x3, 0(c4); \
  )

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_CST_SRC12_BYPASS( 12, 0, 0, clh, csh, 0xffffffffffffccdd, 0,  tdat );
  TEST_CST_SRC12_BYPASS( 13, 0, 1, clh, csh, 0xffffffffffffbccd, 2,  tdat );
  TEST_CST_SRC12_BYPASS( 14, 0, 2, clh, csh, 0xffffffffffffbbcc, 4,  tdat );
  TEST_CST_SRC12_BYPASS( 15, 1, 0, clh, csh, 0xffffffffffffabbc, 6, tdat );
  TEST_CST_SRC12_BYPASS( 16, 1, 1, clh, csh, 0xffffffffffffaabb, 8, tdat );
  TEST_CST_SRC12_BYPASS( 17, 2, 0, clh, csh, 0xffffffffffffdaab, 10, tdat );

  TEST_CST_SRC21_BYPASS( 18, 0, 0, clh, csh, 0x2233, 0,  tdat );
  TEST_CST_SRC21_BYPASS( 19, 0, 1, clh, csh, 0x1223, 2,  tdat );
  TEST_CST_SRC21_BYPASS( 20, 0, 2, clh, csh, 0x1122, 4,  tdat );
  TEST_CST_SRC21_BYPASS( 21, 1, 0, clh, csh, 0x0112, 6, tdat );
  TEST_CST_SRC21_BYPASS( 22, 1, 1, clh, csh, 0x0011, 8, tdat );
  TEST_CST_SRC21_BYPASS( 23, 2, 0, clh, csh, 0x3001, 10, tdat );

  li a0, 0xbeef
  cllc c1, tdat
  csh a0, 6(c1)

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
tdat1:  .half 0xbeef
tdat2:  .half 0xbeef
tdat3:  .half 0xbeef
tdat4:  .half 0xbeef
tdat5:  .half 0xbeef
tdat6:  .half 0xbeef
tdat7:  .half 0xbeef
tdat8:  .half 0xbeef
tdat9:  .half 0xbeef
tdat10: .half 0xbeef

RVTEST_DATA_END
