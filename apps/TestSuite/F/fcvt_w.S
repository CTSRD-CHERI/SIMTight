# See LICENSE.testsuite for license details

#*****************************************************************************
# fcvt_w.S
#-----------------------------------------------------------------------------
#
# Test fcvt{wu|w|lu|l}.s instructions.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32UF
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Arithmetic tests
  #-------------------------------------------------------------

  # Flags not checked

  TEST_FP_INT_OP_S_ZFINX( 2,  fcvt.w.s, 0x00,         -1, -1.1, rtz);
  TEST_FP_INT_OP_S_ZFINX( 3,  fcvt.w.s, 0x00,         -1, -1.0, rtz);
  TEST_FP_INT_OP_S_ZFINX( 6,  fcvt.w.s, 0x00,          1,  1.0, rtz);
  TEST_FP_INT_OP_S_ZFINX( 7,  fcvt.w.s, 0x00,          1,  1.1, rtz);
  TEST_FP_INT_OP_S_ZFINX(12, fcvt.wu.s, 0x10,          0, -3.0, rtz);

  #TEST_FP_INT_OP_S( 2,  fcvt.w.s, 0x01,         -1, -1.1, rtz);
  #TEST_FP_INT_OP_S( 3,  fcvt.w.s, 0x00,         -1, -1.0, rtz);
  #TEST_FP_INT_OP_S( 4,  fcvt.w.s, 0x01,          0, -0.9, rtz);
  #TEST_FP_INT_OP_S( 5,  fcvt.w.s, 0x01,          0,  0.9, rtz);
  #TEST_FP_INT_OP_S( 6,  fcvt.w.s, 0x00,          1,  1.0, rtz);
  #TEST_FP_INT_OP_S( 7,  fcvt.w.s, 0x01,          1,  1.1, rtz);
  #TEST_FP_INT_OP_S( 8,  fcvt.w.s, 0x10,     -1<<31, -3e9, rtz);
  #TEST_FP_INT_OP_S( 9,  fcvt.w.s, 0x10,  (1<<31)-1,  3e9, rtz);

  #TEST_FP_INT_OP_S(12, fcvt.wu.s, 0x10,          0, -3.0, rtz);
  #TEST_FP_INT_OP_S(13, fcvt.wu.s, 0x10,          0, -1.0, rtz);
  #TEST_FP_INT_OP_S(14, fcvt.wu.s, 0x01,          0, -0.9, rtz);
  #TEST_FP_INT_OP_S(15, fcvt.wu.s, 0x01,          0,  0.9, rtz);
  #TEST_FP_INT_OP_S(16, fcvt.wu.s, 0x00,          1,  1.0, rtz);
  #TEST_FP_INT_OP_S(17, fcvt.wu.s, 0x01,          1,  1.1, rtz);
  #TEST_FP_INT_OP_S(18, fcvt.wu.s, 0x10,          0, -3e9, rtz);
  #TEST_FP_INT_OP_S(19, fcvt.wu.s, 0x00, 3000000000,  3e9, rtz);

  # test negative NaN, negative infinity conversion
  #TEST_CASE( 42, x1, 0x000000007fffffff, la x1, tdat  ; flw f1,  0(x1); fcvt.w.s x1, f1)
  #TEST_CASE( 44, x1, 0xffffffff80000000, la x1, tdat  ; flw f1,  8(x1); fcvt.w.s x1, f1)

  # test positive NaN, positive infinity conversion
  #TEST_CASE( 52, x1, 0x000000007fffffff, la x1, tdat  ; flw f1,  4(x1); fcvt.w.s x1, f1)
  #TEST_CASE( 54, x1, 0x000000007fffffff, la x1, tdat  ; flw f1, 12(x1); fcvt.w.s x1, f1)

  # test NaN, infinity conversions to unsigned integer
  #TEST_CASE( 62, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1,  0(x1); fcvt.wu.s x1, f1)
  #TEST_CASE( 63, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1,  4(x1); fcvt.wu.s x1, f1)
  #TEST_CASE( 64, x1,                  0, la x1, tdat  ; flw f1,  8(x1); fcvt.wu.s x1, f1)
  #TEST_CASE( 65, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1, 12(x1); fcvt.wu.s x1, f1)

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

# -NaN, NaN, -inf, +inf
tdat:
.word 0xffffffff
.word 0x7fffffff
.word 0xff800000
.word 0x7f800000

RVTEST_DATA_END
