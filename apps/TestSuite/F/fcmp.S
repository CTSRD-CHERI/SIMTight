# See LICENSE.testsuite for license details

#*****************************************************************************
# fcmp.S
#-----------------------------------------------------------------------------
#
# Test f{eq|lt|le}.s instructions.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32UF
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Arithmetic tests
  #-------------------------------------------------------------

  # Flags not checked

  TEST_FP_CMP_OP_S_ZFINX( 2, feq.s, 0, 1, -1.36, -1.36)
  TEST_FP_CMP_OP_S_ZFINX( 3, fle.s, 0, 1, -1.36, -1.36)
  TEST_FP_CMP_OP_S_ZFINX( 4, flt.s, 0, 0, -1.36, -1.36)

  TEST_FP_CMP_OP_S_ZFINX( 5, feq.s, 0, 0, -1.37, -1.36)
  TEST_FP_CMP_OP_S_ZFINX( 6, fle.s, 0, 1, -1.37, -1.36)
  TEST_FP_CMP_OP_S_ZFINX( 7, flt.s, 0, 1, -1.37, -1.36)

  # Only sNaN should signal invalid for feq.
  TEST_FP_CMP_OP_S_ZFINX( 8, feq.s, 0x00, 0, NaN, 0)
  TEST_FP_CMP_OP_S_ZFINX( 9, feq.s, 0x00, 0, NaN, NaN)
  #TEST_FP_CMP_OP_S_ZFINX(10, feq.s, 0x10, 0, sNaNf, 0)

  # qNaN should signal invalid for fle/flt.
  TEST_FP_CMP_OP_S_ZFINX(11, flt.s, 0x10, 0, NaN, 0)
  TEST_FP_CMP_OP_S_ZFINX(12, flt.s, 0x10, 0, NaN, NaN)
  #TEST_FP_CMP_OP_S_ZFINX(13, flt.s, 0x10, 0, sNaNf, 0)
  TEST_FP_CMP_OP_S_ZFINX(14, fle.s, 0x10, 0, NaN, 0)
  TEST_FP_CMP_OP_S_ZFINX(15, fle.s, 0x10, 0, NaN, NaN)
  #TEST_FP_CMP_OP_S_ZFINX(16, fle.s, 0x10, 0, sNaNf, 0)

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
