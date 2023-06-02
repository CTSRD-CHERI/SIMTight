# See LICENSE for license details.

#*****************************************************************************
# camoadd_w.S
#-----------------------------------------------------------------------------
#
# Test camoadd.w instruction.
#

#include "riscv_test.h"
#include "test_macros.h"
#include <Config.h>

RVTEST_RV32U
RVTEST_CODE_BEGIN

#ifdef _TEST_SIMT_

  # Base of banked SRAMs
  #define ADDR (\
    1 << (DRAMAddrWidth + DRAMBeatLogBytes) - \
    1 << (SIMTLogLanes + SIMTLogWarps + SIMTLogBytesPerStack) - \
    1 << (SIMTLogSRAMBanks + SIMTLogWordsPerSRAMBank + 2))

  # Use only a single thread for this test
  csrrw t3, 0xf14, zero
  bnez t3, pass

#else

  #define ADDR 0
  j fail

#endif

  # Read almighty cap and set address
  cspecialrw c1, ddc, c0
  li x2, ADDR
  csetaddr c1, c1, x2

  TEST_CASE(2, x4, 1, \
    li x2, 1; \
    csw x2, 0(c1); \
    li x3, 3; \
    camoand.w x4, x3, 0(c1); \
  )

  TEST_CASE(3, x4, 1, \
    clw x4, 0(c1); \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
