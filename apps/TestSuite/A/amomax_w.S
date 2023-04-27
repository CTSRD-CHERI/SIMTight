# See LICENSE for license details.

#*****************************************************************************
# amomax_w.S
#-----------------------------------------------------------------------------
#
# Test amomax.w instruction.
#

#include "riscv_test.h"
#include "test_macros.h"
#include "Config.h"

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

  TEST_CASE(2, a4, 1, \
    li a0, ADDR+4; \
    li a3, 1; \
    sw a3, 0(a0); \
    li a3, -3; \
    amomax.w a4, a3, 0(a0); \
  )

  TEST_CASE(3, a4, 1, \
    li a0, ADDR+4; \
    lw a4, 0(a0); \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
