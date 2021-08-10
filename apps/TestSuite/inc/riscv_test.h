// See LICENSE for license details.

#ifndef _ENV_PHYSICAL_SINGLE_CORE_H
#define _ENV_PHYSICAL_SINGLE_CORE_H

#include "encoding.h"
#include "Config.h"

//-----------------------------------------------------------------------
// Begin Macro
//-----------------------------------------------------------------------

#define RVTEST_RV32U                                                    \
  .macro init;                                                          \
  .endm

#define RVTEST_RV32UF                                                   \
  .macro init;                                                          \
  RVTEST_FP_ENABLE;                                                     \
  .endm

#define RVTEST_FP_ENABLE                                                \
  li a0, MSTATUS_FS & (MSTATUS_FS >> 1);                                \
  csrs mstatus, a0;                                                     \
  csrwi fcsr, 0

#define RISCV_MULTICORE_DISABLE                                         \
  csrr a0, mhartid;                                                     \
  1: bnez a0, 1b

#ifdef _TEST_SIMT_
  #define RVTEST_CODE_BEGIN .global _test_start; _test_start: SIMT_Push
#else
  #define RVTEST_CODE_BEGIN .global _test_start; _test_start:
#endif

//-----------------------------------------------------------------------
// End Macro
//-----------------------------------------------------------------------

#define RVTEST_CODE_END                                                 \
        j .  

//-----------------------------------------------------------------------
// Pass/Fail Macro
//-----------------------------------------------------------------------

#define TESTNUM x28

#ifndef _TEST_SIMT_

#define CSR_UARTCanPut 0x802
#define CSR_UARTPut 0x803

#define RVTEST_PASS                                                     \
        li TESTNUM, 1;                                                  \
        1: csrrw t0, CSR_UARTCanPut, zero;                              \
        beq t0, zero, 1b;                                               \
        csrw CSR_UARTPut, TESTNUM;                                      \
        li t0, MemBase;                                                 \
        jr t0;

#define RVTEST_FAIL                                                     \
        sll TESTNUM, TESTNUM, 1;                                        \
        1:csrrw t0, CSR_UARTCanPut, zero;                               \
        beq t0, zero, 1b;                                               \
        csrw CSR_UARTPut, TESTNUM;                                      \
        li t0, MemBase;                                                 \
        jr t0;

#else

#define SIMT_Push .word 0x00050009
#define SIMT_Pop .word 0x00051009
#define CSR_WarpCmd 0x830

#ifdef _TEST_SIMT_
  #define CONVERGE SIMT_Pop
#else
  #define CONVERGE
#endif

#define RVTEST_PASS                                                     \
        CONVERGE;                                                       \
        li TESTNUM, 3;                                                  \
        csrw CSR_WarpCmd, TESTNUM;

#define RVTEST_FAIL                                                     \
        li TESTNUM, 1;                                                  \
        csrw CSR_WarpCmd, TESTNUM;

#endif

//-----------------------------------------------------------------------
// Data Section Macro
//-----------------------------------------------------------------------

#define EXTRA_DATA

#define RVTEST_DATA_BEGIN                                               \
        EXTRA_DATA                                                      \
        .align 4; .global begin_signature; begin_signature:

#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:

#endif
