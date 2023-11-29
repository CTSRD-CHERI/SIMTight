#include <Config.h>
#include <MemoryMap.h>
#include <Pebbles/CSRs/StatIds.h>

// Hardware parameters
pub const MEM_BASE                     : u32 = MemBase;
pub const SIMT_LANES                   : usize = SIMTLanes;
pub const SIMT_WARPS                   : usize = SIMTWarps;
pub const SIMT_LOG_LANES               : usize = SIMTLogLanes;
pub const SIMT_LOG_WARPS               : usize = SIMTLogWarps;
pub const DRAMBEAT_BYTES               : u32 = DRAMBeatBytes;
pub const SBDCACHE_LOG_LINES           : u32 = SBDCacheLogLines;
pub const SHARED_LOCAL_MEM_BASE        : u64 = LOCAL_MEM_BASE_LINK;
pub const SIMT_LOG_SRAM_BANKS          : u32 = SIMTLogSRAMBanks;
pub const SIMT_LOG_WORDS_PER_SRAM_BANK : u32 = SIMTLogWordsPerSRAMBank;

// Hardware features
pub const EN_RF_SCALARISATION     : u32 = SIMTEnableRegFileScalarisation;
pub const EN_SCALAR_UNIT          : u32 = SIMTEnableScalarUnit;
pub const EN_STORE_BUFFER         : u32 = SIMTEnableSVStoreBuffer;
pub const EN_CAP_RF_SCALARISATION : u32 = SIMTEnableCapRegFileScalarisation;

// Stat counter ids
pub const STAT_CYCLES              : u32 = STAT_SIMT_CYCLES;
pub const STAT_INSTRS              : u32 = STAT_SIMT_INSTRS;
pub const STAT_MAX_VEC_REGS        : u32 = STAT_SIMT_MAX_VEC_REGS;
pub const STAT_MAX_CAP_VEC_REGS    : u32 = STAT_SIMT_MAX_CAP_VEC_REGS;
pub const STAT_SCALARISABLE_INSTRS : u32 = STAT_SIMT_SCALARISABLE_INSTRS;
pub const STAT_RETRIES             : u32 = STAT_SIMT_RETRIES;
pub const STAT_SUSP_BUBBLES        : u32 = STAT_SIMT_SUSP_BUBBLES;
pub const STAT_SCALAR_SUSP_BUBBLES : u32 = STAT_SIMT_SCALAR_SUSP_BUBBLES;
pub const STAT_SCALAR_ABORTS       : u32 = STAT_SIMT_SCALAR_ABORTS;
pub const STAT_DRAM_ACCESSES       : u32 = STAT_SIMT_DRAM_ACCESSES;
pub const STAT_TOTAL_VEC_REGS      : u32 = STAT_SIMT_TOTAL_VEC_REGS;
pub const STAT_TOTAL_CAP_VEC_REGS  : u32 = STAT_SIMT_TOTAL_CAP_VEC_REGS;
pub const STAT_SB_LOAD_HIT         : u32 = STAT_SIMT_SB_LOAD_HIT;
pub const STAT_SB_LOAD_MISS        : u32 = STAT_SIMT_SB_LOAD_MISS;
pub const STAT_SB_CAP_LOAD_HIT     : u32 = STAT_SIMT_SB_CAP_LOAD_HIT;
pub const STAT_SB_CAP_LOAD_MISS    : u32 = STAT_SIMT_SB_CAP_LOAD_MISS;
