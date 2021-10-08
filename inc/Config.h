#define NOTE(string)

NOTE("SIMTight SoC configuration")

NOTE("DRAM configuration")
NOTE("==================")

NOTE("Size of data field in DRAM request/response")
#define DRAMBeatBits 512
#define DRAMBeatBytes 64
#define DRAMBeatHalfs 32
#define DRAMBeatWords 16
#define DRAMBeatLogBytes 6

NOTE("Max burst size = 2^(DRAMBurstWidth-1)")
#define DRAMBurstWidth 4

NOTE("Size of DRAM in beats")
#define DRAMAddrWidth 26

NOTE("Maximum number of inflight requests supported by DRAM wrapper")
#define DRAMLogMaxInFlight 5

NOTE("SIMT configuration")
NOTE("==================")

NOTE("Number of lanes per SIMT core")
#define SIMTLanes 32
#define SIMTLogLanes 5

NOTE("Number of warps per SIMT core")
#define SIMTWarps 64
#define SIMTLogWarps 6

NOTE("Use extra pipeline stage (pre-execute)?")
#define SIMTUseExtraPreExecStage 1

NOTE("Number of bits used to track divergence nesting level")
#define SIMTLogMaxNestLevel 5

NOTE("Stack size (in bytes) for each SIMT thread")
#define SIMTLogBytesPerStack 19

NOTE("Size of each SRAM bank (in words)")
#define SIMTLogWordsPerSRAMBank 9

NOTE("Enable SIMT stat counters")
#define SIMTEnableStatCounters 1

NOTE("Size of SRAM multicast id in coalescing unit")
#define SIMTMcastIdSize 4

NOTE("Use full-throughput divider (rather than sequential divider)")
#define SIMTUseFullDivider 0

NOTE("Latency of full-throughput divider")
#define SIMTFullDividerLatency 12

NOTE("Use shared immutable PCC meta-data for all threads in kernel?")
#define SIMTUseSharedPCC 1

NOTE("CPU configuration")
NOTE("=================")

NOTE("Size of tightly coupled instruction memory")
#define CPUInstrMemLogWords 13

NOTE("Number of cache lines (line size == DRAM beat size)")
#define SBDCacheLogLines 9

NOTE("Use register forwarding for increased IPC but possibly lower Fmax?")
#define CPUEnableRegForwarding 0

NOTE("Tagged memory")
NOTE("=============")

NOTE("Is tagged memory enabled? (Needed for CHERI)")
#define EnableTaggedMem 0

NOTE("Tag cache: line size")
#define TagCacheLogBeatsPerLine 1

NOTE("Tag cache: number of set-associative ways")
#define TagCacheLogNumWays 2

NOTE("Tag cache: number of sets")
#define TagCacheLogSets 7

NOTE("Tag cache: max number of inflight memory requests")
#define TagCacheLogMaxInflight 5

NOTE("Tag cache: max number of pending requests per way")
#define TagCachePendingReqsPerWay 16

NOTE("CHERI support")
NOTE("=============")

NOTE("Is CHERI enabled? (If so, see UseClang and EnableTaggedMem settings)")
#define EnableCHERI 0

NOTE("Trace accesses to capability register file in simulation?")
#define EnableCapRegFileTrace 0

NOTE("Compiler")
NOTE("========")

NOTE("Use clang rather than gcc? (Currently required if CHERI enabled)")
#define UseClang 0

NOTE("Memory map")
NOTE("==========")

NOTE("Memory base (after tag bit region)")
#define MemBase 134217728

NOTE("Space reserved for boot loader")
#define MaxBootImageBytes 1024
