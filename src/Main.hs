-- Top-level of SIMTight SoC

module Main where

-- SoC parameters
#include <Config.h>

-- Blarney imports
import Blarney
import Blarney.Queue
import Blarney.Stream
import Blarney.Option
import Blarney.SourceSink
import Blarney.Connectable
import Blarney.Interconnect
import Blarney.ClientServer
import Blarney.Vector qualified as V
import Blarney.Vector (Vec, fromList, toList)
import Blarney.VendorIP.StreamClockCrosser

-- Pebbles imports
import Pebbles.IO.JTAGUART
import Pebbles.Memory.SBDCache
import Pebbles.Memory.Alignment
import Pebbles.Memory.Interface
import Pebbles.Memory.BankedSRAMs
import Pebbles.Memory.TagController
import Pebbles.Memory.CoalescingUnit
import Pebbles.Memory.DRAM.Bus
import Pebbles.Memory.DRAM.Wrapper
import Pebbles.Memory.DRAM.Interface
import Pebbles.Util.SinkVectoriser
import Pebbles.Pipeline.Interface
import Pebbles.Pipeline.Scalar
import Pebbles.Pipeline.SIMT
import Pebbles.Pipeline.SIMT.Management

-- SIMTight imports
import Core.SIMT
import Core.Scalar

-- SoC top-level interface
-- =======================

-- | SoC inputs
data SoCIns =
  SoCIns {
    socCPUClk :: Bit 1
    -- ^ CPU domain clock
  , socCPURst :: Bit 1
    -- ^ CPU domain reset
  , socSIMTClk :: Bit 1
    -- ^ SIMT domain clock
  , socSIMTRst :: Bit 1
    -- ^ SIMT domain reset
  , socUARTIns :: AvalonJTAGUARTIns
    -- ^ JTAG UART inputs (CPU domain)
  , socDRAMIns :: AvalonDRAMIns
    -- ^ DRAM inputs (SIMT domain)
  }
  deriving (Generic, Interface)

-- | SoC outputs
data SoCOuts =
  SoCOuts {
    socUARTOuts :: AvalonJTAGUARTOuts
    -- ^ JTAG UART outputs (CPU domain)
  , socDRAMOuts :: AvalonDRAMOuts
    -- ^ DRAM outputs (SIMT domain)
  }
  deriving (Generic, Interface)

-- Sub-domain interfaces
-- =====================

-- The SoC is split into two clock domains: one for the CPU and data
-- cache, and the other for the SIMT core and everything else.  The
-- domains are connected using Avalon streams, allowing clock-crossing
-- to be handled by vendor IP.

-- | Inputs to the CPU domain
data CPUDomainIns =
  CPUDomainIns {
    cpuDomainUARTIns :: AvalonJTAGUARTIns
  , cpuDomainFromDRAM :: Stream (DRAMResp ())
  , cpuDomainFromSIMT :: Stream SIMTResp
  }
  deriving (Generic, Interface)

-- | Outputs from the CPU domain
data CPUDomainOuts =
  CPUDomainOuts {
    cpuDomainUARTOuts :: AvalonJTAGUARTOuts
  , cpuDomainToDRAM :: Stream (DRAMReq ())
  , cpuDomainToSIMT :: Stream SIMTReq
  }
  deriving (Generic, Interface)

-- | Inputs to the SIMT domain
data SIMTDomainIns =
  SIMTDomainIns {
    simtDomainDRAMIns :: AvalonDRAMIns
  , simtDomainMgmtReqsFromCPU :: Stream SIMTReq
  , simtDomainDRAMReqsFromCPU :: Stream (DRAMReq ())
  }
  deriving (Generic, Interface)

-- | Outputs from the SIMT domain
data SIMTDomainOuts =
  SIMTDomainOuts {
    simtDomainDRAMOuts :: AvalonDRAMOuts
  , simtDomainMgmtRespsToCPU :: Stream SIMTResp
  , simtDomainDRAMRespsToCPU :: Stream (DRAMResp ())
  }
  deriving (Generic, Interface)

-- CPU domain
-- ==========

-- | CPU domain containing scalar core and L1 data cache
makeCPUDomain :: (Clock, Reset) -> CPUDomainIns -> Module CPUDomainOuts
makeCPUDomain (clk, rst) =
  makeBoundaryWithClockAndReset (clk, rst) "CPUDomain" \ins -> mdo

    -- Scalar core
    cpuOuts <- makeCPUCore
      ScalarCoreIns {
        scalarUartIn = fromUART
      , scalarMemReqs = cpuMemReqs
      , scalarMemResps = cpuMemResps
      , scalarSIMTResps = ins.cpuDomainFromSIMT
      }
 
    -- Data cache
    (cpuMemReqs, cpuMemResps, dramReqs) <- makeCPUDataCache
      ins.cpuDomainFromDRAM

    -- Avalon JTAG UART wrapper module
    (fromUART, avlUARTOuts) <- makeJTAGUART
      cpuOuts.scalarUartOut
      ins.cpuDomainUARTIns

    return
      CPUDomainOuts {
        cpuDomainUARTOuts = avlUARTOuts
      , cpuDomainToDRAM = dramReqs
      , cpuDomainToSIMT = cpuOuts.scalarSIMTReqs
      }

-- CPU core (synthesis boundary)
makeCPUCore = makeBoundary "CPUCore" (makeScalarCore config)
  where
    config =
      ScalarCoreConfig {
        scalarCoreInstrMemInitFile = Just "boot.mif"
      , scalarCoreInstrMemLogNumInstrs = CPUInstrMemLogWords
      , scalarCoreInitialPC = MemBase
      , scalarCoreEnableRegForwarding = CPUEnableRegForwarding == 1
      , scalarCoreEnableCHERI = EnableCHERI == 1
      , scalarCoreCapRegInitFile =
          if EnableCHERI == 1
            then Just (scalarCapRegInitFile ++ ".mif")
            else Nothing
      }

-- CPU data cache (synthesis boundary)
makeCPUDataCache =
  makeBoundary "CPUDataCache" (makeSBDCache @ScalarPipelineInstrInfo)

-- SIMT domain
-- ===========

makeSIMTDomain :: (Clock, Reset) -> SIMTDomainIns -> Module SIMTDomainOuts
makeSIMTDomain (clk, rst) =
  makeBoundaryWithClockAndReset (clk, rst) "SIMTDomain" \ins -> mdo

    let dramReqs0 = ins.simtDomainDRAMReqsFromCPU
    let simtMgmtReqs = ins.simtDomainMgmtReqsFromCPU

    -- SIMT core
    simtMgmtResps <- makeSIMTAccelerator simtMgmtReqs memReqs memResps

    -- SIMT memory subsystem
    (memReqs, memResps, dramReqs1) <- makeSIMTMemSubsystem dramResps1

    -- DRAM bus
    ((dramResps0, dramResps1), dramReqs) <-
      makeDRAMBus (dramReqs0, dramReqs1) dramResps

    -- Optional tag controller
    (dramResps, dramFinalReqs) <-
      if EnableTaggedMem == 1
        then makeTagController dramReqs dramFinalResps
        else makeNullTagController dramReqs dramFinalResps

    -- DRAM instance
    -- (No DRAM buffering needed when tag controller is in use;
    -- it performs its own buffering)
    (dramFinalResps, avlDRAMOuts) <-
      if EnableTaggedMem == 1
        then makeDRAMUnstoppable dramFinalReqs (ins.simtDomainDRAMIns)
        else makeDRAM dramFinalReqs (ins.simtDomainDRAMIns)

    return
      SIMTDomainOuts {
        simtDomainDRAMOuts = avlDRAMOuts
      , simtDomainMgmtRespsToCPU = simtMgmtResps
      , simtDomainDRAMRespsToCPU = dramResps0
      }

-- SIMT accelerator (synthesis boundary)
makeSIMTAccelerator = makeBoundary "SIMTAccelerator" (makeSIMTCore config)
  where
    config =
      SIMTCoreConfig {
        simtCoreInstrMemInitFile = Nothing
      , simtCoreInstrMemLogNumInstrs = CPUInstrMemLogWords
      , simtCoreInstrMemBase = MemBase
      , simtCoreExecBoundary = True
      , simtCoreEnableCHERI = EnableCHERI == 1
      , simtCoreUseFullDivider =
          if SIMTUseFullDivider == 1
            then Just SIMTFullDividerLatency
            else Nothing
      }

-- SIMT memory subsystem
-- =====================

type SIMTMemReqInfo = (SIMTPipelineInstrInfo, Vec SIMTLanes MemReqInfo)

makeSIMTMemSubsystem ::
     -- | DRAM responses
     Stream (DRAMResp ())
     -- | DRAM requests and per-lane mem units
  -> Module ( Sink (SIMTPipelineInstrInfo, Vec SIMTLanes (Option MemReq))
            , Source (SIMTPipelineInstrInfo, Vec SIMTLanes (Option MemResp))
            , Stream (DRAMReq ()) )
makeSIMTMemSubsystem dramResps = mdo
    -- Memory request queue
    memReqsQueue :: Queue (SIMTPipelineInstrInfo,
                            Vec SIMTLanes (Option MemReq)) <-
      makeSizedQueueCore 5
    let memReqs = toStream memReqsQueue

    -- Prepare request for memory subsystem
    let prepareMemReqInfo req =
          -- Remember info needed to process response
          MemReqInfo {
            memReqInfoAddr = truncate req.memReqAddr
          , memReqInfoAccessWidth = req.memReqAccessWidth
          , memReqInfoIsUnsigned = req.memReqIsUnsigned
          }
    let prepareMemReq req =
          req {
            -- Align store-data (account for access width)
            memReqData = writeAlign (req.memReqAccessWidth) (req.memReqData)
          }
    let prepareMemReqs (instrInfo, vec) =
          let memReqInfo = V.map (prepareMemReqInfo . (.val)) vec in
            ((instrInfo, memReqInfo), V.map (fmap prepareMemReq) vec)
    let memReqs1 = mapSource prepareMemReqs memReqs

    -- Coalescing unit
    (memResps, sramReqs, dramReqs) <-
      makeSIMTCoalescingUnit isBankedSRAMAccess
        memReqs1 dramResps sramResps

    -- Banked SRAMs
    sramResps <- makeSIMTBankedSRAMs sramReqs

    -- Process response from memory subsystem
    let processMemResp (memReqInfo, memResp) =
          Option memResp.valid
            memResp.val {
              -- Use info to mux loaded data
              memRespData = loadMux memResp.val.memRespData
                (truncate memReqInfo.memReqInfoAddr)
                memReqInfo.memReqInfoAccessWidth
                memReqInfo.memReqInfoIsUnsigned
            }
    let processMemResps ((instrInfo, memReqInfo), vec) =
          -- Use and drop mem request info (no longer needed)
          (instrInfo, V.map processMemResp (V.zip memReqInfo vec))
    let memResps1 = mapSource processMemResps memResps

    -- Ensure that the SRAM base address is suitably aligned
    -- (If so, remapping SRAM addresses is unecessary)
    if sramBase `mod` sramSize /= 0
      then error "SRAM base address not suitably aligned"
      else return ()

    return (toSink memReqsQueue, memResps1, dramReqs)

  where
    -- SRAM-related addresses
    simtStacksStart = 2 ^ (DRAMAddrWidth + DRAMBeatLogBytes) -
      2 ^ (SIMTLogLanes + SIMTLogWarps + SIMTLogBytesPerStack)
    sramSize = 2 ^ (SIMTLogLanes + SIMTLogWordsPerSRAMBank+2)
    sramBase = simtStacksStart - sramSize

    -- Determine if request maps to banked SRAMs
    -- (Local fence goes to banked SRAMs)
    isBankedSRAMAccess :: MemReq -> Bit 1
    isBankedSRAMAccess req =
      (req.memReqOp .!=. memGlobalFenceOp .&&.
         addr .<. fromInteger simtStacksStart .&&.
           addr .>=. fromInteger sramBase)
      where addr = req.memReqAddr

-- Coalescing unit (synthesis boundary)
makeSIMTCoalescingUnit isBankedSRAMAccess =
  makeBoundary "SIMTCoalescingUnit"
    (makeCoalescingUnit @SIMTMemReqInfo isBankedSRAMAccess)

-- Banked SRAMs (synthesis boundary)
makeSIMTBankedSRAMs =
  makeBoundary "SIMTBankedSRAMs"
    (makeBankedSRAMs @SIMTMemReqInfo)

-- SoC top-level module
-- ====================

-- | SoC top-level. No always blocks (implicit clock ignored).
makeTop :: SoCIns -> Module SoCOuts
makeTop socIns = mdo
  -- Instruction memory alignment requirement
  staticAssert (MemBase `mod` (4 * 2^CPUInstrMemLogWords) == 0)
    "makeTop: Instruction memory alignment requirement not met"

  -- Clock and reset for each domain
  let cpuClkRst = (Clock socIns.socCPUClk, Reset socIns.socCPURst)
  let simtClkRst = (Clock socIns.socSIMTClk, Reset socIns.socSIMTRst)

  -- CPU domain
  cpuOuts <- makeCPUDomain cpuClkRst
    CPUDomainIns {
      cpuDomainUARTIns = socIns.socUARTIns
    , cpuDomainFromDRAM = dramResps
    , cpuDomainFromSIMT = simtMgmtResps
    }

  -- SIMT domain
  simtOuts <- makeSIMTDomain simtClkRst
    SIMTDomainIns {
      simtDomainDRAMIns = socIns.socDRAMIns
    , simtDomainMgmtReqsFromCPU = simtMgmtReqs
    , simtDomainDRAMReqsFromCPU = dramReqs
    }

  -- SIMT management reqs (CPU -> SIMT)
  simtMgmtReqs <- makeStreamClockCrosser
    cpuClkRst
    simtClkRst
    cpuOuts.cpuDomainToSIMT

  -- SIMT management resps (SIMT -> CPU)
  simtMgmtResps <- makeStreamClockCrosser
    simtClkRst
    cpuClkRst
    simtOuts.simtDomainMgmtRespsToCPU

  -- DRAM reqs (CPU -> SIMT)
  dramReqs <- makeStreamClockCrosser
    cpuClkRst
    simtClkRst
    cpuOuts.cpuDomainToDRAM

  -- DRAM resps (SIMT -> CPU)
  dramResps <- makeStreamClockCrosser
    simtClkRst
    cpuClkRst
    simtOuts.simtDomainDRAMRespsToCPU

  return
    SoCOuts {
      socUARTOuts = cpuOuts.cpuDomainUARTOuts
    , socDRAMOuts = simtOuts.simtDomainDRAMOuts
    }

-- Initialisation files
-- ====================

scalarCapRegInitFile :: String
scalarCapRegInitFile = "ScalarCapRegFileInit"

simtCapRegInitFile :: String
simtCapRegInitFile = "SIMTCapRegFileInit"

writeInitFiles :: IO ()
writeInitFiles = do
  if EnableCHERI == 1
    then do
      let numWarps = 2 ^ SIMTLogWarps
      writeSIMTCapRegFileMif numWarps (simtCapRegInitFile ++ ".mif")
      writeSIMTCapRegFileHex numWarps (simtCapRegInitFile ++ ".hex")
      writeScalarCapRegFileMif (scalarCapRegInitFile ++ ".mif")
      writeScalarCapRegFileHex (scalarCapRegInitFile ++ ".hex")
    else return ()

-- Main function
-- =============

-- Generate code
main :: IO ()
main = do
  writeInitFiles
  writeVerilogModule makeTop "SIMTight" "./"
