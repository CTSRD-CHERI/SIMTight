-- Top-level of SIMTight SoC

module Main where

-- SoC parameters
#include <Config.h>

-- Blarney imports
import Blarney
import Blarney.Queue
import Blarney.Stream
import Blarney.SourceSink
import Blarney.Connectable
import Blarney.Interconnect
import Blarney.Avalon.Stream
import qualified Blarney.Vector as V

-- Pebbles imports
import Pebbles.IO.JTAGUART
import Pebbles.Memory.SBDCache
import Pebbles.Memory.Alignment
import Pebbles.Memory.Interface
import Pebbles.Memory.BankedSRAMs
import Pebbles.Memory.WarpPreserver
import Pebbles.Memory.TagController
import Pebbles.Memory.CoalescingUnit
import Pebbles.Memory.DRAM.Bus
import Pebbles.Memory.DRAM.Wrapper
import Pebbles.Memory.DRAM.Interface
import Pebbles.Pipeline.Interface
import Pebbles.Pipeline.SIMT.Management

-- SIMTight imports
import Core.SIMT
import Core.Scalar

-- SoC top-level interface
-- =======================

-- | SoC inputs
data SoCIns =
  SoCIns {
    -- | JTAG UART inputs
    socUARTIns :: AvalonJTAGUARTIns
    -- | DRAM inputs
  , socDRAMIns :: AvalonDRAMIns
  }
  deriving (Generic, Interface)

-- | SoC outputs
data SoCOuts =
  SoCOuts {
    -- | JTAG UART outputs
    socUARTOuts :: AvalonJTAGUARTOuts
    -- | DRAM outputs
  , socDRAMOuts :: AvalonDRAMOuts
  }
  deriving (Generic, Interface)

-- Sub-domain interfaces
-- =====================

-- The SoC is split into two clock domains: one for the CPU and data
-- cache, and the other for the SIMT core and everything else.  The
-- domains are connected using Avalon streams, allowing clock-crossing
-- to be handled externally by the Intel tools.

-- | Inputs to the CPU domain
data CPUDomainIns =
  CPUDomainIns {
    cpuDomainUARTIns :: AvalonJTAGUARTIns
  , cpuDomainFromDRAM :: AvlStream (DRAMResp ())
  , cpuDomainFromSIMT :: AvlStream SIMTResp
  }
  deriving (Generic, Interface)

-- | Outputs from the CPU domain
data CPUDomainOuts =
  CPUDomainOuts {
    cpuDomainUARTOuts :: AvalonJTAGUARTOuts
  , cpuDomainToDRAM :: AvlStream (DRAMReq ())
  , cpuDomainToSIMT :: AvlStream SIMTReq
  }
  deriving (Generic, Interface)

-- | Inputs to the SIMT domain
data SIMTDomainIns =
  SIMTDomainIns {
    simtDomainDRAMIns :: AvalonDRAMIns
  , simtDomainMgmtReqsFromCPU :: AvlStream SIMTReq
  , simtDomainDRAMReqsFromCPU :: AvlStream (DRAMReq ())
  }
  deriving (Generic, Interface)

-- | Outputs from the SIMT domain
data SIMTDomainOuts =
  SIMTDomainOuts {
    simtDomainDRAMOuts :: AvalonDRAMOuts
  , simtDomainMgmtRespsToCPU :: AvlStream SIMTResp
  , simtDomainDRAMRespsToCPU :: AvlStream (DRAMResp ())
  }
  deriving (Generic, Interface)

-- CPU domain
-- ==========

-- | CPU domain containing scalar core and L1 data cache
makeCPUDomain :: CPUDomainIns -> Module CPUDomainOuts
makeCPUDomain = makeBoundary "CPUDomain" \ins -> mdo
  -- Scalar core
  cpuOuts <- makeCPUCore
    ScalarCoreIns {
      scalarUartIn = fromUART
    , scalarMemUnit = cpuMemUnit
    , scalarSIMTResps = ins.cpuDomainFromSIMT.fromAvlStream
    }
 
  -- Data cache
  (cpuMemUnit, dramReqs) <- makeCPUDataCache
    (ins.cpuDomainFromDRAM.fromAvlStream)

  -- Avalon JTAG UART wrapper module
  (fromUART, avlUARTOuts) <- makeJTAGUART
    (cpuOuts.scalarUartOut)
    (ins.cpuDomainUARTIns)

  return
    CPUDomainOuts {
      cpuDomainUARTOuts = avlUARTOuts
    , cpuDomainToDRAM = dramReqs.toAvlStream
    , cpuDomainToSIMT = cpuOuts.scalarSIMTReqs.toAvlStream
    }

-- CPU core (synthesis boundary)
makeCPUCore = makeBoundary "CPUCore" (makeScalarCore config)
  where
    config =
      ScalarCoreConfig {
        scalarCoreInstrMemInitFile = Just "boot.mif"
      , scalarCoreInstrMemLogNumInstrs = CPUInstrMemLogWords
      , scalarCoreInitialPC = MemBase
      , scalarCoreEnableCHERI = EnableCHERI == 1
      , scalarCoreCapRegInitFile =
          if EnableCHERI == 1
            then Just (scalarCapRegInitFile ++ ".mif")
            else Nothing
      }

-- CPU data cache (synthesis boundary)
makeCPUDataCache = makeBoundary "CPUDataCache" (makeSBDCache @InstrInfo)

-- SIMT domain
-- ===========

makeSIMTDomain :: SIMTDomainIns -> Module SIMTDomainOuts
makeSIMTDomain = makeBoundary "SIMTDomain" \ins -> mdo
  let dramReqs0 = ins.simtDomainDRAMReqsFromCPU.fromAvlStream
  let simtMgmtReqs = ins.simtDomainMgmtReqsFromCPU.fromAvlStream

  -- SIMT core
  simtMgmtResps <- makeSIMTAccelerator
    simtMgmtReqs
    simtMemUnits

  -- SIMT memory subsystem
  (simtMemUnits, dramReqs1) <- makeSIMTMemSubsystem dramResps1

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
    , simtDomainMgmtRespsToCPU = simtMgmtResps.toAvlStream
    , simtDomainDRAMRespsToCPU = dramResps0.toAvlStream
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
      , simtCoreCapRegInitFile =
          if EnableCHERI == 1
            then Just (simtCapRegInitFile ++ ".mif")
            else Nothing
      , simtCoreUseIntelDivider =
          if SIMTUseIntelDivider == 1
            then Just SIMTIntelDividerLatency
            else Nothing
      }

-- SIMT memory subsystem
-- =====================

type SIMTMemReqId = (InstrInfo, MemReqInfo)

makeSIMTMemSubsystem ::
     -- | DRAM responses
     Stream (DRAMResp ())
     -- | DRAM requests and per-lane mem units
  -> Module (V.Vec SIMTLanes (MemUnit InstrInfo), Stream (DRAMReq ()))
makeSIMTMemSubsystem dramResps = mdo
    -- Warp preserver
    (memReqs, simtMemUnits) <- makeWarpPreserver memResps1

    -- Prepare request for memory subsystem
    let prepareReq req =
          req {
            -- Align store-data (account for access width)
            memReqData =
              writeAlign (req.memReqAccessWidth) (req.memReqData)
            -- Remember info needed to process response
          , memReqId =
              ( req.memReqId
              , MemReqInfo {
                  memReqInfoAddr = req.memReqAddr.truncate
                , memReqInfoAccessWidth = req.memReqAccessWidth
                , memReqInfoIsUnsigned = req.memReqIsUnsigned
                }
              )
          }
    let memReqs1 = map (mapSource prepareReq) memReqs

    -- Coalescing unit
    (memResps, sramReqs, dramReqs) <-
      makeSIMTCoalescingUnit isBankedSRAMAccess
        (V.fromList memReqs1) dramResps sramResps

    -- Banked SRAMs
    let sramRoute info = info.bankLaneId
    sramResps <- makeSIMTBankedSRAMs sramRoute sramReqs

    -- Process response from memory subsystem
    let processResp resp =
          resp {
            -- | Drop info, no longer needed
            memRespId = resp.memRespId.fst
            -- | Use info to mux loaded data
          , memRespData = loadMux (resp.memRespData)
              (resp.memRespId.snd.memReqInfoAddr.truncate)
              (resp.memRespId.snd.memReqInfoAccessWidth)
              (resp.memRespId.snd.memReqInfoIsUnsigned)
          }
    let memResps1 = map (mapSource processResp) (V.toList memResps)

    -- Ensure that the SRAM base address is suitably aligned
    -- (If so, remapping SRAM addresses is unecessary)
    if sramBase `mod` sramSize /= 0
      then error "SRAM base address not suitably aligned"
      else return ()

    return (V.fromList simtMemUnits, dramReqs)

  where
    -- SRAM-related addresses
    simtStacksStart = 2 ^ (DRAMAddrWidth + DRAMBeatLogBytes) -
      2 ^ (SIMTLogLanes + SIMTLogWarps + SIMTLogBytesPerStack)
    sramSize = 2 ^ (SIMTLogLanes + SIMTLogWordsPerSRAMBank+2)
    sramBase = simtStacksStart - sramSize

    -- Determine if request maps to banked SRAMs
    -- (Local fence goes to banked SRAMs)
    isBankedSRAMAccess :: MemReq t_id -> Bit 1
    isBankedSRAMAccess req =
      req.memReqOp .==. memLocalFenceOp .||.
        (req.memReqOp .!=. memGlobalFenceOp .&&.
           addr .<. fromInteger simtStacksStart .&&.
             addr .>=. fromInteger sramBase)
      where addr = req.memReqAddr

-- Coalescing unit (synthesis boundary)
makeSIMTCoalescingUnit isBankedSRAMAccess =
  makeBoundary "SIMTCoalescingUnit"
    (makeCoalescingUnit @SIMTMemReqId isBankedSRAMAccess)

-- Banked SRAMs (synthesis boundary)
makeSIMTBankedSRAMs route =
  makeBoundary "SIMTBankedSRAMs"
    (makeBankedSRAMs @(BankInfo SIMTMemReqId) route)

-- SoC top-level module
-- ====================

-- | SoC top-level. This module should contain only connections,
-- not logic, because the Quartus project may instantiate the CPU and
-- SIMT domains directly, with only clock-crossing logic between the two.
makeTop :: SoCIns -> Module SoCOuts
makeTop socIns = mdo
  -- Instruction memory alignment requirement
  staticAssert (MemBase `mod` (4 * 2^CPUInstrMemLogWords) == 0)
    "makeTop: Instruction memory alignment requirement not met"

  -- CPU domain
  cpuOuts <- makeCPUDomain
    CPUDomainIns {
      cpuDomainUARTIns = socIns.socUARTIns
    , cpuDomainFromDRAM = simtOuts.simtDomainDRAMRespsToCPU
    , cpuDomainFromSIMT = simtOuts.simtDomainMgmtRespsToCPU
    }

  -- SIMT domain
  simtOuts <- makeSIMTDomain
    SIMTDomainIns {
      simtDomainDRAMIns = socIns.socDRAMIns
    , simtDomainMgmtReqsFromCPU = cpuOuts.cpuDomainToSIMT
    , simtDomainDRAMReqsFromCPU = cpuOuts.cpuDomainToDRAM
    }

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
