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
import qualified Blarney.Vector as V

-- Pebbles imports
import Pebbles.IO.JTAGUART
import Pebbles.Pipeline.Interface
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

-- SoC top-level module
-- ====================

-- | SoC top-level
makeTop :: SoCIns -> Module SoCOuts
makeTop socIns = mdo
  -- Instruction memory alignment requirement
  staticAssert (MemBase `mod` (4 * 2^CPUInstrMemLogWords) == 0)
    "makeTop: Instruction memory alignment requirement not met"

  -- Scalar core
  cpuOuts <- makeCPUCore
    ScalarCoreIns {
      scalarUartIn = fromUART
    , scalarMemUnit = cpuMemUnit
    , scalarSIMTResps = simtMgmtResps
    }

  -- Data cache
  (cpuMemUnit, dramReqs0) <- makeCPUDataCache dramResps0

  -- SIMT core
  simtMgmtResps <- makeSIMTAccelerator
    (cpuOuts.scalarSIMTReqs)
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
      then makeDRAMUnstoppable dramFinalReqs (socIns.socDRAMIns)
      else makeDRAM dramFinalReqs (socIns.socDRAMIns)

  -- Avalon JTAG UART wrapper module
  (fromUART, avlUARTOuts) <- makeJTAGUART
    (cpuOuts.scalarUartOut)
    (socIns.socUARTIns)

  return
    SoCOuts {
      socUARTOuts = avlUARTOuts
    , socDRAMOuts = avlDRAMOuts
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
