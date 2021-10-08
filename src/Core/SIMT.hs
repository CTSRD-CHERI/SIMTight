-- 32-bit SIMT core

module Core.SIMT where

-- SoC configuration
#include <Config.h>

-- Blarney imports
import Blarney
import Blarney.Queue
import Blarney.Stream
import Blarney.PulseWire
import Blarney.SourceSink
import Blarney.Connectable
import Blarney.Interconnect
import Blarney.Vector (Vec, toList)

-- Pebbles imports
import Pebbles.CSRs.Hart
import Pebbles.CSRs.CSRUnit
import Pebbles.CSRs.Custom.Simulate
import Pebbles.CSRs.Custom.SIMTDevice
import Pebbles.Util.Counter
import Pebbles.Pipeline.SIMT
import Pebbles.Pipeline.SIMT.Management
import Pebbles.Pipeline.Interface
import Pebbles.Memory.Interface
import Pebbles.Memory.DRAM.Interface
import Pebbles.Instructions.RV32_I
import Pebbles.Instructions.RV32_M
import Pebbles.Instructions.RV32_A
import Pebbles.Instructions.Mnemonics
import Pebbles.Instructions.RV32_xCHERI
import Pebbles.Instructions.Units.MulUnit
import Pebbles.Instructions.Units.DivUnit
import Pebbles.Instructions.Custom.SIMT

-- CHERI imports
import CHERI.CapLib

-- Haskell imports
import Data.List
import Numeric (showHex)

-- Execute stage
-- =============

-- | SIMT execute stage inputs
data SIMTExecuteIns =
  SIMTExecuteIns {
    execLaneId :: Bit SIMTLogLanes
    -- ^ Lane id
  , execWarpId :: Bit SIMTLogWarps
    -- ^ Warp id
  , execKernelAddr :: Bit 32
    -- ^ Kernel address
  , execWarpCmd :: Wire WarpCmd
    -- ^ Wire containing warp command
  , execMemUnit :: MemUnit InstrInfo
    -- ^ Memory unit interface for lane
  } deriving (Generic, Interface)

-- | Execute stage for a SIMT lane (synthesis boundary)
makeSIMTExecuteStage ::
     Bool
     -- ^ Enable CHERI?
  -> Maybe Int
     -- ^ Use intel divider? (If so, what is its latency?)
  -> SIMTExecuteIns -> State -> Module ExecuteStage
makeSIMTExecuteStage enCHERI useFullDiv =
  makeBoundary "SIMTExecuteStage" \ins s -> do
    -- Multiplier per vector lane
    mulUnit <- makeFullMulUnit

    -- Divider per vector lane
    divUnit <-
      case useFullDiv of
        Nothing -> makeSeqDivUnit
        Just latency -> makeFullDivUnit latency

    -- SIMT warp control CSRs
    csr_WarpCmd <- makeCSR_WarpCmd (ins.execLaneId) (ins.execWarpCmd)
    csr_WarpGetKernel <- makeCSR_WarpGetKernel (ins.execKernelAddr)

    -- CSR unit
    let hartId = zeroExtend (ins.execWarpId # ins.execLaneId)
    csrUnit <- makeCSRUnit $
         csrs_Sim
      ++ [csr_HartId hartId]
      ++ [csr_WarpCmd]
      ++ [csr_WarpGetKernel]
 
    -- Memory requests from execute stage
    (memReqSink, capMemReqSink) <-
      if enCHERI
        then do
          capMemReqSink <- makeCapMemReqSink (ins.execMemUnit.memReqs)
          let memReqSink = mapSink toCapMemReq capMemReqSink
          return (memReqSink, capMemReqSink)
        else return (ins.execMemUnit.memReqs, nullSink)

    -- Pipeline resume requests from memory
    memResumeReqs <- makeMemRespToResumeReq
      enCHERI
      (ins.execMemUnit.memResps)

    -- Merge resume requests
    let resumeReqStream =
          memResumeReqs `mergeTwo`
            mergeTwo (mulUnit.mulResps) (divUnit.divResps)

    -- Resume queue
    resumeQueue <- makePipelineQueue 1
    makeConnection resumeReqStream (resumeQueue.toSink)

    return
      ExecuteStage {
        execute = do
          executeI (Just mulUnit) csrUnit memReqSink s
          executeM mulUnit divUnit s
          if enCHERI
            then executeCHERI csrUnit capMemReqSink s
            else do
              executeI_NoCap csrUnit memReqSink s
              executeA memReqSink s
      , resumeReqs = resumeQueue.toStream
      }

-- Core
-- ====

-- | Configuration parameters
data SIMTCoreConfig =
  SIMTCoreConfig {
    simtCoreInstrMemInitFile :: Maybe String
    -- ^ Initialisation file for instruction memory
  , simtCoreInstrMemLogNumInstrs :: Int
    -- ^ Size of tightly coupled instruction memory
  , simtCoreInstrMemBase :: Integer
    -- ^ Base of instr mem within memory map
  , simtCoreExecBoundary :: Bool
    -- ^ Synthesis boundary on execute stage?
  , simtCoreEnableCHERI :: Bool
    -- ^ Enable CHERI extensions?
  , simtCoreUseExtraPreExecStage :: Bool
    -- ^ Extra pipeline stage?
  , simtCoreCapRegInitFile :: Maybe String
    -- ^ File containing initial capability register file (meta-data only)
  , simtCoreUseFullDivider :: Maybe Int
    -- ^ Use full throughput divider?
    -- (If so, what latency? If not, slow seq divider used)
  }

-- | RV32IM SIMT core
makeSIMTCore ::
     -- | Configuration parameters
     SIMTCoreConfig
     -- | SIMT management requests
  -> Stream SIMTReq
     -- | Memory unit per vector lane
  -> Vec SIMTLanes (MemUnit InstrInfo)
     -- | SIMT management responses
  -> Module (Stream SIMTResp)
makeSIMTCore config mgmtReqs memUnitsVec = mdo
  let memUnits = toList memUnitsVec

  -- Apply stack address interleaving
  let memUnits' = interleaveStacks memUnits

  -- Wire for warp command
  warpCmdWire :: Wire WarpCmd <- makeWire dontCare

  -- Pipeline configuration
  let pipelineConfig =
        SIMTPipelineConfig {
          instrMemInitFile = config.simtCoreInstrMemInitFile
        , instrMemLogNumInstrs = config.simtCoreInstrMemLogNumInstrs
        , instrMemBase = config.simtCoreInstrMemBase
        , logNumWarps = SIMTLogWarps
        , logMaxNestLevel = SIMTLogMaxNestLevel
        , enableStatCounters = SIMTEnableStatCounters == 1
        , capRegInitFile = config.simtCoreCapRegInitFile
        , checkPCCFunc =
            if config.simtCoreEnableCHERI then Just checkPCC else Nothing
        , enableCapRegFileTrace = EnableCapRegFileTrace == 1
        , useExtraPreExecStage = config.simtCoreUseExtraPreExecStage
        , useSharedPCC = SIMTUseSharedPCC == 1
        , decodeStage = concat
            [ decodeI
            , if config.simtCoreEnableCHERI
                then decodeCHERI
                else decodeI_NoCap
            , decodeM
            , if config.simtCoreEnableCHERI
                then decodeCHERI_A
                else decodeA
            , decodeSIMT
            ]
        , executeStage =
            [ makeSIMTExecuteStage
                (config.simtCoreEnableCHERI)
                (config.simtCoreUseFullDivider)
                SIMTExecuteIns {
                  execLaneId = fromInteger i
                , execWarpId = pipelineOuts.simtCurrentWarpId.truncate
                , execKernelAddr = pipelineOuts.simtKernelAddr
                , execWarpCmd = warpCmdWire
                , execMemUnit = memUnit
                }
            | (memUnit, i) <- zip memUnits' [0..] ]
        , simtPushTag = SIMT_PUSH
        , simtPopTag = SIMT_POP
        }

  -- Pipeline instantiation
  pipelineOuts <- makeSIMTPipeline pipelineConfig
    SIMTPipelineIns {
      simtMgmtReqs = mgmtReqs
    , simtWarpCmdWire = warpCmdWire
    }

  return (pipelineOuts.simtMgmtResps)

-- | Stack address interleaver so that accesses to same stack
-- offset by different threads in a warp are coalesced
interleaveStacks :: [MemUnit id] -> [MemUnit id]
interleaveStacks memUnits =
    [ memUnit {
        memReqs = mapSink interleaveReq (memUnit.memReqs)
      }
    | memUnit <- memUnits]
  where
    interleaveReq :: MemReq id -> MemReq id
    interleaveReq req = req { memReqAddr = interleaveAddr (req.memReqAddr) }

    interleaveAddr :: Bit 32 -> Bit 32
    interleaveAddr a =
      if top .==. ones
        then top # stackOffset # stackId # wordOffset
        else a
      where
        top = slice @31 @(SIMTLogWarps+SIMTLogLanes+SIMTLogBytesPerStack) a
        stackId = slice @(SIMTLogWarps+SIMTLogLanes+SIMTLogBytesPerStack-1)
                        @SIMTLogBytesPerStack a
        stackOffset = slice @(SIMTLogBytesPerStack-1) @2 a
        wordOffset = slice @1 @0 a

-- Register file initialisation
-- ============================

-- | Write initial capability reg file (meta-data only, mif format)
writeSIMTCapRegFileMif :: Int -> String -> IO ()
writeSIMTCapRegFileMif numWarps filename =
  writeFile filename $ unlines $
    [ "DEPTH = " ++ show numRegs ++ ";"
    , "WIDTH = " ++ show (valueOf @CapMemMetaWidth) ++ ";"
    , "ADDRESS_RADIX = DEC;"
    , "DATA_RADIX = HEX;"
    , "CONTENT"
    , "BEGIN"
    ] ++
    [ show i ++ " : " ++ showHex nullCapMemMetaInteger ";"
    | i <- [0..numRegs-1]
    ] ++
    ["END"]
  where
    numRegs = 32 * numWarps

-- | Write initial capability reg file (meta-data only, hex format)
writeSIMTCapRegFileHex :: Int -> String -> IO ()
writeSIMTCapRegFileHex numWarps filename =
  writeFile filename $ unlines $
    [ showHex nullCapMemMetaInteger ""
    | i <- [0..numRegs-1]
    ]
  where
    numRegs = 32 * numWarps
