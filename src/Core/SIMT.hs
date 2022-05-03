-- 32-bit SIMT core

module Core.SIMT where

-- SoC configuration
#include <Config.h>

-- Blarney imports
import Blarney
import Blarney.Queue
import Blarney.Stream
import Blarney.Option
import Blarney.PulseWire
import Blarney.SourceSink
import Blarney.Connectable
import Blarney.ClientServer
import Blarney.Interconnect
import Blarney.Vector (Vec, toList, fromList)
import Blarney.Vector qualified as V

-- Pebbles imports
import Pebbles.CSRs.Hart
import Pebbles.CSRs.CSRUnit
import Pebbles.CSRs.Custom.Simulate
import Pebbles.CSRs.Custom.SIMTDevice
import Pebbles.Util.Counter
import Pebbles.Util.SinkVectoriser
import Pebbles.Pipeline.SIMT
import Pebbles.Pipeline.SIMT.Management
import Pebbles.Pipeline.Interface
import Pebbles.Memory.Interface
import Pebbles.Memory.CapSerDes
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
  , execMemReqs :: Sink MemReq
    -- ^ Sink for memory requests
  , execCapMemReqs :: Sink CapMemReq
    -- ^ Sink for capability memory requests
  , execMulReqs :: Sink MulReq
    -- ^ Sink for multiplier requests
  , execDivReqs :: Sink DivReq
    -- ^ Sink for divider requests
  } deriving (Generic, Interface)

-- | Execute stage for a SIMT lane (synthesis boundary)
makeSIMTExecuteStage ::
     Bool
     -- ^ Enable CHERI?
  -> SIMTExecuteIns -> State -> Module ExecuteStage
makeSIMTExecuteStage enCHERI =
  makeBoundary "SIMTExecuteStage" \ins s -> do

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
 
    return
      ExecuteStage {
        execute = do
          executeI (Just ins.execMulReqs) csrUnit ins.execMemReqs s
          executeM ins.execMulReqs ins.execDivReqs s
          if enCHERI
            then executeCHERI csrUnit ins.execCapMemReqs s
            else do
              executeI_NoCap csrUnit ins.execMemReqs s
              executeA ins.execMemReqs s
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
  , simtCoreUseFullDivider :: Maybe Int
    -- ^ Use full throughput divider?
    -- (If so, what latency? If not, slow seq divider used)
  }

-- | 32-bit SIMT core
makeSIMTCore ::
     SIMTCoreConfig
     -- ^ Configuration parameters
  -> Stream SIMTReq
     -- ^ SIMT management requests
  -> Sink (SIMTPipelineInstrInfo, Vec SIMTLanes (Option MemReq))
     -- ^ Memory requests
  -> Stream (SIMTPipelineInstrInfo, Vec SIMTLanes (Option MemResp))
     -- ^ Memory responses
  -> Module (Stream SIMTResp)
     -- ^ SIMT management responses
makeSIMTCore config mgmtReqs memReqs memResps = mdo

  -- Scalar unit enabled?
  let enScalarUnit = SIMTEnableScalarUnit == 1

  -- Multiplier requests
  -- ===================

  -- Vector multiplier unit
  vecMulUnit <- makeFullVecMulUnit

  -- Per lane multiplier request sinks
  mulSinks <- makeSinkVectoriser pipelineOuts.simtInstrInfo vecMulUnit.reqs

  -- Multiplier for scalar unit
  scalarMulUnit <- if enScalarUnit then makeFullMulUnit else return nullServer

  -- Divider requests
  -- ================

  -- Vector division unit
  vecDivUnit <-
    case config.simtCoreUseFullDivider of
      Nothing -> makeSeqVecDivUnit
      Just latency -> makeFullVecDivUnit latency

  -- Per lane divider request sinks
  divSinks <- makeSinkVectoriser pipelineOuts.simtInstrInfo vecDivUnit.reqs

  -- Divider for scalar unit
  scalarDivUnit <-
    if enScalarUnit
      then case config.simtCoreUseFullDivider of
             Nothing -> makeSeqDivUnit
             Just latency -> makeFullDivUnit latency
      else return nullServer

  -- Memory requests
  -- ===============

  -- Apply stack address interleaving
  let ilv (info, vec) = (info, V.map (fmap interleaveStacks) vec)
  let memReqsIlv = mapSink ilv memReqs

  -- Per lane memory request sinks
  (memReqSinks, capMemReqSinks) <-
    if config.simtCoreEnableCHERI
      then do
        -- Serialise CapMemReq to MemReq
        capMemReqs <- makeCapMemReqSerialiser memReqsIlv
        capMemReqsBuffered <- makeSinkBuffer (makePipelineQueue 1) capMemReqs
        -- Sink of vectors to vector of sinks
        capMemSinks <-
          makeSinkVectoriser pipelineOuts.simtInstrInfo capMemReqsBuffered
        -- Convert vector to list
        let capMemSinksList = toList capMemSinks
        let memSinks = map (mapSink toCapMemReq) capMemSinksList
        return (memSinks, capMemSinksList)
      else do
        -- Sink of vectors to vector of sinks
        memSinks <- makeSinkVectoriser pipelineOuts.simtInstrInfo memReqsIlv
        return (toList memSinks, replicate SIMTLanes nullSink)

  -- Responses
  -- =========

  -- Pipeline resume requests from memory
  memResumeReqs <- makeMemRespDeserialiser
                     config.simtCoreEnableCHERI memResps

  -- Merge resume requests
  let resumeReqStream =
        memResumeReqs `mergeTwo`
          (vecMulUnit.resps `mergeTwo` vecDivUnit.resps)

  -- Resume queue
  resumeQueue <- makePipelineQueue 1
  makeConnection resumeReqStream (toSink resumeQueue)

  -- Merge scalar unit resume requests
  let scalarResumeReqStream =
         scalarMulUnit.resps `mergeTwo` scalarDivUnit.resps

  -- Scalar unit resume eueue
  scalarResumeQueue <- makePipelineQueue 1
  makeConnection scalarResumeReqStream (toSink scalarResumeQueue)

  -- Pipeline
  -- ========

  -- Wire for warp command
  warpCmdWire :: Wire WarpCmd <- makeWire dontCare

  -- Pipeline configuration
  let pipelineConfig =
        SIMTPipelineConfig {
          instrMemInitFile = config.simtCoreInstrMemInitFile
        , instrMemLogNumInstrs = config.simtCoreInstrMemLogNumInstrs
        , instrMemBase = config.simtCoreInstrMemBase
        , enableStatCounters = SIMTEnableStatCounters == 1
        , checkPCCFunc =
            if config.simtCoreEnableCHERI then Just checkPCC else Nothing
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
                SIMTExecuteIns {
                  execLaneId = fromInteger i
                , execWarpId = truncate pipelineOuts.simtCurrentWarpId
                , execKernelAddr = pipelineOuts.simtKernelAddr
                , execWarpCmd = warpCmdWire
                , execMemReqs = memSink
                , execCapMemReqs = capMemSink
                , execMulReqs = mulSink
                , execDivReqs = divSink
                }
            | (memSink, capMemSink, mulSink, divSink, i) <-
                zip5 memReqSinks capMemReqSinks
                     (toList mulSinks) (toList divSinks) [0..] ]
        , simtPushTag = SIMT_PUSH
        , simtPopTag = SIMT_POP
        , useRegFileScalarisation = SIMTEnableRegFileScalarisation == 1
        , useCapRegFileScalarisation = SIMTEnableCapRegFileScalarisation == 1
        , useAffineScalarisation = SIMTEnableAffineScalarisation == 1
        , useScalarUnit = enScalarUnit
        , scalarUnitAllowList = []
        , scalarUnitDecodeStage = concat
            [ decodeI
            , decodeI_NoCap
            , decodeM
            ]
        , scalarUnitExecuteStage = \s -> do
            -- CSRs not supported in scalar unit
            let scalarCSRUnit = nullCSRUnit
 
            -- Memory access not supported in scalar unit
            let scalarMemReqs = nullSink

            return
              ExecuteStage {
                execute = do
                  executeI (Just scalarMulUnit.reqs)
                             scalarCSRUnit scalarMemReqs s
                  executeM scalarMulUnit.reqs scalarDivUnit.reqs s
                  executeI_NoCap scalarCSRUnit scalarMemReqs s
              }
        }

  -- Pipeline instantiation
  pipelineOuts <- makeSIMTPipeline pipelineConfig
    SIMTPipelineIns {
      simtMgmtReqs = mgmtReqs
    , simtWarpCmdWire = warpCmdWire
    , simtResumeReqs = toStream resumeQueue
    , simtScalarResumeReqs = toStream scalarResumeQueue
    }

  return pipelineOuts.simtMgmtResps

-- | Stack address interleaver so that accesses to same stack
-- offset by different threads in a warp are coalesced
interleaveStacks :: MemReq -> MemReq
interleaveStacks req =
    req { memReqAddr = interleaveAddr req.memReqAddr }
  where
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
