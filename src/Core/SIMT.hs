-- 32-bit SIMT core

module Core.SIMT where

-- SoC configuration & memory map
#include <Config.h>
#include <MemoryMap.h>

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
import Pebbles.Pipeline.SIMT.RegFile
import Pebbles.Pipeline.SIMT.Management
import Pebbles.Pipeline.Interface
import Pebbles.Memory.Interface
import Pebbles.Memory.CapSerDes
import Pebbles.Memory.CoalescingUnit
import Pebbles.Memory.DRAM.Interface
import Pebbles.Instructions.RV32_I
import Pebbles.Instructions.RV32_M
import Pebbles.Instructions.RV32_A
import Pebbles.Instructions.RV32_F
import Pebbles.Instructions.Mnemonics
import Pebbles.Instructions.RV32_IxCHERI
import Pebbles.Instructions.Units.FPU
import Pebbles.Instructions.Units.SFU
import Pebbles.Instructions.Units.MulUnit
import Pebbles.Instructions.Units.DivUnit
import Pebbles.Instructions.Units.BoundsUnit
import Pebbles.Instructions.Units.SharedUnits
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
  , execSFUReqs :: Sink SFUReq
    -- ^ Sink for capability bounds-setting requests
  , execFPUReqs :: Sink FPUReq
    -- ^ Sink for FPU requests
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
          let sfuDivReqs = if SIMTUseSharedDivUnit == 1
                             then Just ins.execSFUReqs else Nothing
          let sfuFPDivSqrtReqs = if SIMTUseSharedFPDivSqrtUnit == 1
                                   then Just ins.execSFUReqs else Nothing
          executeM ins.execMulReqs ins.execDivReqs sfuDivReqs s
          if enCHERI
            then do
              if SIMTUseSharedBoundsUnit == 1
                then do
                  executeIxCHERIWithSharedBoundsUnit
                    (Just ins.execMulReqs) (Just csrUnit)
                    (Just ins.execCapMemReqs) ins.execSFUReqs s
                else do
                  executeIxCHERI (Just ins.execMulReqs) (Just csrUnit)
                                 (Just ins.execCapMemReqs) s
            else do
              executeI (Just ins.execMulReqs) (Just csrUnit)
                       (Just ins.execMemReqs) s
              executeA ins.execMemReqs s
          when (EnableFP == 1) do executeF ins.execFPUReqs sfuFPDivSqrtReqs s
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
  , simtCoreEnableFP :: Bool
    -- ^ Enable floating-point (Zfinx) extension
  , simtCoreDisableHardDSPBlocks :: Bool
    -- ^ Disable DSP blocks
  , simtCoreUseSharedFPDivSqrt :: Bool
    -- ^ Enable shared FP divider and square root
  }

-- | 32-bit SIMT core
makeSIMTCore ::
     SIMTCoreConfig
     -- ^ Configuration parameters
  -> Stream SIMTReq
     -- ^ SIMT management requests
  -> Sink ( SIMTPipelineInstrInfo
          , Vec SIMTLanes (Option MemReq)
          , Option (ScalarVal 33) )
     -- ^ Memory requests
  -> Stream (SIMTPipelineInstrInfo, Vec SIMTLanes (Option MemResp))
     -- ^ Memory responses
  -> DRAMStatSigs
     -- ^ For DRAM stat counters
  -> CoalUnitPerfStats
     -- ^ For coalescing unit stats 
  -> Module (Stream SIMTResp)
     -- ^ SIMT management responses
makeSIMTCore config mgmtReqs memReqs memResps dramStatSigs coalStats = mdo

  -- Scalar unit enabled?
  let enScalarUnit = SIMTEnableScalarUnit == 1

  -- Multiplier requests
  -- ===================

  -- Vector multiplier unit
  vecMulUnit <- makeFullVecMulUnit

  -- Per lane multiplier request sinks
  mulSinks <- makeSinkVectoriser
    (\vec -> (pipelineOuts.simtInstrInfo, vec)) vecMulUnit.reqs

  -- Multiplier for scalar unit
  scalarMulUnit <- if enScalarUnit then makeFullMulUnit else return nullServer

  -- Scalar unit multiplier sink (with pipeline info inserted)
  let scalarMulSink = mapSink ((,) pipelineOuts.simtScalarInstrInfo)
                              scalarMulUnit.reqs

  -- Divider requests
  -- ================

  -- Vector division unit
  vecDivUnit <-
    if SIMTUseSharedDivUnit == 1
      then return nullServer
      else case config.simtCoreUseFullDivider of
             Nothing -> makeSeqVecDivUnit
             Just latency -> makeFullVecDivUnit latency

  -- Per lane divider request sinks
  divSinks <- makeSinkVectoriser
    (\vec -> (pipelineOuts.simtInstrInfo, vec)) vecDivUnit.reqs

  -- Divider for scalar unit
  scalarDivUnit <-
    if enScalarUnit
      then case config.simtCoreUseFullDivider of
             Nothing -> makeSeqDivUnit
             Just latency -> makeFullDivUnit latency
      else return nullServer

  -- Scalar unit divider sink (with pipeline info inserted)
  let scalarDivSink = mapSink ((,) pipelineOuts.simtScalarInstrInfo)
                              scalarDivUnit.reqs

  -- Bounds unit requests
  -- ====================

  let useSFU = SIMTUseSharedDivUnit == 1 ||
                 SIMTUseSharedFPDivSqrtUnit == 1 ||
                   (config.simtCoreEnableCHERI && SIMTUseSharedBoundsUnit == 1)
  sfu <-
    if useSFU
      then do
        sfus <- makeVecSFU
          SFUConfig {
            enDivUnit = SIMTUseSharedDivUnit == 1
          , divLatency = SIMTFullDividerLatency
          , enBoundsUnit = config.simtCoreEnableCHERI &&
                             SIMTUseSharedBoundsUnit == 1
          , enFPUnit = SIMTUseSharedFPDivSqrtUnit == 1
          , disableHardFPBlocks = config.simtCoreDisableHardDSPBlocks
          }
        makeSharedUnits @1 sfus
      else return nullServer

  sfuSinks <- makeSinkVectoriser
    (\vec -> (pipelineOuts.simtInstrInfo, vec)) sfu.reqs

  -- FPU requests
  -- ============

  -- Vector FPU
  vecFPU <-
    if config.simtCoreEnableFP
      then makeVecFPU config.simtCoreDisableHardDSPBlocks
                      (not config.simtCoreUseSharedFPDivSqrt) False
      else return nullServer

  -- Per lane divider request sinks
  fpuSinks <- makeSinkVectoriser
    (\vec -> (pipelineOuts.simtInstrInfo,
                toFPUOpcode pipelineOuts.simtOpcode, vec)) vecFPU.reqs

  -- Memory requests
  -- ===============

  -- Apply stack address interleaving
  let ilv (info, vec, scal) = (info, interleaveStacks info.warpId vec, scal)
  let memReqsIlv = mapSink ilv memReqs

  -- Per lane memory request sinks
  (memReqSinks, capMemReqSinks) <-
    if config.simtCoreEnableCHERI
      then do
        -- Serialise CapMemReq to MemReq
        capMemReqs <- makeCapMemReqSerialiser memReqsIlv
        -- Sink of vectors to vector of sinks
        capMemSinks <-
          makeSinkVectoriser
            (\vec -> (pipelineOuts.simtInstrInfo, vec,
                        pipelineOuts.simtScalarisedOpB))
            capMemReqs
        -- Convert vector to list
        let capMemSinksList = toList capMemSinks
        let memSinks = map (mapSink toCapMemReq) capMemSinksList
        return (memSinks, capMemSinksList)
      else do
        let scVal =
              fmap (\s -> ScalarVal {
                            val = zeroExtend s.val
                          , stride = s.stride
                          , mask = s.mask
                          })
                   pipelineOuts.simtScalarisedOpB.scalarisedVal
        -- Sink of vectors to vector of sinks
        memSinks <- makeSinkVectoriser
            (\vec -> (pipelineOuts.simtInstrInfo, vec, scVal))
          memReqsIlv
        return (toList memSinks, replicate SIMTLanes nullSink)

  -- Responses
  -- =========

  -- Pipeline resume requests from memory
  memResumeReqs <- makeMemRespDeserialiser
                     config.simtCoreEnableCHERI memResps

  -- Merge resume requests
  let resumeReqStream = mergeTree $
           [ vecMulUnit.resps ]
        ++ [ vecDivUnit.resps ]
        ++ [ sfu.resps | useSFU ]
        ++ [ memResumeReqs ]
        ++ [ vecFPU.resps | config.simtCoreEnableFP ] 

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
        , useSharedPCC = SIMTUseFixedPCC == 1
        , decodeStage = concat $
            [ if config.simtCoreEnableCHERI
                then decodeIxCHERI ++ decodeAxCHERI
                else decodeI ++ decodeA
            , decodeM
            , decodeSIMT
            ] ++
            [ decodeF | config.simtCoreEnableFP ]
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
                , execSFUReqs = sfuSink
                , execFPUReqs = fpuSink
                }
            | (memSink, capMemSink, mulSink, divSink, sfuSink, fpuSink, i) <-
                zip7 memReqSinks capMemReqSinks
                     (toList mulSinks) (toList divSinks)
                     (toList sfuSinks) (toList fpuSinks) [0..] ]
        , simtPushTag = SIMT_PUSH
        , simtPopTag = SIMT_POP
        , useRegFileScalarisation = SIMTEnableRegFileScalarisation == 1
        , useCapRegFileScalarisation =
               config.simtCoreEnableCHERI
            && SIMTEnableCapRegFileScalarisation == 1
        , useAffineScalarisation = SIMTEnableAffineScalarisation == 1
        , useScalarUnit = enScalarUnit
        , scalarUnitAllowList =
            [ ADD, SUB, SLT, SLTU, AND, OR, XOR
            , LUI, AUIPC
            , SLL, SRL, SRA
            , JAL, JALR
            , BEQ, BNE, BLT, BLTU, BGE, BGEU
            , MUL, DIV
            , SIMT_PUSH, SIMT_POP
            ] ++ if config.simtCoreEnableCHERI
                   then [ CGetPerm, CGetType, CGetBase, CGetLen
                        , CGetTag, CGetSealed, CGetFlags, CGetAddr
                        , CAndPerm, CSetFlags, CSetAddr, CIncOffset
                        , CSetBounds, CSetBoundsExact, CRRL, CRAM
                        , CMove, CClearTag, CSpecialRW, CSealEntry
                        ]
                   else []
        , scalarUnitDecodeStage = concat
            [ if config.simtCoreEnableCHERI then decodeIxCHERI else decodeI
            , decodeM
            , decodeSIMT
            ]
        , scalarUnitAffineAdd =
            if enScalarUnit && SIMTEnableAffineScalarisation == 1
              then Just ADD else Nothing
        , scalarUnitAffineCMove =
            if config.simtCoreEnableCHERI &&
                 enScalarUnit && SIMTEnableAffineScalarisation == 1
              then Just CMove else Nothing
        , scalarUnitAffineCIncOffset =
            if config.simtCoreEnableCHERI &&
                 enScalarUnit && SIMTEnableAffineScalarisation == 1
              then Just CIncOffset else Nothing
        , scalarUnitExecuteStage = \s -> do
            return
              ExecuteStage {
                execute = do
                  if config.simtCoreEnableCHERI
                    then executeIxCHERI (Just scalarMulSink)
                                        Nothing Nothing s
                    else executeI (Just scalarMulSink)
                                  Nothing Nothing s
                  executeM scalarMulSink scalarDivSink Nothing s
              }
        , regSpillBaseAddr = let a << b = a * 2^b in REG_SPILL_BASE
        , useLRUSpill = SIMTUseLRUSpill == 1
        , useRRSpill = SIMTUseRRSpill == 1
        , useSharedVectorScratchpad =
           SIMTUseSharedVecScratchpad == 1
        , usesCapA = instrsThatUseCapMetaDataA
        , usesCapB = usesCapMetaDataB
        , shareCapSRFPort = SIMTShareCapSRFPort == 1
        }

  -- Pipeline instantiation
  pipelineOuts <- makeSIMTPipeline pipelineConfig
    SIMTPipelineIns {
      simtMgmtReqs = mgmtReqs
    , simtWarpCmdWire = warpCmdWire
    , simtResumeReqs = toStream resumeQueue
    , simtScalarResumeReqs = toStream scalarResumeQueue
    , simtDRAMStatSigs = dramStatSigs
    , simtMemReqs = fromList memReqSinks
    , simtCoalStats = coalStats
    }

  return pipelineOuts.simtMgmtResps

-- | Stack address resolver; interleave stacks so that accesses to
-- same stack offset by different threads in a warp are coalesced
interleaveStacks :: Bit SIMTLogWarps
                 -> Vec SIMTLanes (Option MemReq)
                 -> Vec SIMTLanes (Option MemReq)
interleaveStacks warpId vec = V.fromList
  [ fmap (interleaveStack (fromInteger laneId)) req
  | (req, laneId) <- zip (V.toList vec) [0..] ]
  where
    interleaveStack :: Bit SIMTLogLanes -> MemReq -> MemReq
    interleaveStack laneId req =
        req { memReqAddr = interleaveAddr req.memReqAddr }
      where
        interleaveAddr :: Bit 32 -> Bit 32
        interleaveAddr a =
          if slice @31 @SIMTLogBytesPerStack a .==. ones
            then ones # stackOffset # warpId # laneId # wordOffset
            else a
          where
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
