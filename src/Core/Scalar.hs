-- 32-bit scalar core with 5-stage pipeline

module Core.Scalar where

-- Blarney imports
import Blarney
import Blarney.Stream
import Blarney.SourceSink
import Blarney.Interconnect

-- Pebbles imports
import Pebbles.CSRs.Trap
import Pebbles.CSRs.CSRUnit
import Pebbles.CSRs.CycleCount
import Pebbles.CSRs.Custom.Simulate
import Pebbles.CSRs.Custom.UART
import Pebbles.CSRs.Custom.InstrMem
import Pebbles.CSRs.Custom.SIMTHost
import Pebbles.Pipeline.Scalar
import Pebbles.Pipeline.Interface
import Pebbles.Pipeline.SIMT.Management
import Pebbles.Instructions.RV32_I
import Pebbles.Instructions.RV32_M
import Pebbles.Instructions.RV32_xCHERI
import Pebbles.Instructions.Units.MulUnit
import Pebbles.Instructions.Units.DivUnit
import Pebbles.Instructions.Custom.CacheManagement
import Pebbles.Memory.Interface
import Pebbles.Memory.DRAM.Interface

-- CHERI imports
import CHERI.CapLib

-- Haskell imports
import Numeric (showHex)

-- | Configuration parameters
data ScalarCoreConfig =
  ScalarCoreConfig {
    scalarCoreInstrMemInitFile :: Maybe String
    -- ^ Initialisation file for instruction memory
  , scalarCoreInstrMemLogNumInstrs :: Int
    -- ^ Size of tightly coupled instruction memory
  , scalarCoreInitialPC :: Integer
    -- ^ Initial PC
  , scalarCoreEnableRegForwarding :: Bool
    -- ^ Enable register forwading to avoid data hazards
  , scalarCoreEnableCHERI :: Bool
    -- ^ Enable CHERI extensions
  , scalarCoreCapRegInitFile :: Maybe String
    -- ^ File containing initial capability register file (meta-data only)
  }

-- | Scalar core inputs
data ScalarCoreIns =
  ScalarCoreIns {
    scalarUartIn :: Stream (Bit 8)
    -- ^ UART input
  , scalarMemUnit :: MemUnit InstrInfo
    -- ^ Memory unit
  , scalarSIMTResps :: Stream SIMTResp
    -- ^ Management responses from SIMT core
  } deriving (Generic, Interface)

-- | Scalar core outputs
data ScalarCoreOuts =
  ScalarCoreOuts {
    scalarUartOut :: Stream (Bit 8)
    -- ^ UART output
  , scalarSIMTReqs :: Stream SIMTReq
    -- ^ Management requests to SIMT core
  } deriving (Generic, Interface)

-- | RV32IM core with UART input and output channels
makeScalarCore ::
     ScalarCoreConfig
     -- ^ Configuration parameters
  -> ScalarCoreIns
     -- ^ Scalar core inputs
  -> Module ScalarCoreOuts
     -- ^ Scalar core outputs
makeScalarCore config inputs = mdo
  -- UART CSRs
  (uartCSRs, uartOut) <- makeCSRs_UART (inputs.scalarUartIn)

  -- Instruction memory CSRs
  imemCSRs <- makeCSRs_InstrMem (pipeline.writeInstr)

  -- SIMT management CSRs
  (simtReqs, simtCSRs) <- makeCSRs_SIMTHost (inputs.scalarSIMTResps)

  -- Cycle count CSRs
  cycleCSRs <- makeCSR_CycleCount

  -- Trap CSRs
  (trapCSRs, trapRegs) <- makeCSRs_Trap

  -- CSR unit
  csrUnit <- makeCSRUnit $
       csrs_Sim
    ++ uartCSRs
    ++ imemCSRs
    ++ simtCSRs
    ++ cycleCSRs
    ++ trapCSRs
 
  -- Multiplier
  mulUnit <- makeHalfMulUnit

  -- Divider
  divUnit <- makeSeqDivUnit

  -- Memory requests from core
  (memReqSink, capMemReqSink) <-
    if config.scalarCoreEnableCHERI
      then do
        capMemReqSink <- makeCapMemReqSink (inputs.scalarMemUnit.memReqs)
        let memReqSink = mapSink toCapMemReq capMemReqSink
        return (memReqSink, capMemReqSink)
      else return (inputs.scalarMemUnit.memReqs, nullSink)

  -- Pipeline resume requests from memory
  memResumeReqs <- makeMemRespToResumeReq
    (config.scalarCoreEnableCHERI)
    (inputs.scalarMemUnit.memResps)

  -- Processor pipeline
  pipeline <- makeScalarPipeline
    ScalarPipelineConfig {
      instrMemInitFile = config.scalarCoreInstrMemInitFile
    , instrMemLogNumInstrs = config.scalarCoreInstrMemLogNumInstrs
    , enableRegForwarding = config.scalarCoreEnableRegForwarding
    , initialPC = config.scalarCoreInitialPC
    , capRegInitFile = config.scalarCoreCapRegInitFile
    , decodeStage = concat
        [ decodeI
        , if config.scalarCoreEnableCHERI then [] else decodeI_NoCap
        , decodeM
        , decodeCacheMgmt
        , if config.scalarCoreEnableCHERI then decodeCHERI else []
        ]
    , executeStage = \s -> return
        ExecuteStage {
          execute = do
            executeI Nothing csrUnit memReqSink s
            executeM mulUnit divUnit s
            executeCacheMgmt memReqSink s
            if config.scalarCoreEnableCHERI
              then executeCHERI csrUnit capMemReqSink s
              else executeI_NoCap csrUnit memReqSink s
        , resumeReqs = mergeTree
            [ memResumeReqs
            , mulUnit.mulResps
            , divUnit.divResps
            ]
        }
    , trapCSRs = trapRegs
    , checkPCCFunc =
        if config.scalarCoreEnableCHERI then Just checkPCC else Nothing
    }

  return
    ScalarCoreOuts {
      scalarUartOut = uartOut
    , scalarSIMTReqs = simtReqs
    }

-- Register file initialisation
-- ============================

-- | Write initial capability reg file (meta-data only, mif format)
writeScalarCapRegFileMif :: String -> IO ()
writeScalarCapRegFileMif filename =
  writeFile filename $ unlines $
    [ "DEPTH = 32;"
    , "WIDTH = " ++ show (valueOf @CapPipeMetaWidth) ++ ";"
    , "ADDRESS_RADIX = DEC;"
    , "DATA_RADIX = HEX;"
    , "CONTENT"
    , "BEGIN"
    ] ++
    [ show i ++ " : " ++ showHex nullCapPipeMetaInteger ";"
    | i <- [0..31]
    ] ++
    ["END"]

-- | Write initial capability reg file (meta-data only, hex format)
writeScalarCapRegFileHex :: String -> IO ()
writeScalarCapRegFileHex filename =
  writeFile filename $ unlines $
    [ showHex nullCapPipeMetaInteger ""
    | i <- [0..31]
    ]
