module Util.ClockCross where

-- Blarney imports
import Blarney
import Blarney.Stream
import Blarney.Core.BV

-- | Avalon stream clock crosser inputs
data AvalonStreamCrossIns w =
  AvalonStreamCrossIns {
    crossClkIn :: Bit 1
    -- ^ Input stream clock
  , crossRstIn :: Bit 1
    -- ^ Input stream reset
  , crossClkOut :: Bit 1
    -- ^ Output stream clock
  , crossRstOut :: Bit 1
    -- ^ Output stream reset
  , crossValidIn :: Bit 1
    -- ^ Input stream valid signal
  , crossDataIn :: Bit w
    -- ^ Input stream data signal
  , crossReadyOut :: Bit 1
    -- ^ Output stream ready signal
  }

-- | Avalon stream clock crosser outputs
data AvalonStreamCrossOuts w =
  AvalonStreamCrossOuts {
    crossReadyIn :: Bit 1
    -- ^ Input stream ready signal
  , crossValidOut :: Bit 1
    -- ^ Output streaam valid signal
  , crossDataOut :: Bit w
    -- ^ Output streaam data signal
  }

-- | Avalon stream clock crossing primtive
avalonStreamCross ::
     Int
  -> AvalonStreamCrossIns w
  -> AvalonStreamCrossOuts w
avalonStreamCross width ins =
  AvalonStreamCrossOuts {
    crossReadyIn = FromBV in_ready
  , crossValidOut = FromBV out_valid
  , crossDataOut = FromBV out_data
  }
  where
    custom =
      Custom {
        customName = "AvalonStreamClockCrosser"
      , customInputs =
          [ ("in_clk", 1)
          , ("in_reset", 1)
          , ("out_clk", 1)
          , ("out_reset", 1)
          , ("in_valid", 1)
          , ("in_data", width)
          , ("out_ready", 1)
          ]
      , customOutputs =
          [ ("in_ready", 1)
          , ("out_valid", 1)
          , ("out_data", width)
          ]
      , customParams =
          [ "DATA_WIDTH" :-> show width
          ]
      , customIsClocked = False
      , customResetable = False
      , customNetlist   = Nothing
      }

    [in_ready, out_valid, out_data] =
      makePrim custom
        [ ins.crossClkIn.toBV
        , ins.crossRstIn.toBV
        , ins.crossClkOut.toBV
        , ins.crossRstOut.toBV
        , ins.crossValidIn.toBV
        , ins.crossDataIn.toBV
        , ins.crossReadyOut.toBV
        ]
        [Just "in_ready", Just "out_valid", Just "out_data"]

-- | Stream clock crosser
makeAvalonStreamClockCrosser :: forall a. Bits a =>
     (Clock, Reset)
     -- ^ Input stream clock & reset
  -> (Clock, Reset)
     -- ^ Output stream clock & reset
  -> Stream a
     -- ^ Input stream
  -> Module (Stream a)
     -- ^ Output stream
makeAvalonStreamClockCrosser
    (Clock clkIn, Reset rstIn)
    (Clock clkOut, Reset rstOut)
    streamIn = do

  consumeWire :: Wire (Bit 1) <- makeWire false

  let width = sizeOf (undefined :: a)

  let outStream = avalonStreamCross width
        AvalonStreamCrossIns {
          crossClkIn = clkIn
        , crossRstIn = rstIn
        , crossClkOut = clkOut
        , crossRstOut = rstOut
        , crossValidIn = streamIn.canPeek
        , crossDataIn = streamIn.peek.pack
        , crossReadyOut = consumeWire.val
        }

  always do
    when (streamIn.canPeek .&&. outStream.crossReadyIn) do
      streamIn.consume

  return
    Source {
      consume = do consumeWire <== true
    , canPeek = outStream.crossValidOut
    , peek = outStream.crossDataOut.unpack
    }
