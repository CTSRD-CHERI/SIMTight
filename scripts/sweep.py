#!/usr/bin/env python3

import sys
import os

def printUsage():
  print("Usage:")
  print("  sweep.py")
  print("    test          Test each config in simulation")
  print("    synth         Synthesise each config using quartus DSE")
  print("    bench         Run benchmarks on FPGA")

# Check args
if len(sys.argv) < 2:
  printUsage()
  sys.exit(-1)

# Various configs with different settings
config = {}
config["GCC"] = []
config["Clang"] = [("UseClang", "1")]
config["CHERI"] = config["Clang"] + [
    ("EnableTaggedMem", "1")
  , ("EnableCHERI", "1")
  ]
config["RegFileScalarisation"] = [
    ("SIMTEnableRegFileScalarisation", "1")
  , ("SIMTEnableAffineScalarisation", "1")
  , ("SIMTEnableCapRegFileScalarisation", "1")
  ]
config["ScalarUnit"] = config["RegFileScalarisation"] + [
    ("SIMTEnableScalarUnit", "1")
  ]
config["StoreBuffer"] = config["RegFileScalarisation"] + [
    ("SIMTEnableSVStoreBuffer", "1")
  ]
config["DynRegSpill256"] = config["RegFileScalarisation"] + [
    ("SIMTRegFileSize", "256")
  , ("SIMTCapRegFileSize", "256")
  , ("SIMTUseSharedVecScratchpad", "1")
  , ("SIMTUseLRUSpill", "1")
  ]
config["DynRegSpill512"] = config["RegFileScalarisation"] + [
    ("SIMTRegFileSize", "512")
  , ("SIMTCapRegFileSize", "512")
  , ("SIMTUseSharedVecScratchpad", "1")
  , ("SIMTUseLRUSpill", "1")
  ]
config["DynRegSpill768"] = config["RegFileScalarisation"] + [
    ("SIMTRegFileSize", "768")
  , ("SIMTCapRegFileSize", "768")
  , ("SIMTUseSharedVecScratchpad", "1")
  , ("SIMTUseLRUSpill", "1")
  ]
config["DynRegSpill1024"] = config["RegFileScalarisation"] + [
    ("SIMTRegFileSize", "1024")
  , ("SIMTCapRegFileSize", "1024")
  , ("SIMTUseSharedVecScratchpad", "1")
  , ("SIMTUseLRUSpill", "1")
  ]
config["DynRegSpill2047"] = config["RegFileScalarisation"] + [
    ("SIMTRegFileSize", "2047")
  , ("SIMTCapRegFileSize", "2047")
  , ("SIMTUseSharedVecScratchpad", "1")
  , ("SIMTUseLRUSpill", "1")
  ]
config["StaticHalfRF"] = [
    ("UseRV32E", "1")
  ]
config["BaselineProfile"] = (
    config["Clang"]
  + config["DynRegSpill768"]
  + [ ("SIMTUseSharedFPDivSqrtUnit", "1")
    ]
  )
config["CHERIBaselineProfile"] = (
    config["CHERI"]
  + [ ("SIMTUseSharedFPDivSqrtUnit", "1")
    , ("SIMTEnableRegFileScalarisation", "1")
    , ("SIMTEnableAffineScalarisation", "1")
    , ("SIMTRegFileSize", "768")
    , ("SIMTUseLRUSpill", "1")
    ]
  )
config["CHERIProfile"] = (
    config["CHERIBaselineProfile"]
  + [ ("SIMTEnableCapRegFileScalarisation", "1")
    , ("SIMTCapRegFileSize", "768")
    , ("SIMTUseSharedVecScratchpad", "1")
    , ("SIMTCapRFUseInitValOpt", "1")
    , ("SIMTShareCapSRFPort", "1")
    , ("SIMTUseFixedPCC", "1")
    , ("SIMTUseSharedBoundsUnit", "1")
    ]
  )
config["StackCache"] = [("SIMTEnableSVStoreBuffer", "1")]
config["RawFP"] = [("EnableFP", "1"), ("DisableHardDSPBlocks", "1")]

# Combinations of configs that are of interest
configCombos = [
#  ["GCC"],
#  ["GCC", "StoreBuffer", "DynRegSpill2047"],
#  ["GCC", "StoreBuffer", "DynRegSpill2047", "ScalarUnit"],
#  ["BaselineProfile"],
#  ["CHERIProfile"],
#  ["BaselineProfile", "StackCache"],
#  ["CHERIProfile", "StackCache"],
   ["CHERIProfile", "StackCache"] +
     (["RawFP"] if sys.argv[1] == "synth" else []),
   ["CHERIBaselineProfile", "StackCache"] +
     (["RawFP"] if sys.argv[1] == "synth" else []),
   ["BaselineProfile", "StackCache"] +
     (["RawFP"] if sys.argv[1] == "synth" else []),
]

# Config combos of interest when benchmarking only
benchCombos = [
#  ["GCC", "RegFileScalarisation", "DynRegSpill1024"],
#  ["GCC", "RegFileScalarisation", "DynRegSpill512"],
#  ["GCC", "RegFileScalarisation", "DynRegSpill256"],
#  ["GCC", "StaticHalfRF"]
]

# Get directory containing script
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Get dir containing repo
repoDir = os.path.dirname(scriptDir)

# Reset the repo to a clean state
def clean():
  os.chdir(repoDir)
  os.system("make clean > /dev/null")
  os.system("git checkout --quiet de10-pro/DE10_Pro.qsf")
  os.system("git checkout --quiet inc/Config.h")

# Apply settings to config file
def applySettings(combo):
  os.chdir(repoDir + "/scripts")
  for conf in combo:
    for setting in config[conf]:
      os.system("./modconf.py set " + setting[0] + " " + setting[1])

if sys.argv[1] == "test":
  # Remove old log file
  os.chdir(repoDir + "/test")
  os.system("rm -f test.log")
  # Test each combination in simulation
  for combo in configCombos:
    name = "Baseline" if combo == [] else "+".join(combo)
    print("Config: " + name)
    clean()
    applySettings(combo)
    os.chdir(repoDir + "/test")
    os.system("echo >> test.log")
    os.system("echo ====== " + name + " ====== >> test.log")
    os.system("echo >> test.log")
    os.system("./test.sh >> test.log")
elif sys.argv[1] == "synth":
  # Remove old log file
  os.chdir(repoDir + "/de10-pro")
  os.system("rm -f synth.log")
  os.system("rm -rf Synth-*")
  # Synthesise each combination using quartus DSE
  for combo in configCombos:
    name = "Baseline" if combo == [] else "+".join(combo)
    print("Config: " + name)
    clean()
    applySettings(combo)
    # Run DSE
    os.chdir(repoDir + "/src")
    os.system("make > /dev/null")
    os.chdir(repoDir + "/de10-pro")
    os.system("./prepare_dse.sh")
    os.system('make many > /dev/null')
    # Save report
    os.system("echo >> synth.log")
    os.system("echo ====== " + name + " ====== >> synth.log")
    os.system("echo >> synth.log")
    os.system("make report | grep -v 'Info:' >> synth.log")
    os.system("mkdir -p Synth-" + name)
    os.system("find dse/*" +
              " | grep DE10_Pro.fit.place.rpt" +
              " | xargs -i cp --parents {} Synth-" + name)
  clean()
elif sys.argv[1] == "bench":
  # Remove old log file
  os.chdir(repoDir + "/test")
  os.system("rm -f bench.log")
  # Benchmark each combination in simulation
  for combo in (benchCombos + configCombos):
    name = "Baseline" if combo == [] else "+".join(combo)
    print("Config: " + name)
    clean()
    applySettings(combo)
    # Synthesise
    os.chdir(repoDir + "/src")
    os.system("make > /dev/null")
    os.chdir(repoDir + "/de10-pro") # Default to revD FPGA
    os.system("make one > /dev/null")
    # Run benchmarks
    os.chdir(repoDir + "/test")
    # Default to revD FPGA
    os.system("./test.sh --fpga-d --stats --apps-only >> " + name + ".bench")
else:
  printUsage()
  sys.exit(-1)
