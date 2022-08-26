#!/usr/bin/env python3

import sys
import os

# Various configs with different settings
config = {}
config["CHERI"] = [
    ("EnableTaggedMem", "1")
  , ("EnableCHERI", "1")
  , ("UseClang", "1")
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

# Combinations of configs that are "interesting"
configCombos = [
    []
  , ["RegFileScalarisation"]
  , ["ScalarUnit"]
  , ["StoreBuffer"]
  , ["ScalarUnit", "StoreBuffer"]
  , ["CHERI"]
  , ["CHERI", "RegFileScalarisation"]
  , ["CHERI", "StoreBuffer"]
  ]

# Get directory containing script
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Get dir containing repo
repoDir = os.path.dirname(scriptDir)

# Test each combination
for combo in configCombos:
  os.chdir(repoDir + "/scripts")
  os.system("make -C .. clean > /dev/null")
  os.system("./modconf.py restore")
  for conf in combo:
    for setting in config[conf]:
      os.system("./modconf.py set " + setting[0] + " " + setting[1])
  os.chdir("../test")
  os.system("echo >> test.log")
  os.system("echo ====== " + "+".join(combo) + " ====== >> test.log")
  os.system("echo >> test.log")
  os.system("./test.sh >> test.log")
