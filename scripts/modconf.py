#!/usr/bin/env python3

# Support scriptable modification of inc/Config.h

import sys
import os

# Get directory containing script
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Get dir containing repo
repoDir = os.path.dirname(scriptDir)

def printUsage():
  print("Usage:")
  print("  modconf.py restore")
  print("  modconf.py set <param> <value>")

# Check args
if len(sys.argv) < 2:
  printUsage()
  sys.exit(-1)

if sys.argv[1] == "restore":
  os.chdir(repoDir)
  os.system("git checkout --quiet inc/Config.h")
elif sys.argv[1] == "set":
  if len (sys.argv) != 4:
    printUsage()
    sys.exit(-1)
  os.chdir(repoDir)
  configFile = open("inc/Config.h")
  configLines = []
  found = False
  for line in configFile:
    if line.startswith("#define " + sys.argv[2]):
      found = True
      configLines.append("#define " + sys.argv[2] + " " + sys.argv[3] + "\n")
    else:
      configLines.append(line)
  configFile.close()
  if not found:
    print("Setting not found:", sys.argv[2])
    sys.exit(-1)
  configFile = open("inc/Config.h", "w")
  for line in configLines:
    configFile.write(line)
  configFile.close()
else:
  print("Unrecognise command:", sys.argv[1])
  sys.exit(-1)
