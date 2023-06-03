#!/usr/bin/env python3

import re
import sys

def printUsage():
  print("Usage: getstat.py [STATNAME] [FILE1] [FILE2]")

def getStats(filename, statname):
  f = open(filename, "r")
  stats = []
  for line in f:
    result = re.search("[\[,]" + statname + "=[\d\.]+", line)
    dropn = len(sys.argv[1]) + 2
    if result:
      stat = result.group()[dropn:]
      print(stat)
      stats.append(float(stat))
  f.close()
  return stats

# Check args
numArgs = len(sys.argv)
if numArgs != 3 and numArgs != 4:
  printUsage()
  sys.exit(-1)

stats1 = getStats(sys.argv[2], sys.argv[1])
print("Mean: ", sum(stats1) / len(stats1))

if numArgs == 4:
  stats2 = getStats(sys.argv[3], sys.argv[1])
  print("Mean: ", sum(stats2) / len(stats2))
  overheads=[]
  for (s1, s2) in zip(stats1, stats2):
    o = 1-(s1/s2)
    print("%.2f" % o)
    overheads.append(o)
  print("Mean: ", sum(overheads) / len(overheads))
