#!/usr/bin/env python3

# Take a sequence of traces enabled by EnableCapRegFileTrace and
# determine if capability meta-data can be scalarised.

import subprocess
import sys

# Check args
if len(sys.argv) != 2:
  print("Usage: scalarise.py [FILE]")
  sys.exit()

# Open trace(s)
if sys.argv[1] == "-":
  f = sys.stdin
else:
  f = open(sys.argv[1], 'r')
  if f == None:
    print("File not found: ", sys.argv[2])
    sys.exit()

# Constants
NumWarps = 64
WarpSize = 32
NullCap = 0x1f690003f0

# Helper function
def allSame(items):
    return all(x == items[0] for x in items)

# Parse trace(s)
def parseTraces():
  traces = []
  currentTrace = []
  currentTime = 0
  for line in f:
    # Strip newline
    line = line.strip()
    # Split line into fields
    fields = line.split()
    # Ignore irrelevant trace lines
    if fields[0] != "[CapRegFileTrace]": continue
    # Construct trace record
    record = {}
    record["op"] = fields[1]
    for field in fields[2:]:
      (key, val) = field.split("=")
      record[key] = int(val, 0)
    # Look for start of new trace
    if record["time"] < currentTime:
      currentTime = 0
      traces.append(currentTrace)
      currentTrace = [record]
    else:
      currentTime = record["time"]
      currentTrace.append(record)
  if currentTrace: traces.append(currentTrace)
  return traces

# Check that reg file loads always read the same capability meta-data
def simpleScalarise(trace):
  # Per-thread register file
  regFile = [ [ [ 0 for reg in range(0,32) ]
                for lane in range(0, WarpSize) ]
              for warp in range(0, NumWarps) ]
  capRegFile = [ [ [ NullCap for reg in range(0,32) ]
                   for lane in range(0, WarpSize) ]
                 for warp in range(0, NumWarps) ]
  # PCs where we've seen cap meta-data divergence
  divergingPCs = []
  # Execute trace, looking for loss of scalarisation
  for rec in trace:
    if rec["op"] == "write" or rec["op"] == "resume":
      regFile[rec["warp"]][rec["lane"]][rec["rd"]] = rec["addr"]
      capRegFile[rec["warp"]][rec["lane"]][rec["rd"]] = rec["cap"]
    elif rec["op"] == "read":
      mask = rec["active"]
      for src in ["rs1", "rs2"]:
        reg = rec[src]
        if reg != 0:
          caps = []
          addrs = []
          for lane in range(0, WarpSize):
            if mask & (2**lane):
              caps.append(capRegFile[rec["warp"]][lane][rec[src]])
              addrs.append(regFile[rec["warp"]][lane][rec[src]])
          if not allSame(caps) and rec["pc"] not in divergingPCs:
            print("Cap meta-data divergence for", src,
                  "at pc", hex(rec["pc"]))
            print(rec)
            for (cap, addr) in zip(caps, addrs):
              print(hex(cap), hex(addr))
            divergingPCs.append(rec["pc"])

# Check that reg file loads always read the same capability meta-data,
# while maitaining only a per-warp (not per-thread) capability meta-data
# register file
def scalarise(trace):
  # Per-warp scalar register file
  scalarRegFile = [ [NullCap for reg in range(0, 32)]
                      for warp in range(0, NumWarps) ]
  uniformMask = [ [2**WarpSize-1 for reg in range(0, 32)]
                      for warp in range(0, NumWarps) ]
  # PCs where we've seen cap meta-data divergence
  divergingPCs = []
  # Execute trace, looking for loss of scalarisation
  for rec in trace:
    if rec["op"] == "write" or rec["op"] == "resume":
      if rec["rd"] != 0:
        currentVal = scalarRegFile[rec["warp"]][rec["rd"]]
        if rec["cap"] == currentVal:
          uniformMask[rec["warp"]][rec["rd"]] |= 2 ** rec["lane"]
        else:
          scalarRegFile[rec["warp"]][rec["rd"]] = rec["cap"]
          uniformMask[rec["warp"]][rec["rd"]] = 2 ** rec["lane"]
    elif rec["op"] == "read":
      mask = rec["active"]
      for src in ["rs1", "rs2"]:
        reg = rec[src]
        if reg != 0:
          if (mask & uniformMask[rec["warp"]][rec[src]]) != mask:
            if rec["pc"] not in divergingPCs:
              print("Cap meta-data divergence for", src,
                    "at pc", hex(rec["pc"]))
              divergingPCs.append(rec["pc"])

count = 0
for trace in parseTraces():
  print("Trace number", count)
  simpleScalarise(trace)
  print("  OK")
  count += 1
