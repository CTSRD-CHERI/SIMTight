#! /usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Check that last command succeeded
assert() {
  if [ "$2" != "" ]; then
    echo -ne "$2"
  fi
  if [ $1 != 0 ]; then
    echo -e "${RED}FAILED${NC}$4"
    exit -1
  else
    echo -ne "${GREEN}ok${NC}"
  fi
  if [ "$3" != "" ]; then
    echo -ne "$3"
  fi
  echo
}

# Kill simulator if running
cleanup() {
  if [ "$SIM_PID" != "" ]; then
    kill $SIM_PID
  fi
}

# Function to extract stat count from program output
# Parameterised by reduction function
getStat() {
  if grep -E ^$1: $tmpLog > /dev/null; then
    local N=$(grep -E ^$1: $tmpLog | \
      cut -d' ' -f2 | \
      python3 -c "import sys; print($2((int(l, 16) for l in sys.stdin)))")
    echo "$N"
  else
    echo ""
  fi
}

# Function to run app and check success (and emit stats)
getStats() {
  local tmpLog=$1
  local OK=$(grep "Self test: PASSED" $tmpLog)
  local CYCLES=$(getStat "Cycles" "sum")
  local INSTRS=$(getStat "Instrs" "sum")
  local VEC_REGS=$(getStat "MaxVecRegs" "max")
  local CAP_VEC_REGS=$(getStat "MaxCapVecRegs" "max")
  local TOTAL_VEC_REGS=$(getStat "TotalVecRegs" "sum")
  local TOTAL_CAP_VEC_REGS=$(getStat "TotalCapVecRegs" "sum")
  local SCALARISABLE=$(getStat "ScalarisableInstrs" "sum")
  local SCALARISED=$(getStat "ScalarisedInstrs" "sum")
  local RETRIES=$(getStat "Retries" "sum")
  local SUSPS=$(getStat "Susps" "sum")
  local SCALAR_SUSPS=$(getStat "ScalarSusps" "sum")
  local SCALAR_ABORTS=$(getStat "ScalarAborts" "sum")
  local DRAM_ACCS=$(getStat "DRAMAccs" "sum")
  local SB_LOAD_HIT=$(getStat "SBLoadHit" "sum")
  local SB_LOAD_MISS=$(getStat "SBLoadMiss" "sum")
  local SB_CAP_LOAD_HIT=$(getStat "SBCapLoadHit" "sum")
  local SB_CAP_LOAD_MISS=$(getStat "SBCapLoadMiss" "sum")
  local IPC=$(python3 -c "print('%.2f' % (float(${INSTRS}) / ${CYCLES}))")
  local OPTIONAL_STATS=""
  if [ "$VEC_REGS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,MaxVecRegs=$VEC_REGS"
  fi
  if [ "$CAP_VEC_REGS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,MaxCapVecRegs=$CAP_VEC_REGS"
  fi
  if [ "$TOTAL_VEC_REGS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,TotalVecRegs=$TOTAL_VEC_REGS"
  fi
  if [ "$TOTAL_CAP_VEC_REGS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,TotalCapVecRegs=$TOTAL_CAP_VEC_REGS"
  fi
  if [ "$SCALARISABLE" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,ScalarisableInstrs=$SCALARISABLE"
  fi
  if [ "$SCALARISED" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,ScalarisedInstrs=$SCALARISED"
  fi
  if [ "$RETRIES" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,Retries=$RETRIES"
  fi
  if [ "$SUSPS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,Susps=$SUSPS"
  fi
  if [ "$SCALAR_SUSPS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,ScalarSusps=$SCALAR_SUSPS"
  fi
  if [ "$SCALAR_ABORTS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,ScalarAborts=$SCALAR_ABORTS"
  fi
  if [ "$SB_LOAD_HIT" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,SBLoadHit=$SB_LOAD_HIT"
  fi
  if [ "$SB_LOAD_MISS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,SBLoadMiss=$SB_LOAD_MISS"
  fi
  if [ "$SB_CAP_LOAD_HIT" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,SBCapLoadHit=$SB_CAP_LOAD_HIT"
  fi
  if [ "$SB_CAP_LOAD_MISS" != "" ]; then
    OPTIONAL_STATS="$OPTIONAL_STATS,SBCapLoadMiss=$SB_CAP_LOAD_MISS"
  fi
  if [ "$EmitStats" != "" ]; then
    test "$OK" != ""
    assert $? "" " [IPC=$IPC,Instrs=$INSTRS,Cycles=$CYCLES,DRAMAccs=$DRAM_ACCS$OPTIONAL_STATS]"
  else
    test "$OK" != ""
    assert $? "" ""
  fi
}

# Prepare simulator
prepare_sim() {
  echo -n "SIMTight build: "
  make -s -C .. verilog > /dev/null
  assert $?
  echo -n "Simulator build: "
  make -s -C .. sim > /dev/null
  assert $?
  echo -n "Starting simulator: "
  pushd . > /dev/null
  cd ../sim
  if [ "$LogSim" == "" ]; then
    ./sim &
    SIM_PID=$!
  else
    (stdbuf -oL -eL ./sim &> $SIM_LOG_FILE) &
    SIM_PID=$!
  fi
  sleep 1
  popd > /dev/null
  ps -p $SIM_PID > /dev/null
  assert $?
  if [ "$LogSim" != "" ]; then
    echo "Simulator log: $SIM_LOG_FILE"
  fi
  trap cleanup EXIT
  echo
}

# Prepare FPGA
prepare_fpga() {
  # Check that quartus is in scope
  echo -n "Quartus available: "
  JTAG_CABLE=$(type -P quartus)
  assert $?
  # Look for FPGA image
  QPROG="de10-pro"
  if [ "$TestFPGA" == "yup-e" ]; then
    QPROG="de10-pro-e"
  fi
  if [ "$NoPgm" == "" ]; then
    echo -n "FPGA image available: "
    test -f "../$QPROG/output_files/DE10_Pro.sof"
    assert $? "" "" " (run 'make' in $QPROG dir)"
  fi
  # Check that FPGA is visisble
  echo -n "FPGA available: "
  JTAG_CABLE=$(jtagconfig 2> /dev/null | grep DE10-Pro)
  test "$JTAG_CABLE" != ""
  assert $?
  # Program FPGA
  if [ "$NoPgm" == "" ]; then
    echo -n "Programming FPGA: "
    RETRY=0
    while true; do
      make -s -C ../$QPROG download-sof > /dev/null
      if [ "$?" == "0" ]; then
        break
      fi
      if [ "$RETRY" == "5" ]; then
        echo "Failed to program FPGA"
        exit -1
      fi
      RETRY=$(($RETRY+1))
      sleep 10
    done
  fi
  echo
}
