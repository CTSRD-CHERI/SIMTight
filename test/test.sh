#! /usr/bin/env bash

APPS=(
  Samples/VecAdd
  Samples/Histogram
  Samples/Reduce
  Samples/Scan
  Samples/Transpose
  Samples/MatVecMul
  Samples/MatMul
  InHouse/BlockedStencil
  InHouse/StripedStencil
  InHouse/VecGCD
)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Options
# =======

TestSim=
TestFPGA=
NoPgm=
AppsOnly=
LogSim=
EmitStats=

# Arguments
# =========

while :
do
  case $1 in
    -h|--help)
      echo "Run test-suite and example apps"
      echo "  --sim        run in simulation (verilator)"
      echo "  --fpga       run on FPGA (de10-pro)"
      echo "  --no-pgm     don't reprogram FPGA"
      echo "  --apps-only  run apps only (not test-suite)"
      echo "  --log-sim    log simulator output to sim-log.txt"
      echo "  --stats      emit performance stats"
      exit
      ;;
    --sim)
      TestSim=yup
      ;;
    --fpga)
      TestFPGA=yup
      ;;
    --no-pgm)
      NoPgm=yup
      ;;
    --apps-only)
      AppsOnly=yup
      ;;
    --log-sim)
      LogSim=yup
      ;;
    --stats)
      EmitStats=yup
      ;;
    -?*)
      printf 'Ignoring unknown flag: %s\n' "$1" >&2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
  esac
  shift
done

if [ "$TestSim" == "" ] && [ "$TestFPGA" == "" ]; then
  TestSim=yup
fi

WorkingDir=`pwd`

# Helper functions
# ================

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

# Preparation
# ===========

# Prepare simulator
SIM_PID=
SIM_LOG_FILE=sim-log.txt
if [ "$TestSim" != "" ]; then
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
    stdbuf -oL ./sim > $WorkingDir/$SIM_LOG_FILE &
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
fi

# Prepare FPGA
if [ "$TestFPGA" != "" ] ; then
  # Check that quartus is in scope
  echo -n "Quartus available: "
  JTAG_CABLE=$(type -P quartus)
  assert $?
  # Look for FPGA image
  echo -n "FPGA image available: "
  test -f "../de10-pro/output_files/DE10_Pro.sof"
  assert $? "" "" " (run 'make' in 'de10-pro' dir)"
  # Check that FPGA is visisble
  echo -n "FPGA available: "
  JTAG_CABLE=$(jtagconfig 2> /dev/null | grep DE10-Pro)
  test "$JTAG_CABLE" != ""
  assert $?
  # Program FPGA
  if [ "$NoPgm" == "" ]; then
    echo -n "Programming FPGA: "
    make -s -C ../de10-pro download-sof > /dev/null
    assert $?
  fi
  echo
fi

# Test Suite
# ==========

# In simulation
if [[ "$TestSim" != "" && "$AppsOnly" == "" ]]; then
  echo "Test Suite (CPU, Simulation)"
  echo "============================"
  echo
  make -s -C ../apps/TestSuite test-cpu-sim
  assert $? "\nSummary: "
  echo
  echo "Test Suite (SIMT Core, Simulation)"
  echo "=================================="
  echo
  make -s -C ../apps/TestSuite test-simt-sim
  assert $? "\nSummary: "
  echo
fi

# On FPGA
if [[ "$TestFPGA" != "" && "$AppsOnly" == "" ]] ; then
  echo "Test Suite (CPU, FPGA)"
  echo "======================"
  echo
  make -s -C ../apps/TestSuite test-cpu
  assert $? "\nSummary: "
  echo
  echo "Test Suite (SIMT Core, FPGA)"
  echo "============================"
  echo
  make -s -C ../apps/TestSuite test-simt
  assert $? "\nSummary: "
  echo
fi

# Sample Apps
# ===========

# Function to run app and check success (and emit stats)
checkApp() {
  local Run=$1
  local APP=$2
  local tmpDir=$3
  local APP_MANGLED=$(echo $APP | tr '/' '-')
  local tmpLog=$tmpDir/$APP_MANGLED.log
  $(cd ../apps/$APP && $Run > $tmpLog)
  local OK=$(grep "Self test: PASSED" $tmpLog)
  local CYCLES=$(grep -E ^Cycles: $tmpLog | cut -d' ' -f2)
  local INSTRS=$(grep -E ^Instrs: $tmpLog | cut -d' ' -f2)
  local VEC_REGS=$(grep -E ^MaxVecRegs: $tmpLog | cut -d' ' -f2)
  local CAP_VEC_REGS=$(grep -E ^MaxCapVecRegs: $tmpLog | cut -d' ' -f2)
  local DCYCLES=$(python -c "print('%d' % (0x${CYCLES}))")
  local DINSTRS=$(python -c "print('%d' % (0x${INSTRS}))")
  local IPC=$(python -c "print('%.2f' % (float(0x${INSTRS}) / 0x${CYCLES}))")
  if [ "$VEC_REGS" != "" ]; then
    local DVEC_REGS=$(python -c "print('%d' % (0x${VEC_REGS}))")
    VEC_REGS=",VecRegs=$DVEC_REGS"
  fi
  if [ "$CAP_VEC_REGS" != "" ]; then
    local DCAP_VEC_REGS=$(python -c "print('%d' % (0x${CAP_VEC_REGS}))")
    CAP_VEC_REGS=",CapVecRegs=$DCAP_VEC_REGS"
  fi
  local OPTIONAL_STATS="$VEC_REGS$CAP_VEC_REGS"
  if [ "$EmitStats" != "" ]; then
    test "$OK" != ""
    assert $? "" " [IPC=$IPC,Instrs=$DINSTRS,Cycles=$DCYCLES$OPTIONAL_STATS]"
  else
    test "$OK" != ""
    assert $? "" ""
  fi
}

# In simulation
if [ "$TestSim" != "" ]; then
  echo "Apps (Simulation)"
  echo "================="
  echo
  tmpDir=$(mktemp -d -t simtight-test-XXXX)
  for APP in ${APPS[@]}; do
    echo -n "$APP (build): "
    make -s -C ../apps/$APP RunSim
    assert $?
    echo -n "$APP (run): "
    checkApp ./RunSim $APP $tmpDir
  done
fi

# On FPGA
if [ "$TestFPGA" != "" ] ; then
  echo "Apps (FPGA)"
  echo "==========="
  echo
  tmpDir=$(mktemp -d -t simtight-test-XXXX)
  for APP in ${APPS[@]}; do
    echo -n "$APP (build): "
    make -s -C ../apps/$APP Run
    assert $?
    echo -n "$APP (run): "
    checkApp ./Run $APP $tmpDir
  done
fi

echo
echo "All tests passed"
