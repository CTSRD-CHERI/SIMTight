#! /usr/bin/env bash

source test-lib.sh

APPS=(
  Samples/VecAdd
  Samples/Histogram
  Samples/Reduce
  Samples/Scan
  Samples/Transpose
  Samples/MatVecMul
  Samples/MatMul
  Samples/BitonicSortSmall
  Samples/BitonicSortLarge
  Samples/SparseMatVecMul
  InHouse/BlockedStencil
  InHouse/StripedStencil
  InHouse/VecGCD
  InHouse/MotionEst
)

# Options
# =======

TestSim=
TestFPGA=
NoPgm=
AppsOnly=
LogSim=
EmitStats=
SkipTests=
SkipCPU=

# Arguments
# =========

while :
do
  case $1 in
    -h|--help)
      echo "Run test-suite and example apps"
      echo "  --sim         run in simulation (verilator)"
      echo "  --fpga-d      run on FPGA (de10-pro revD)"
      echo "  --fpga-e      run on FPGA (de10-pro revE)"
      echo "  --no-pgm      don't reprogram FPGA"
      echo "  --apps-only   run apps only (not test-suite)"
      echo "  --log-sim     log simulator output to sim-log.txt"
      echo "  --stats       emit performance stats"
      echo "  --skip-tests  skip riscv-tests"
      echo "  --skip-cpu    skip CPU testing"
      exit
      ;;
    --sim)
      TestSim=yup
      ;;
    --fpga-d)
      TestFPGA=yup-d
      ;;
    --fpga-e)
      TestFPGA=yup-e
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
    --skip-tests)
      SkipTests=yup
      ;;
    --skip-cpu)
      SkipCPU=yup
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

# Preparation
# ===========

# Prepare simulator
SIM_PID=
SIM_LOG_FILE=`pwd`/sim-log.txt
if [ "$TestSim" != "" ]; then
  prepare_sim
fi

# Prepare FPGA
if [ "$TestFPGA" != "" ]; then
  prepare_fpga
fi

# Test Suite
# ==========

if [ "$SkipTests" == "" ]; then

  # In simulation
  if [[ "$TestSim" != "" && "$AppsOnly" == "" ]]; then
    if [ "$SkipCPU" == "" ]; then
      echo "Test Suite (CPU, Simulation)"
      echo "============================"
      echo
      make -s -C ../apps/TestSuite test-cpu-sim
      assert $? "\nSummary: "
      echo
    fi
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
  getStats $tmpLog
}

# In simulation
if [ "$TestSim" != "" ]; then
  echo "Apps (Simulation)"
  echo "================="
  echo
  if [ "$EmitStats" != "" ]; then
    echo "(NOTE: simulation stats can be misleading due to tiny workloads)"
    echo
  fi
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
