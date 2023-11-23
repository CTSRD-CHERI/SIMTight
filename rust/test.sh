#! /usr/bin/env bash

source ../test/test-lib.sh

APPS=(
  vec_add
  histogram
  reduce
  scan
  transpose
  mat_vec_mul
  mat_mul
  bitonic_sort_small
)

# Options
# =======

TestSim=
TestFPGA=
NoPgm=
LogSim=
EmitStats=

while :
do
  case $1 in
    -h|--help)
      echo "Run examples"
      echo "  --sim         run in simulation (verilator)"
      echo "  --fpga-d      run on FPGA (de10-pro revD)"
      echo "  --fpga-e      run on FPGA (de10-pro revE)"
      echo "  --no-pgm      don't reprogram FPGA"
      echo "  --log-sim     log simulator output to sim-log.txt"
      echo "  --stats       emit performance stats"
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

# Preparation
# ===========

# Prepare simulator
SIM_PID=
SIM_LOG_FILE=sim-log.txt
if [ "$TestSim" != "" ]; then
  prepare_sim
fi

# Prepare FPGA
if [ "$TestFPGA" != "" ]; then
  prepare_fpga
fi

# Examples
# ========

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
  make clean 2> /dev/null
  if [ "$EmitStats" != "" ]; then
    echo "(NOTE: simulation stats can be misleading due to tiny workloads)"
    echo
  fi
  tmpDir=$(mktemp -d -t simtight-test-XXXX)
  for APP in ${APPS[@]}; do
    echo -n "$APP (build): "
    make EXAMPLE=$APP -s 2> /dev/null
    assert $?
    echo -n "$APP (run): "
    tmpLog=$tmpDir/$APP.log
    make EXAMPLE=$APP -s run-sim > $tmpLog 2> /dev/null
    getStats $tmpLog
  done
fi

# On FPGA
if [ "$TestFPGA" != "" ] ; then
  echo "Apps (FPGA)"
  echo "==========="
  echo
  make clean 2> /dev/null
  tmpDir=$(mktemp -d -t simtight-test-XXXX)
  for APP in ${APPS[@]}; do
    echo -n "$APP (build): "
    make EXAMPLE=$APP FEATURES=large_data_set -s 2> /dev/null
    assert $?
    echo -n "$APP (run): "
    make EXAMPLE=$APP FEATURES=large_data_set -s run > $tmpLog 2> /dev/null
    getStats $tmpLog
  done
fi

echo
echo "All tests passed"
