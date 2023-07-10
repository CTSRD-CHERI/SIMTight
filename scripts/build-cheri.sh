#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script builds the Cheri-LLVM library
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SIMTIGHT=${SCRIPT_DIR}/..

cd $SIMTIGHT/cheri-tools/cheribuild && \
./cheribuild.py sdk-riscv64-purecap \
--llvm/source-directory ../llvm-project \
--cheribsd/source-directory ../cheribsd \
--gdb/source-directory ../gdb \
--run/custom-qemu-path ../qemu \
--source-root ../cheri
