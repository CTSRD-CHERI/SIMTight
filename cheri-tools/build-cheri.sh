#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script builds the Cheri-LLVM library
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# --------------------------------------------------------------------
# The absolute path to the directory of this script.
# --------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SIMTIGHT=${SCRIPT_DIR}/..

# --------------------------------------------------------------------
# Clean the cheri tools directory for a new build
# --------------------------------------------------------------------
mkdir -p $SIMTIGHT/cheri-tools
rm -rf $SIMTIGHT/cheri-tools/cheri
rm -rf $SIMTIGHT/cheri-tools/cheribuild
rm -rf $SIMTIGHT/cheri-tools/llvm-project

# --------------------------------------------------------------------
# Commit info 
# --------------------------------------------------------------------
branch="simtight"
cheribuild_commithash="a1bdfa225b575606f7303435c4c75ea9942d697f"
llvm_commithash="19d402e23fcaa197e1d40547da403dc17e13c7ae"

# --------------------------------------------------------------------
# Clone the submodules with certain commit number
# --------------------------------------------------------------------
cd $SIMTIGHT/cheri-tools

# Cheribuild
git clone --recursive git@github.com:CTSRD-CHERI/cheribuild.git
cd cheribuild
git fetch --depth=1 origin $cheribuild_commithash
git checkout $cheribuild_commithash -b $branch
cd ..

# LLVM
git clone --recursive git@github.com:CTSRD-CHERI/llvm-project.git
cd llvm-project
git fetch --depth=1 origin $llvm_commithash
git checkout $llvm_commithash -b $branch
cd ..

# --------------------------------------------------------------------
# Build cheri tools 
# --------------------------------------------------------------------
cd $SIMTIGHT/cheri-tools/cheribuild && \
./cheribuild.py llvm \
--llvm/source-directory ../llvm-project \
--source-root ../cheri
