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

# --------------------------------------------------------------------
# Clean the cheri tools directory for a new build
# --------------------------------------------------------------------
rm -rf $SCRIPT_DIR/cheri
rm -rf $SCRIPT_DIR/cheribuild
rm -rf $SCRIPT_DIR/llvm-project

# --------------------------------------------------------------------
# Commit info 
# --------------------------------------------------------------------
branch="simtight"
cheribuild_commithash="a1bdfa225b575606f7303435c4c75ea9942d697f"
llvm_commithash="19d402e23fcaa197e1d40547da403dc17e13c7ae"

# --------------------------------------------------------------------
# Clone the submodules with certain commit number
# --------------------------------------------------------------------
cd $SCRIPT_DIR/

# Cheribuild
git clone --recursive https://github.com/CTSRD-CHERI/cheribuild.git
cd cheribuild
git fetch --depth=1 origin $cheribuild_commithash
git checkout $cheribuild_commithash -b $branch
cd ..

# LLVM
git clone --recursive https://github.com/CTSRD-CHERI/llvm-project.git
cd llvm-project
git fetch --depth=1 origin $llvm_commithash
git checkout $llvm_commithash -b $branch
git apply $SCRIPT_DIR/disable-ptr-scev.diff
cd ..

# --------------------------------------------------------------------
# Build cheri tools 
# --------------------------------------------------------------------
cd $SCRIPT_DIR/cheribuild && \
./cheribuild.py llvm \
--llvm/source-directory ../llvm-project \
--source-root ../cheri
