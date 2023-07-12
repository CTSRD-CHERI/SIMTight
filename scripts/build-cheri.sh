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
rm -rf $SIMTIGHT/cheri-tools/*

# --------------------------------------------------------------------
# Commit info 
# --------------------------------------------------------------------
branch="simtight"
cheribuild_commithash="a1bdfa225b575606f7303435c4c75ea9942d697f"
llvm_commithash="19d402e23fcaa197e1d40547da403dc17e13c7ae"
gdb_commithash="f7eb074720f52900ef0718e2f3ca3b6ecdcfea52"
qemu_commithash="01a05aaf95bcbc9eba282839de18b2f6713429eb"
cheribsd_commithash="52b88aac0808acf4989ddd4f27aedfc0ad47d538"

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

# gdb 
git clone --recursive git@github.com:CTSRD-CHERI/gdb.git
cd gdb
git fetch --depth=1 origin $gdb_commithash
git checkout $gdb_commithash -b $branch
cd ..

# qemu 
git clone --recursive git@github.com:CTSRD-CHERI/qemu.git
cd qemu
git fetch --depth=1 origin $qemu_commithash
git checkout $qemu_commithash -b $branch
cd ..

# cheribsd 
git clone --recursive git@github.com:CTSRD-CHERI/cheribsd.git
cd cheribsd
git fetch --depth=1 origin $cheribsd_commithash
git checkout $cheribsd_commithash -b $branch
cd ..

# --------------------------------------------------------------------
# Build cheri tools 
# --------------------------------------------------------------------
cd $SIMTIGHT/cheri-tools/cheribuild && \
./cheribuild.py sdk-riscv64-purecap \
--llvm/source-directory ../llvm-project \
--cheribsd/source-directory ../cheribsd \
--gdb/source-directory ../gdb \
--run/custom-qemu-path ../qemu \
--source-root ../cheri
