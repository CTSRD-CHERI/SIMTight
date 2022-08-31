<div class="title-block" style="text-align: center;" align="center">

<h1>SIMTight: a CHERI-enabled GPGPU

<br>

<p>Supported by

<p><img src="doc/UKRI_Logo.svg" width="250"><br>

<p>Digit Security by Design (DSbD) Programme

</div>

## Overview 

SIMTight is an efficient FPGA implementation of the _Single
Instruction Multiple Threads (SIMT)_ model popularised by NVIDIA GPUs.

Features:

  * RISC-V instruction set (RV32IMAxCHERI) 
  * Low-area design with high IPC on classic GPGPU workloads
  * Strong [CHERI](http://cheri-cpu.org) memory safety and isolation
  * Dynamic scalarisation (automatic detection of scalar
    behaviour in hardware)
  * Parallel scalar/vector pipelines, exploiting scalarisation for
    increased throughput
  * Register file and store buffer compression, exploiting scalarisation for
    reduced onchip storage and energy
  * Eliminates register size and spill overhead of CHERI almost entirely
  * Runs [CUDA-like C++ library](doc/NoCL.md) and [benchmark suite](apps/)
    (in pure capability mode)
  * Implemented in Haskell using the
    [Blarney](https://github.com/blarney-lang/blarney)
    hardware description library
  * Modular separation of instruction set and pipelines using the
    [Pebbles](//github.com/blarney-lang/pebbles)
    framework

SIMTight is being developed on the [CAPcelerate
project](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V000381/1),
part of the UKRI's Digital Security by Design programme.

## Default SoC

The default SIMTight SoC consists of a host CPU and a 32-lane
64-warp GPGPU sharing DRAM, both supporting the CHERI-RISC-V ISA.

<img src="doc/SoC.svg" width="450">

A [sample project](de10-pro/) is included for the
[DE10-Pro revD](http://de10-pro.terasic.com) FPGA development board.

## Build instructions

We'll need Verilator, a RISC-V compiler, and GHC 9.2.1 or later.

On Ubuntu 20.04, we can do:

```sh
$ sudo apt install verilator
$ sudo apt install gcc-riscv64-unknown-elf
```

For GHC 9.2.1 or later, [ghcup](https://www.haskell.org/ghcup/) can be
used.

Now, we recursively clone the repo:

```sh
$ git clone --recursive https://github.com/CTSRD-CHERI/SIMTight
```

Inside the repo, there are various things to try.  For example, to
build and run the SIMTight simulator:

```sh
$ cd sim
$ make
$ ./sim
```

While the simulator is running, we can build and run the test suite
in a separate terminal:

```sh
$ cd apps/TestSuite
$ make test-cpu-sim     # Run on the CPU
$ make test-simt-sim    # Run on the SIMT core
```

Alternatively, we can run one of the SIMT kernels:

```sh
$ cd apps/Samples/Histogram
$ make RunSim
$ ./RunSim
```

To run all tests and benchmarks, we can use the [test](test/test.sh)
script.  This script will launch the simulator automatically, so we
first make sure it's not already running.

```sh
$ killall sim
$ cd test
$ ./test.sh            # Run in simulation
```

To build an FPGA image for the [DE10-Pro
revD](http://de10-pro.terasic.com) board (Quartus 21.3pro or later
recommended):

```sh
$ cd de10-pro
$ make                 # Assumes quartus is in your PATH
$ make download-sof    # Assumes DE10-Pro is connected via USB
```

We can now run a SIMT kernel on FPGA:

```sh
$ cd apps/Samples/Histogram
$ make
$ ./Run
```

To run the test suite and all benchmarks on FPGA:

```sh
$ cd test
$ ./test.sh --fpga     # Assumes FPGA image built and FPGA connected via USB
```

Use the `--stats` option to generate performance stats.

## Enabling CHERI :cherries:

To enable CHERI, some additional preparation is required.  First, edit
[inc/Config.h](inc/Config.h) and apply the following settings:

  * `#define EnableCHERI 1`
  * `#define EnableTaggedMem 1`
  * `#define UseClang 1`

Second, install the CHERI-Clang compiler using
[cheribuild](https://github.com/CTSRD-CHERI/cheribuild).  Assuming all
of [cheribuild's
dependencies](https://github.com/CTSRD-CHERI/cheribuild#pre-build-setup)
are met, we can simply do:

```sh
$ git clone https://github.com/CTSRD-CHERI/cheribuild
$ cd cheribuild
$ ./cheribuild.py sdk-riscv64-purecap
```

Note that a clean build on or after 19 Aug 2021 is required.  By
default, this will install the compiler into `~/cheri/`.  We then need
to add the compiler to our `PATH`:

```sh
export PATH=~/cheri/output/sdk/bin:$PATH
```

We musn't forget to `make clean` in the root of the SIMTight repo any
time [inc/Config.h](inc/Config.h) is changed.  At this point, all of
the standard build instructions should work as before.

## Enabling scalarisation

Scalarisation is an optimastion that spots _uniform_ and _affine_ vectors and
processes them more efficiently as scalars, reducing on-chip storage and
increasing performance density.  An _affine_ vector is one in which there is a
constant stride between each element; a _uniform_ vector is an affine vector
where the stride is zero, i.e. all elements are equal.

SIMTight implements _dynamic scalarisation_ (i.e. in hardware, at
runtime), and it can be enabled separately for the integer register
file and the register file holding capability meta-data.  To enable
scalarisation of both register files, edit
[inc/Config.h](inc/Config.h) and apply the following settings:

  * `#define SIMTEnableRegFileScalarisation 1`
  * `#define SIMTEnableCapRegFileScalarisation 1`

These options alone only enable scalarisation of uniform vectors.  To
enable scalariastion of affine vectors, apply the following settings

  * `#define SIMTEnableAffineScalarisation 1`
  * `#define SIMTAffineScalarisationBits 4`

The second of these parameters defines the number of bits used to
represent the constant stride between vector elements.  Note that
affine scalarisation is never used in the register file holding
capability meta-data, where it wouldn't make much sense.

SIMTight exploits scalarisation to reduce regiser file storage
requirements, typically saving hundreds of kilobytes of register
memory per CHERI-enabled SIMT core.  It also supports an experimental
_scalarised vector store buffer_ to reduce the cost of register spills
(runtime and DRAM overheads), at low hardware cost, which can be
enabled as follows.

  * `#define SIMTEnableSVStoreBuffer 1`

SIMTight also exploits scalarisation to process scalarisable
instructions using a dedicated scalar pipeline, which can be enabled
with:

  * `#define SIMTEnableScalarUnit 1`

The scalar pipeline allows an entire warp to be executed on a single execution
unit in a single cycle (when an instruction is detected as scalarisable), _and
operates in parallel with the main vector pipeline_.  For many workloads, this
increases perforance density significantly.

Possible future work: _partial_ scalarisation and _inter-warp_ scalarisation.
The register file implementation is cleanly separated from the pipeline so
these improvements can be explored in a self-contained manner.
