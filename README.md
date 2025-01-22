# SIMTight

SIMTight is a fully synthesisable GPGPU core implementing the _Single
Instruction Multiple Threads (SIMT)_ model, featuring:

  * RISC-V instruction set (`rv32ima_zfinx_xcheri`)
  * Low-area design with high IPC on classic GPGPU workloads
  * Dynamic scalarisation (automatic detection of scalar
    behaviour in hardware without ISA/compiler mods)
  * Parallel scalar/vector pipelines, exploiting scalarisation for
    increased throughput
  * Register file and cache compression, exploiting scalarisation for
    reduced on-chip storage and energy
  * Strong [CHERI](http://cheri-cpu.org) memory safety and isolation
  * Runs [CUDA-like C++ library](doc/NoCL.md) and [benchmark suite](apps/)
    (in pure capability mode when CHERI enabled)

Further details about SIMTight can be found in the following publications.

  * *Advanced Dynamic Scalarisation for RISC-V GPGPUs*, ICCD 2024
    ([paper](https://www.repository.cam.ac.uk/handle/1810/373257), [slides](doc/iccd2024-slides.pdf))

SIMTight is being developed on the [CAPcelerate
project](https://ctsrd-cheri.github.io/capcelerate-website/),
part of the UKRI's Digital Security by Design programme.

## Default SoC

The default SIMTight SoC consists of a host CPU and a 32-lane 64-warp
streaming multiprocessor sharing DRAM, both supporting the
CHERI-RISC-V ISA.  A sample project is included for the
[DE10-Pro](http://de10-pro.terasic.com) ([revD](de10-pro/) and
[revE](de10-pro-e/)) FPGA development board.

<div style="text-align: center;" align="center">
<img src="doc/SoC.svg" width="450">
</div>

## Build instructions

We'll need Verilator, a RISC-V compiler, and GHC 9.2.1 or later.

On Ubuntu 20.04 or 22.04, we can do:

```sh
$ sudo apt install verilator
$ sudo apt install gcc-riscv64-unknown-elf
$ sudo apt install libgmp-dev
```

For GHC 9.2.1 or later, [ghcup](https://www.haskell.org/ghcup/) can be
used.

If you're having difficulty meeting the dependencies, please
use our [docker container](docker/).

## Getting started

Recursively clone the repo:

```sh
$ git clone --recursive https://github.com/CTSRD-CHERI/SIMTight
```

Inside the repo, there are various things to try.  For example, to
build and run the SIMTight simulator:

```sh
$ cd sim
$ make
$ ./sim &
```

With the simulator running in the background, we can build and run the
test suite:

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
revE](http://de10-pro.terasic.com) board (Quartus 21.3pro or later
recommended):

```sh
$ cd de10-pro-e
$ make                 # Assumes quartus is in your PATH
$ make download-sof    # Assumes DE10-Pro revE is connected via USB
```

We can now run a SIMT kernel on FPGA:

```sh
$ cd apps/Samples/Histogram
$ make
$ ./Run
```

To run the test suite and all benchmarks on a DE10-Pro revE FPGA:

```sh
$ cd test
$ ./test.sh --fpga-e    # Assumes FPGA image built and FPGA connected via USB
```

Use the `--stats` option to generate performance stats.

## Enabling CHERI :cherries:

To enable CHERI, some additional preparation is required.  First, edit
[inc/Config.h](inc/Config.h) and apply the following settings:

  * `#define EnableCHERI 1`
  * `#define EnableTaggedMem 1`
  * `#define UseClang 1`

Second, install the CHERI-Clang compiler using our
[script](cheri-tools/build-cheri.sh).  Assuming
all of [cheribuild's
dependencies](https://github.com/CTSRD-CHERI/cheribuild#pre-build-setup)
are met, we can simply do:

```sh
$ cd cheri-tools
$ ./build-cheri.sh
```

This will install the compiler into
`$(pwd)/cheri/output/sdk/bin`, which we can then add to our
`PATH`:

```sh
export PATH=$(pwd)/cheri/output/sdk/bin:$PATH
```

If you're having difficulty meeting any of [cheribuild's
dependencies](https://github.com/CTSRD-CHERI/cheribuild#pre-build-setup),
please use our [docker container](docker/).

We musn't forget to `make clean` in the root of the SIMTight repo any
time [inc/Config.h](inc/Config.h) is changed.  At this point, all of
the standard build instructions should work as before.

CHERI instructions for getting and setting bounds on capabilities are quite
expensive in terms of logic area and typically not performance critical.
Therefore, it can be useful to share bounds getting/setting logic between
vector lanes:

  * `#define SIMTUseSharedBoundsUnit 1`

Various optimisations are enabled by this setting. It leads to a large
reduction in area overhead, at almost no performance cost accross the
benchmark suite.

Another option that reduces the area overhead of CHERI is:

  * `#define SIMTUseFixedPCC 1`

But beware, this setting removes some CHERI functionality. Specifically, it
tells the SIMT core to ignore changes to the bounds and permissions of the PCC.
Once the bounds and permissions of the PCC for each warp are set at kernel
startup, they can never be changed.

## Enabling scalarisation

Scalarisation is an optimastion that detects _uniform_ and _affine_
vectors and processes them more efficiently as scalars, reducing
on-chip storage and increasing performance density.  An _affine_
vector is one in which there is a constant stride between each
element; a _uniform_ vector is an affine vector where the stride is
zero, i.e. all elements are equal.

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

Note that affine scalarisation is never used in the register file
holding capability meta-data, where it wouldn't make much sense.

SIMTight exploits scalarisation to reduce register file storage
requirements. Hence, it is desirable to set the number of physical
registers to a value smaller than the number of architectural
registers.  In cases where scalarisation cannot prevent overflow of
the physical register file, the hardware implements _dynamic register
spilling_, where registers are evicted to and fetched from DRAM as
required.  In the default configuration, the size of the physical
register files is equal to the number of architectural registers (so
dynamic spilling is not required):

  * `#define SIMTLogRegFileSize 11`
  * `#define SIMTLogCapRegFileSize 11`

At the moment we have two spill policies: pick-first and
least-recently-used. To enable the latter:

  * `#define SIMTUseLRUSpill 1`

When CHERI is enabled, it's possible to share vector register memory
between the integer and capability meta-data register files.  

  * `#define SIMTUseSharedVecScratchpad 1`

In this case, both register file sizes must be defined the same.  This
option causes a one cycle pipeline bubble when loading a capability
meta-data vector from the register file.

SIMTight also supports an experimental _scalarised vector store buffer_ (also
referred to as the compressed stack cache) to reduce the cost of
compiler-inserted register spills (as opposed to hardware-inserted dynamic
spills), at low hardware cost, which can be enabled as follows.

  * `#define SIMTEnableSVStoreBuffer 1`

As well as reducing on-chip storage, scalarisation is also exploited to
improve runtime performance: enabling a scalar pipeline in the SIMT
core allows an entire warp to be executed on a single execution unit
in a single cycle (when the instruction is detected as scalarisable),
_and operates in parallel with the main vector pipeline_. For many
workloads, this increases perforance density significantly.

  * `#define SIMTEnableScalarUnit 1`

To enable the intial value optimisation (also referred to as the null value optimisation) in the capability meta-data register file:

  * `#define SIMTCapRFUseInitValOpt 1`

This a simple form of partial scalarisation allowing compact storage
of vectors that can be partioned into an arbitrary scalar value and
the initial value (null capability meta-data in this case) using a bit
mask.

<div style="text-align: center;" align="center">
<br>
<br>
<p>Supported by
<p><img src="doc/UKRI_Logo.svg" width="250"><br>
<p>Digital Security by Design (DSbD) Programme
</div>
