# NoCL

NoCL is an ultra lightweight library for writing CUDA-like compute
kernels in plain C++ (no special compute language required, hence the
name).  It has been developed to allow CUDA kernels to be ported to
our CHERI-enabled [SIMTight GPGPU
SoC](https://github.com/CTSRD-CHERI/SIMTight) without needing any new
compiler work.  The NoCL API tries to abstract over the target
architecture, so alternative implementations should be possible.  Our
current implementation is defined in a [single header
file](/inc/NoCL.h).  This document starts at a fairly high level of
abstraction and gradually introduces details that are increasingly
specific to our current implementation for SIMTight.

## Device side example

Here's a basic compute kernel written in NoCL:

```cpp
#include <NoCL.h>

// Kernel for pointwise addition of two input arrays
struct VecAdd : Kernel {
  int len;
  int *a, *b, *result;

  void kernel() {
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      result[i] = a[i] + b[i];
  }
};
```

Like in CUDA, a kernel is a piece of code that runs for every _block_
in a _grid_, and for every _thread_ in a _block_.  Kernels have
implicit access to the dimensions of the grid and block (`gridDim` and
`blockDim`), and to the unique ids of the block and thread (`blockIdx`
and `threadIdx`).

Unlike in CUDA, a kernel is defined as a _class_ rather than a
function.  This is a natural way to capture the implicit CUDA
variables in C++; by subclassing the generic `Kernel` class,
methods gain access to `gridDim`, `blockDim`, `blockIdx`, `threadIdx`,
and other bits of implicit state discussed later.  Using a class also
makes it easy to copy kernel invocations, e.g. from host to device;
this is not really true of function applications.  Another attraction
is that programmers are encouraged to set each argument by name,
aiding readability for kernel invocations with many parameters.

The example kernel above assumes that the entire computation will be
performed by a single one-dimensional thread block.  Each thread
starts at an array index equal to the id of the thread, and walks
through both input arrays with a stride equal to the size of the
thread block.

## Host side example

On the host side, the kernel can be invoked as follows.

```cpp
int main()
{
  // Array size for testing
  const int N = 65536;

  // Input and output vectors
  nocl_aligned int a[N], b[N], result[N];

  // Initialise inputs
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2*i;
  }

  // Instantiate kernel
  VecAdd k;

  // Use a single block of threads
  // (All threads in SIMTight)
  k.blockDim.x = SIMTWarps * SIMTLanes;

  // Assign parameters
  k.len = N;
  k.a = a;
  k.b = b;
  k.result = result;

  // Invoke kernel
  noclRunKernel(&k);

  // Check result
  bool ok = true;
  for (int i = 0; i < N; i++) ok = ok && result[i] == 3*i;

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
```

We use the `nocl_aligned` macro to specify an alignment requirement on
all arrays passed to the kernel; this is not necessary but can be
important for efficiency.  For simplicity, we declare all arrays on
the host's stack (residing in DRAM); the SIMTight software framework
is baremetal and there is no heap allocator available yet.  Global
variables could also be used, and these would be accessible by the
host and device without the need for parameter passing.

The block size is set to the number of warps available multiplied by
the warp size, i.e. all hardware threads in the SIMTight core.  The
function `noclRunKernel` starts the kernel and waits for it to
complete, i.e. performs _synchronous_ kernel invocation.
_Asynchronous_ invocation is also possible but not yet provided by the
API.

## Single executable

An unusual and somewhat restrictive assumption made by NoCL is that
the host and device processors use the same instruction set (SIMTight
uses RISC-V for both) and that an application containing host and
device code is compiled to a single executable.  This doesn't mean
that all device code can run on the host or viceversa (e.g. there may
be CSRs and custom instructions available on one but not the other),
but host and device do share data and code sections, and can refer to
common symbols.

## Private memory

Every hardware thread has its own register file and stack.  There is
no need to explicitly mark variables as `private`; anything that would
normally be allocated on the stack _is_ private.  In SIMTight, the
stacks of all device threads reside in a region of DRAM; they are
interleaved at word granularity for efficiency of coalescing (see
later), but this is invisible to software provided the host does not
have access to the region.  When CHERI is enabled, capabilities can be
used to ensure that the host cannot access the device stacks region,
and also that device threads cannot access each others stacks or the
host stack.

## Shared local memory, atomics, and barrier synchronisation

A key feature of CUDA is efficient local memory that is shared by all
threads in a block.  In NoCL, shared local memory can be allocated
using the impllict `shared` variable (inherited from the `Kernel`
class) of type `SharedLocalMem`.  Here's an example kernel that
declares a 256-element shared local array per thread block:

```cpp
// Kernel for computing 256-bin histograms
struct Histogram : Kernel {
  int len;
  unsigned char* input;
  int* bins;

  void kernel() {
    // Store histogram bins in shared local memory
    int* histo = shared.array<int, 256>();

    // Initialise bins
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      histo[i] = 0;

    __syncthreads();

    // Update bins
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      atomicAdd(&histo[input[i]], 1);

    __syncthreads();

    // Write bins to global memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      bins[i] = histo[i];
  }
};
```

As the memory is shared, we must take care to avoid race conditions
by using atomics (e.g. `atomicAdd`) and/or a synchronisation barriers
(`__syncthreads()` synchronises all threads in a block).

Shared local memory is optimised for parallel random access using an
array of SRAM banks and a fast switching network. However, programmers
should be aware of _bank conflicts_ and try to minimise them.  A bank
conflict occurs when multiple threads in a warp access the same SRAM
bank at the same time.  The SRAM banks are interleaved, so the precise
condition for a bank conflict between two accesses is:

```hs
isBankConflict :: (Int, Int) -> Bool
isBankConflict (addrA, addrB) =
  (addrA `mod` SIMTSRAMBanks) == (addrB `mod` SIMTSRAMBanks)
```

When CHERI is enabled, we can enforce the desirable property that
threads in different blocks cannot access each others shared local
data, even though they may all reside in the same physical memory.

## Global memory

Any memory that is not _private_ or _shared local_ is _global_, i.e.
accessible by the host and the device, and is stored in shared DRAM.

## Constant memory

In CUDA, `constant` memory is small region of read-only memory that is
cached on-chip for efficient simultaneous access by multiple threads.
NoCL does not yet support CUDA-style constant memory.

## Memory coalescing

Access to DRAM in SIMTight is optimised by _coalescing_.  For
efficient NoCL kernels targetting SIMTight, the programmer should be
aware of SIMTight's two coalescing strategies:

  * _SameAddress_: If all active threads in a warp access the same
    address, these accesses will be coalesced to a single DRAM
    transaction.

  * _SameBlock_: If all active threads in a warp access the same
    block of memory in a lane-aligned manner, this will be coalesced
    into a single DRAM transaction.

These two strategies are repeatedly applied until all the requests
from the warp have been resolved.  More details can be found in
SIMTight's [coalescing
unit](https://github.com/blarney-lang/pebbles/blob/master/src/Pebbles/Memory/CoalescingUnit.hs).  Note that the
_SameAddress_ strategy is also applied to shared local memory
accesses, avoiding bank conflicts in a fairly common case.

## Thread divergence/convergence

Thread divergence occurs when threads executing in lock-step (i.e. in
the same warp) take different execution paths through the kernel.
This will usually result in reduced IPC because a SIMT core may only
be capable of fetching one instruction at a time, and diverged threads
may well be executing different instructions.  Therefore it is
desirable for diverged threads to reconverge as soon as possible.  In
CUDA, the compiler will insert extra instructions to handle
divergence/convergence behind the scenes.  In NoCL, we currently
assume an unmodified compiler.  Instead, the user provides
divergence/convergence information explicitly.

SIMTight's divergence/convergence model is captured very concisely as
follows: the SIMT core will always activate the threads in a warp with
the largest _nesting level_ and the _same PC_.  The nesting level of a
thread is controlled by two primitive functions: `noclPush()` and
`noclPop()`.  The former increments the nesting level, and the latter
decrements it.  Initially, when a kernel starts executing, the
nestling level will be one.

Suppose we have a kernel with a nested conditional statement

```cpp
  if (a < b) {
    if (c < d) {
      // Do something
    }
    // Do something else
  }
```

To specify convergence in NoCL, we can write

```cpp
  noclPush();
  if (a < b) {
    noclPush();
    if (c < d) {
      // Do something
    }
    noclPop();
    // Do something else
  }
  noclPop();
```

Another way to understand these primitives is as follows: `noclPop()`
will cause the SIMT core to wait for all threads in the warp that
were active at the previous respective call to `noclPush()`.

Sometimes this level of detail is unecessary and we can use the
following helper function.

```cpp
inline void noclConverge { noclPop(); noclPush(); }
```

Think of `noclConverge()` as follows: wait for all threads that were
active at the last call to `noclConverge()` (or at the start of kernel
execution, whichever is closest in program order).  Often we write
seqeuences of conditionals as:

```cpp
  if (a < b) {
    // Do something
  }
  noclConverge();

  if (e > g) {
    // Do something else
  }
  noclConverge();
```

Note that `__syncthreads()` includes an implicit call to
`noclConverge()`.

In future, we may look at automatic insertion of these instructions by
the compiler.  But for now they're something the NoCL programmer
should be aware of when targetting SIMT-style hardware.

## Example mapping of CUDA threads to hardware threads

One of the jobs of the NoCL implementation is to map an abritrary
number of CUDA/NoCL software threads to a finite number of hardware
threads.  Suppose there are 2048 hardware threads and the programmer
requests 8192 software threads; specifically, a block size of 64x2 and
a grid size of 8x8.  The NoCL mapper will proceed as follows

  * First, the mapper requires that the X dimension of the block must
    be a multiple of the warp size, which is 32 by default.  This is
    satisifed by the requested X dimension of 64.
    Two warps (64 hardware threads) will therefore be allocated to
    the block X dimension.

  * The mapper will then consider the requested block Y dimension of 2.
    There are sufficient hardware threads to allocate 2 warps to
    the block Y dimension.  The total number of warps used is now 4;
    that's 128 hardware threads in total.

  * Now NoCL will consider the requested grid X dimension of 8.  There
    asre sufficient hardware threads to map 8 blocks of 128
    threads to the grid X dimension.  The total number of hardware
    threads now in use is 1024.

  * Finally NoCL will consider the requested grid Y dimenion of 8.
    There are only enough resources available to allocate 2 rows of
    1024 threads to hardware threads.  NoCL will therefore use a
    loop of 4 iterations to handle all 8 rows of the grid.

In summary, the NoCL mapper is greedy.  It first allocates hardware
threads to the block X dimension, then the block Y dimension, then
grid X dimension, then the grid Y dimension.  When it runs out of
hardware threads, it will resort to looping.

The current mapper has various restrictions for simplicity, but it
will complain if these restrictions are violated.  For example, it
currently requires the block size to be less than the number of
hardware threads available, and does not yet support the Z dimension.

## SIMTight command/response protocol

SIMTight's scalar core and SIMT core communicate via a management
stream in each direction.  In the Scalar -> SIMT direction, the
following commands are available:

  * `SetWarpsPerBlockCmd`: The SIMT core has very little understanding of
    CUDA/NoCL thread blocks, but the hardware barrier
    synchronisation mechanism needs to know which warps belong to
    which block so that it can correctly synchronise only threads
    belonging to the same block.  Synchronising between blocks would
    not meet the semantics of CUDA's `__syncthreads()`.
    So this command tells the SIMT core the number of warps per
    block; it must be a power of two, and the SIMT core will assume that
    warp ids are allocated to blocks contiguously.

  * `SetKernelAddrCmd` writes a pointer (or capability)
    to the kernel invocation into a special register on
    the SIMT core.

  * `StartKernelCmd` causes the SIMT core to start
    executing code at a given PC.

  * `GetStatCmd` requests the value of one of the
    SIMT core's performance counters.

In the SIMT -> Scalar direction, the following command responses are
possible:

  * After a `StartKernelCmd`, and after the
    kernel completes execution on the SIMT core, an exit code will be
    returned to the scalar core.

  * After a `GetStatCmd` command, the SIMT core will sent
    a response containing the value of the performance counter
    requested.

For a full set of commands, responses, and performance counters, see
the [SIMT management
interface](https://github.com/blarney-lang/pebbles/blob/master/src/Pebbles/Pipeline/SIMT/Management.hs).
