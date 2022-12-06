# Fast Zeroing

To avoid information leakage between kernel instances, it is desirable
to zero memory when a kernel completes so that, if the memory
allocator gives that memory to a future kernel instance, then it
cannot read senstive data from the prior kernel instance. In cases
where kernels run concurrently (not supported yet), it may also be
desirable to zero certain memory areas during a context switch.

SIMTight has experimental support for *fast zeroing* of memory. To
enable it, set the following flag in [Config.h](inc/Config.h):

  * `#define EnableFastZeroing 1`

It works as follows:

  1. The hardware maintains a *zero table* in DRAM with

       * 1 bit for every *DRAM beat* (by default 512 bits, defined by
        `DRAMBeatBits` in [Config.h](inc/Config.h)).
         A DRAM beat is the amount of data that can be written to
         DRAM in a single clock cycle.

       * This bit denotes whether the entire beat is 0 or not (if the bit is
         zero then that means the entire beat is zero, otherwise it is
         non-zero)

       * By default, with 4GB of DRAM, there are 2^26 of these bits
         (an 8MB table stored in DRAM).

  2. For efficiency, this table is cached on-chip. 

       * The cache line size is a multiple of the beat size, and an entire
         beat can be written to the cache in a single clock cycle.

       * Therefore it is possible to zero up to `DRAMBeatBits *
         DRAMBeatBits` bits of memory in a single clock cycle.
         By default, that's up to 32KB of memory in a single clock
         cycle.

       * The amount of memory being fast-zeroed must be a multiple of
         the beat size (64 bytes by default)

       * [Config.h](inc/Config.h) defines the cache line size
         (`ZeroCacheLogBeatsPerLine`), number of ways
         (`ZeroCacheLogNumWays`), number of sets (`ZeroCacheLogSets`),
         along with various other parameters.

  3. The zero cache is instantiated in the tag controller. The
     algorithm for maintaining the zero cache/table is as follows:

       * **Load operations**:

           - Issue load to DRAM and zero cache in parallel.

           - On responses, mask out the DRAM data if ther zero cache
             tells us it is zero.

       * **Full store operations** (i.e. a store where all bytes are
         being written):

            - Issue store to DRAM and mark beat as non-zero in the
              zero cache, in parallel

       * **Partial store operations** (i.e. a store where not all
         bytes are being written):

            - If, according to the zero cache, the beat was zero before
              the store, we need to write zeros to DRAM for all the bytes
              that are not being touched by the partial store, so we:

            - Forward the store to DRAM and lookup the zero cache, in
              parallel.

            - While we wait for the response from the zero cache,
              we want to continue servicing other requests, in a
              pipelined manner. However, given that this store operation
              is not yet complete, we must prevent requests that access
              the same beat from proceeding. To do this, we add the
              address of the store to a blacklist.

            - Hopefully subsequent requests will access addresses that
              are not in the blacklist, but if they do, they will
              stall the tag controller's pipeline.  The capacity of the
              blacklist is `ZeroCacheMaxOutstandingPartialStores`,
              defined in [Config.h](inc/Config.h).

            - When the response from the zero cache eventually comes,
              we do three things: (1) zero untouched bytes by writing
              to DRAM; (2) mark the beat as non-zero by writing to
              the zero cache; (3) remove the address from the blacklist.

One source of overhead from this feature is the handling of partial
stores. Partial stores arise when the coalescing unit is not able to
group lots of narrow (32-bit) stores from multiple threads into a
single wide (512-bit) store.  This can happen, for example, in the
presence of thread divergence, where some threads in a warp become
inactive until they reconverge.  Another source of overhead is cache
misses, introduced by aliasing or large working sets.

This fast zeroing feature is largely untested, and completely
unevaluated.  No measurements have been made to undestand the
efficiency (or otherwise) of the approach under various operating
conditions.

The interface to fast zeroing from software is defined
[here](/inc/FastZero.h).
