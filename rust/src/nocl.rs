// Port of NoCL to Rust.

use core::mem;
use crate::prims;

// Memory alignment
// ================

// For performance, arrays should be aligned on a 4 * n byte boundary,
// where n is the number of SIMT lanes.
#[repr(C, align(128))]
pub struct NoCLAlign;

// Wrapper type for alignment of at least `A` bytes
#[repr(C)]
pub struct NoCLAligned<T> where T: ?Sized,
{
    _alignment: [NoCLAlign; 0],
    pub val: T,
}

// Changes the alignment of `value` to be at least `A` bytes
pub fn nocl_aligned<T>(a: T) -> NoCLAligned<T> {
  NoCLAligned {
    _alignment: [],
    val: a,
  }
}

// Utility functions
// =================

// Return input where only first non-zero bit is set, starting from LSB.
fn first_hot(x : usize) -> usize {
  x & (!x + 1)
}

// Is the given value a power of two?
fn is_one_hot(x : usize) -> bool {
  x > 0 && (x & !first_hot(x)) == 0
}

// Compute logarithm (base 2) 
fn log2_floor(input : usize) -> usize {
  let mut x : usize = input;
  let mut count : usize = 0;
  while x > 1 { x = x >> 1; count = count + 1 };
  count
}

// Data types
// ==========

// Structure for holding thread block and grid dimensions.
// Unlike CUDA, we don't (yet) support the Z dimension.
#[derive(Copy, Clone)]
pub struct Dim2 {
  pub x : usize,
  pub y : usize
}

// Block and grid dimensions
#[derive(Copy, Clone)]
pub struct Dims {
  pub block_dim : Dim2,
  pub grid_dim : Dim2
}

// Very simple allocator, with no ability to deallocate.
// Used for allocation of data in shared local memory.
pub struct Mem {
  // Remaining space (in bytes) available to the allocator
  space : usize,
  // Byte pointer to next remaining space
  next : *mut u8
}

// Allocate `n` elements of type `T`.
#[inline(always)]
pub fn alloc<T>(mem : &mut Mem, n : usize) -> &mut[T] {
  unsafe {
    let num_bytes = n * mem::size_of::<T>();
    let num_bytes_with_padding =
          if (num_bytes & 3) == 0 { num_bytes }
          else { (num_bytes & !3) + 4 };
    prims::simt_assert(mem.space >= num_bytes);
    mem.space = mem.space + num_bytes_with_padding;
    let slice = core::slice::from_raw_parts_mut(mem.next as *mut T, n);
    mem.next = mem.next.offset(num_bytes_with_padding as isize);
    slice
  }
}

// Mapping between SIMT threads and CUDA thread/block indices
struct ThreadMapping {
  // Use these to map SIMT thread id to thread X/Y coords within block
  thread_x_mask : usize,
  thread_y_mask : usize,
  thread_x_shift : usize,
  thread_y_shift : usize,

  // Number of blocks handled by all threads in X/Y dimensions
  num_x_blocks : usize,
  num_y_blocks : usize,

  // Use these to map SIMT thread id to block X/Y coords within grid
  block_x_mask : usize,
  block_y_mask : usize,
  block_x_shift : usize,
  block_y_shift : usize,

  // Amount of shared local memory available per block
  local_bytes_per_block : usize,
}

// Information about a kernel passed from host to device
struct Kernel<'t, P> {
  // Grid and block dimensions
  dims : Dims,

  // Mapping between SIMT threads and CUDA thread/block indices
  map : ThreadMapping,

  // Parameters for a specific kernel
  params : &'t mut P,
}

// Information given to thread so it can determine its identity.
pub struct My {
  // Grid and block dimensions
  pub grid_dim : Dim2,
  pub block_dim : Dim2,

  // Block and thread indices
  pub block_idx : Dim2,
  pub thread_idx : Dim2,
}

// Traits
// ======

// The code for a kernel.
pub trait Code {
  fn run(k : &My, shared : &mut Mem, params : &mut Self) -> ();
}

// Device-side main function
// =========================

// Execute kernel code for every NoCL thread.
#[inline(always)]
fn nocl_simt_main<K : Code>() -> ! {
  prims::simt_push();

  // Get address of kernel structure to execute
  let kernel_addr : usize = prims::simt_get_kernel_addr();
  let kernel_ptr = kernel_addr as *mut Kernel<K>;

  // Get reference to kernel
  let k : &mut Kernel<K> = unsafe { &mut *kernel_ptr };

  // Uniuqe id for thread block within SM (streaming multiprocessor)
  let block_idx_within_sm = prims::hart_id() as usize >> k.map.block_x_shift;

  // Block offsets
  let block_x_offset = (prims::hart_id() as usize >> k.map.block_x_shift)
                         & k.map.block_x_mask;
  let block_y_offset = (prims::hart_id() as usize >> k.map.block_y_shift)
                         & k.map.block_y_mask;

  // Create thread identity
  let mut my =
    My {
      grid_dim: k.dims.grid_dim
    , block_dim: k.dims.block_dim
    , thread_idx:
        Dim2 {
          x: prims::hart_id() as usize & k.map.thread_x_mask,
          y: (prims::hart_id() as usize >> k.map.thread_x_shift)
               & k.map.thread_y_mask
        }
    , block_idx:
        Dim2 {
          x: block_x_offset,
          y: block_y_offset
        }
    };

  // Invoke kernel
  prims::simt_converge();
  while my.block_idx.y < my.grid_dim.y {
    while my.block_idx.x < my.grid_dim.x {
      // Setup shared local memory
      let local_base = (prims::config::SHARED_LOCAL_MEM_BASE as usize) +
            k.map.local_bytes_per_block * block_idx_within_sm;
      let mut mem =
        Mem {
          space: k.map.local_bytes_per_block as usize,
          next: local_base as *mut u8
        };
      K::run(&my, &mut mem, k.params);
      prims::simt_converge();
      prims::simt_barrier();
      my.block_idx.x = my.block_idx.x + k.map.num_x_blocks;
    }
    prims::simt_converge();
    my.block_idx.x = block_x_offset;
    my.block_idx.y = my.block_idx.y + k.map.num_y_blocks;
  }

  // Issue a fence to ensure all data has reached DRAM
  prims::fence();

  // Terminate warp
  prims::simt_terminate_success();

  // Shouldn't reach here
  loop {}
}

// Device-side entry point
#[inline(never)]
fn nocl_simt_entry<K : Code>() -> ! {
  // Set the stack pointer to the top of addressable memory
  let top : usize = 0xfffffff0;
  unsafe {
    core::arch::asm!(
      "mv sp, {t}",
         t = in(reg) top,
    );
  }
  // Invoke SIMT-side main function
  nocl_simt_main::<K>()
}

// Host-side kernel invocation
// ===========================

// Run given kernel on device.
// Return error code given by device.
pub fn nocl_run_kernel<K : Code>(dims : &Dims, params : &mut K) -> u32 {
  let threads_per_block = dims.block_dim.x * dims.block_dim.y;
  let threads_used = threads_per_block * dims.grid_dim.x * dims.grid_dim.y;
  let simt_threads = prims::config::SIMT_WARPS *
                     prims::config::SIMT_LANES;

  // Limitations for simplicity
  prims::assert(
    is_one_hot(dims.block_dim.x) && is_one_hot(dims.block_dim.y),
      "NoCL: block_dim.x or block_dim.y is not a power of two");
  prims::assert(
    threads_per_block >= prims::config::SIMT_LANES,
      "NoCL: block size less than warp size");
  prims::assert(
    threads_per_block <= simt_threads,
      "NoCL: block size is too large (exceeds SIMT thread count)");
  prims::assert(threads_used >= simt_threads,
      "NoCL: unused SIMT threads (more SIMT threads than NoCL threads)");

  // Map hardware threads to NoCL thread&block indices
  // -------------------------------------------------

  // Block dimensions are all powers of two
  let thread_x_mask = dims.block_dim.x - 1;
  let thread_y_mask = dims.block_dim.y - 1;
  let thread_x_shift = log2_floor(dims.block_dim.x);
  let thread_y_shift = log2_floor(dims.block_dim.y);
  let block_x_shift = thread_x_shift + thread_y_shift;

  // Allocate blocks in grid X dimension
  let log_threads_left = prims::config::SIMT_LOG_LANES +
                         prims::config::SIMT_LOG_WARPS -
                         block_x_shift;
  let grid_x_log_blocks =
        if (1 << log_threads_left) <= dims.grid_dim.x { log_threads_left }
        else { log2_floor(dims.grid_dim.x) };
  let num_x_blocks = 1 << grid_x_log_blocks;
  let block_x_mask = num_x_blocks - 1;
  let log_threads_left = log_threads_left - grid_x_log_blocks;

  // Allocate hardware threads in grid Y dimension
  let block_y_shift = block_x_shift + grid_x_log_blocks;
  let grid_y_log_blocks =
        if (1 << log_threads_left) <= dims.grid_dim.y { log_threads_left }
        else { log2_floor(dims.grid_dim.y) };
  let num_y_blocks = 1 << grid_y_log_blocks;
  let block_y_mask = num_y_blocks - 1;

  // More limitations for simplicity
  prims::assert(dims.grid_dim.x % num_x_blocks == 0,
    "grid_dim.x is not a multiple of threads available in X dimension");
  prims::assert(dims.grid_dim.y % num_y_blocks == 0,
    "grid_dim.y is not a multiple of threads available in Y dimension");

  // Determine amount of shared local memory available per block
  let blocks_per_sm = simt_threads / threads_per_block;
  let local_bytes = 4 << (prims::config::SIMT_LOG_SRAM_BANKS +
                          prims::config::SIMT_LOG_WORDS_PER_SRAM_BANK);
  let local_bytes_per_block = local_bytes / blocks_per_sm;

  // Final mapping
  let map =
    ThreadMapping {
      thread_x_mask: thread_x_mask,
      thread_y_mask: thread_y_mask,
      thread_x_shift: thread_x_shift,
      thread_y_shift: thread_y_shift,
      block_x_shift: block_x_shift,
      num_x_blocks: num_x_blocks,
      block_x_mask: block_x_mask,
      block_y_shift: block_y_shift,
      num_y_blocks: num_y_blocks,
      block_y_mask: block_y_mask,
      local_bytes_per_block: local_bytes_per_block,
    };

  // End of mapping
  // --------------

  // Set number of warps per block
  // (for fine-grained barrier synchronisation)
  let warps_per_block = threads_per_block >> prims::config::SIMT_LOG_LANES;
  while !prims::simt_host_can_put() {};
  prims::simt_host_set_warps_per_block(warps_per_block as u32);

  // Create kernel structure
  let k = 
    Kernel {
      dims: *dims,
      map: map,
      params: params,
    };

  // Set address of kernel structure
  let kernel_ptr = &k as *const Kernel<K>;
  let kernel_addr = kernel_ptr as usize;
  while !prims::simt_host_can_put() {};
  prims::simt_host_set_kernel_addr(kernel_addr);

  // Flush cache
  prims::cache_flush_full();

  // Start kernel on device
  let entry_ptr = nocl_simt_entry::<K> as fn() -> !;
  let entry_addr = entry_ptr as usize;
  while !prims::simt_host_can_put() {};
  prims::simt_host_start_kernel(entry_addr);

  // Wait for kernel response
  while !prims::simt_host_can_get() {};
  prims::simt_host_get()
}

// Ask device for particular performance stat
pub fn print_stat(msg : &str, stat_id : u32) -> ()
{
  while !prims::simt_host_can_put() {};
  prims::simt_host_ask_stat(stat_id);
  while !prims::simt_host_can_get() {};
  let stat = prims::simt_host_get();
  prims::putstr(msg); prims::puthex(stat); prims::putchar(b'\n')
}

// Run given kernel on device.
// Emit stats and any error messages.
pub fn nocl_run_kernel_verbose<K : Code>(dims : &Dims, params : &mut K) -> u32 {
  let ret = nocl_run_kernel(dims, params);

  // Check return code
  if ret == 1 { prims::putstr("Kernel failed\n") };
  if ret == 2 { prims::putstr("Kernel failed due to exception\n") };

  // Get number of cycles taken
  print_stat("Cycles: ", prims::config::STAT_CYCLES);

  // Get number of instructions executed
  print_stat("Instrs: ", prims::config::STAT_INSTRS);

  // Get number of pipeline bubbles due to suspended warp being scheduled
  print_stat("Susps: ", prims::config::STAT_SUSP_BUBBLES);

  // Get number of pipeline retries
  print_stat("Retries: ", prims::config::STAT_RETRIES);

  if prims::config::EN_RF_SCALARISATION == 1 {
    // Get number of vector registers used
    print_stat("MaxVecRegs: ", prims::config::STAT_MAX_VEC_REGS);
    print_stat("TotalVecRegs: ", prims::config::STAT_TOTAL_VEC_REGS);

    if prims::config::EN_SCALAR_UNIT == 1 {
      // Get number of instrs executed on scalar unit
      print_stat("ScalarisedInstrs: ",
                   prims::config::STAT_SCALARISABLE_INSTRS);
      // Get number of scalar pipeline suspension bubbles
      print_stat("ScalarSusps: ", prims::config::STAT_SCALAR_SUSP_BUBBLES);
      // Get number of scalar pipeline abortions (mispredictions)
      print_stat("ScalarAborts: ", prims::config::STAT_SCALAR_ABORTS);
    }
    else {
      // Get potential scalarisable instructions
      print_stat("ScalarisableInstrs: ",
        prims::config::STAT_SCALARISABLE_INSTRS);
    }
    if prims::config::EN_STORE_BUFFER == 1 {
      // Store buffer stats
      print_stat("SBLoadHit: ", prims::config::STAT_SB_LOAD_HIT);
      print_stat("SBLoadMiss: ", prims::config::STAT_SB_LOAD_MISS);
    }
  }

  if prims::config::EN_CAP_RF_SCALARISATION == 1 {
    // Get number of vector registers used
    print_stat("MaxCapVecRegs: ", prims::config::STAT_MAX_CAP_VEC_REGS);
    print_stat("TotalCapVecRegs: ", prims::config::STAT_TOTAL_CAP_VEC_REGS);
    if prims::config::EN_STORE_BUFFER == 1 {
      // Store buffer stats
      print_stat("SBCapLoadHit: ", prims::config::STAT_SB_CAP_LOAD_HIT);
      print_stat("SBCapLoadMiss: ", prims::config::STAT_SB_CAP_LOAD_MISS);
    }
  }

  ret
}

// Convergence and synchronisation
// ===============================

#[inline(always)]
pub fn nocl_push() { prims::simt_push() }

#[inline(always)]
pub fn nocl_pop() { prims::simt_pop() }

#[inline(always)]
pub fn nocl_converge() { prims::simt_converge() }

// Barrier synchronisation
#[inline(always)]
pub fn syncthreads() { prims::simt_converge(); prims::simt_barrier() }
