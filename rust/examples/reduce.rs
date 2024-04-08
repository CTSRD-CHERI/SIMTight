#![no_std]
#![no_main]
#![allow(unused)]

// Crates being used
// =================

use riscv_rt::entry;

extern crate nocl;
use nocl::*;
use nocl::rand::*;
use nocl::prims::*;

extern crate alloc;
use alloc::vec;
use alloc::vec::*;
use alloc::boxed::*;

// Benchmark
// =========

const BLOCK_SIZE: usize = (prims::config::SIMT_WARPS *
                           prims::config::SIMT_LANES) as usize;

struct Reduce {
  input  : Buffer<i32>,
  sum    : Box<i32>
}

impl Code for Reduce {

#[inline(always)]
fn run (my : &My, shared : &mut Mem, params: &mut Reduce) {
  let mut block = alloc::<i32>(shared, BLOCK_SIZE);

  // Sum global memory
  block[my.thread_idx.x] = 0;
  for i in (my.thread_idx.x .. params.input.len()).step_by(my.block_dim.x) {
    block[my.thread_idx.x] += params.input[i]
  }

  syncthreads();

  // Sum shared local memory
  let mut i = my.block_dim.x >> 1;
  while (i > 0) {
    if my.thread_idx.x < i {
      block[my.thread_idx.x] += block[my.thread_idx.x + i];
    }
    syncthreads();
    i >>= 1
  }

  // Write sum to global memory
  if (my.thread_idx.x == 0) { *params.sum = block[0] }
}

}

#[entry]
fn main() -> ! {
  nocl_init();

  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 3000;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1000000;

  // Input and output vectors
  let mut input : Vec<i32> = vec![0; N];
  let mut sum = 0;

  // Initialise inputs
  let mut seed : u32 = 1;
  let mut acc : i32 = 0;
  for i in 0..N {
    let r = rand15(&mut seed) as i32;
    input[i] = r;
    acc += r
  }

  // Use a single block of threads
  let dims =
    Dims {
      block_dim: Dim2 { x: BLOCK_SIZE,
                        y: 1 },
      grid_dim: Dim2 { x: 1, y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    Reduce { input : input.into(),
             sum   : sum.into() };

  // Invoke kernel
  params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let ok = *params.sum == acc;

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
