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

// Benchmark
// =========

const BLOCK_SIZE: usize = (prims::config::SIMT_WARPS *
                           prims::config::SIMT_LANES) as usize;

struct Reduce<'t> {
  len    : usize,
  input  : &'t [i32],
  sum    : &'t mut i32
}

impl Code for Reduce<'_> {

#[inline(always)]
fn run<'t> (my : &My, shared : &mut Mem, params: &mut Reduce<'t>) {
  let mut block = alloc::<i32>(shared, BLOCK_SIZE);

  // Sum global memory
  block[my.thread_idx.x] = 0;
  for i in (my.thread_idx.x .. params.len).step_by(my.block_dim.x) {
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
  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 3000;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1000000;

  // Input and output vectors
  let mut input : NoCLAligned<[i32; N]> = nocl_aligned([0; N]);
  let mut sum : i32 = 0;

  // Initialise inputs
  let mut seed : u32 = 1;
  let mut acc : i32 = 0;
  for i in 0..N {
    let r = rand15(&mut seed) as i32;
    input.val[i] = r;
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
    Reduce { len   : N,
             input : &input.val[..],
             sum   : &mut sum };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let ok = sum == acc;

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
