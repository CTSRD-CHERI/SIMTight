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

struct Histogram {
  len   : usize,
  input : Buffer<u8>,
  bins  : Buffer<i32>
}

impl Code for Histogram {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut Histogram) {
  // Store histogram bins in shared local memory
  let mut histo = alloc::<i32>(shared, 256);

  // Initialise bins
  for i in (my.thread_idx.x .. 256).step_by(my.block_dim.x) {
    histo[i] = 0
  }

  syncthreads();

  for i in (my.thread_idx.x .. params.len).step_by(my.block_dim.x) {
    atomic_add(&mut histo[params.input[i] as usize], 1);
  }

  syncthreads();

  // Write bins to global memory
  for i in (my.thread_idx.x .. 256).step_by(my.block_dim.x) {
    params.bins[i] = histo[i]
  }
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
  let mut input : Vec<u8> = vec![0; N];
  let mut bins : Vec<i32> = vec![0; 256];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..N {
    input[i] = (rand15(&mut seed) & 0xff) as u8;
  }

  // Use a single block of threads
  let dims =
    Dims {
      block_dim: Dim2 { x: prims::config::SIMT_WARPS *
                           prims::config::SIMT_LANES,
                        y: 1 },
      grid_dim: Dim2 { x: 1, y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    Histogram { len   : N,
                input : input.into(),
                bins  : bins.into() };

  // Invoke kernel
  params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let mut ok = true;
  let mut golden_bins : [i32 ; 256] = [0; 256];
  for i in 0 .. 256 { golden_bins[i] = 0 }
  for i in 0 .. N {
    let idx = params.input[i] as usize;
    golden_bins[idx] += 1
  }
  for i in 0 .. 256 {
    ok = ok && params.bins[i] == golden_bins[i]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
