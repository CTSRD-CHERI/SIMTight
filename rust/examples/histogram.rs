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

struct Histogram<'t> {
  len    : usize,
  input  : &'t [u8],
  bins   : &'t mut [i32]
}

impl Code for Histogram<'_> {

#[inline(always)]
fn run<'t> (my : &My, shared : &mut Mem, params: &mut Histogram<'t>) {
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
  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 3000;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1000000;

  // Input and output vectors
  let mut input : NoCLAligned<[u8; N]> = nocl_aligned([0; N]);
  let mut bins : NoCLAligned<[i32; 256]> = nocl_aligned([0; 256]);

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..N {
    input.val[i] = (rand15(&mut seed) & 0xff) as u8;
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
                input : &input.val[..],
                bins  : &mut bins.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  let mut golden_bins : [i32 ; 256] = [0; 256];
  for i in 0 .. 256 { golden_bins[i] = 0 }
  for i in 0 .. N {
    let idx = input.val[i] as usize;
    golden_bins[idx] += 1
  }
  for i in 0 .. 256 {
    ok = ok && bins.val[i] == golden_bins[i]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
