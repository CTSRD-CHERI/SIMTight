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

struct Scan {
  len    : usize,
  input  : Buffer<i32>,
  output : Buffer<i32>
}

impl Code for Scan {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut Scan) {
  let mut temp_in = alloc::<i32>(shared, BLOCK_SIZE);
  let mut temp_out = alloc::<i32>(shared, BLOCK_SIZE);

  // Shorthand for local thread id
  let t = my.thread_idx.x;

  for x in (0 .. params.len).step_by(my.block_dim.x) {
    // Load data
    temp_out[t] = params.input[x+t];
    syncthreads();

    // Local scan
    let mut offset = 1;
    while offset < my.block_dim.x {
      core::mem::swap(&mut temp_in, &mut temp_out);
      if t >= offset {
        temp_out[t] = temp_in[t] + temp_in[t - offset];
      }
      else {
        temp_out[t] = temp_in[t];
      }
      syncthreads();
      offset = offset << 1
    }

    // Store data
    let acc = if x > 0 { params.output[x-1] } else { 0 };
    params.output[x+t] = temp_out[t] + acc
  }
}

}

#[entry]
fn main() -> ! {
  nocl_init();

  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 4096;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1024000;

  // Input and output vectors
  let mut input : Vec<i32> = vec![0; N];
  let mut output : Vec<i32> = vec![0; N];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..N {
    input[i] = rand15(&mut seed) as i32
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
    Scan { len    : N,
           input  : input.into(),
           output : output.into() };

  // Invoke kernel
  let params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let mut ok = true;
  let mut acc : i32 = 0;
  for i in 0..N {
    acc += params.input[i];
    ok = ok && params.output[i] == acc
  }


  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
