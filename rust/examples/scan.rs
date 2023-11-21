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

struct Scan<'t> {
  len    : usize,
  input  : &'t [i32],
  output : &'t mut [i32]
}

impl Code for Scan<'_> {

#[inline(always)]
fn run<'t> (my : &My, shared : &mut Mem, params: &mut Scan<'t>) {
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
  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 4096;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1024000;

  // Input and output vectors
  let mut input : NoCLAligned<[i32; N]> = nocl_aligned([0; N]);
  let mut output : NoCLAligned<[i32; N]> = nocl_aligned([0; N]);

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..N {
    input.val[i] = rand15(&mut seed) as i32
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
           input  : &input.val[..],
           output : &mut output.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  let mut acc : i32 = 0;
  for i in 0..N {
    acc += input.val[i];
    ok = ok && output.val[i] == acc
  }


  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
