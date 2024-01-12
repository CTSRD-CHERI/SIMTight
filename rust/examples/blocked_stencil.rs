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

struct SimpleStencil {
  x_size  : usize,
  y_size  : usize,
  in_buf  : Box<[i32]>,
  out_buf : Box<[i32]>
}

impl Code for SimpleStencil {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut SimpleStencil) {
  let block_size = my.block_dim.x * my.block_dim.y;
  let x = my.block_idx.x * my.block_dim.x + my.thread_idx.x;
  let y = my.block_idx.y * my.block_dim.y + my.thread_idx.y;
  let ind = y * params.x_size + x;

  // Load values into local memory
  let mut c = alloc::<i32>(shared, block_size);
  c[my.thread_idx.y * my.block_dim.x + my.thread_idx.x] = params.in_buf[ind];
  syncthreads();

  let mut result = c[my.thread_idx.y * my.block_dim.x + my.thread_idx.x];
  if x < params.x_size - 1 {
    if my.thread_idx.x == my.block_dim.x - 1 {
      result += params.in_buf[ind + 1]
    }
    else {
      result += c[my.thread_idx.y * my.block_dim.x + my.thread_idx.x + 1]
    }
  }
  nocl_converge();

  if x > 0 {
    if my.thread_idx.x == 0 {
      result += params.in_buf[ind - 1]
    }
    else {
      result += c[my.thread_idx.y * my.block_dim.x + my.thread_idx.x - 1]
    }
  }
  nocl_converge();

  if y < params.y_size - 1 {
    if my.thread_idx.y == my.block_dim.y - 1 {
      result += params.in_buf[(y + 1) * params.x_size + x];
    }
    else {
      result += c[(my.thread_idx.y + 1) * my.block_dim.x + my.thread_idx.x]
    }
  }
  nocl_converge();

  if y > 0 {
    if my.thread_idx.y == 0 {
      result += params.in_buf[(y - 1) * params.x_size + x]
    }
    else {
      result += c[(my.thread_idx.y - 1) * my.block_dim.x + my.thread_idx.x]
    }
  }
  nocl_converge();

  params.out_buf[ind] = result
}

}

#[entry]
fn main() -> ! {
  nocl_init();

  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const BUF_SIZE_X : usize = 64;
  #[cfg(not(feature = "large_data_set"))]
  const BUF_SIZE_Y : usize = 64;
  #[cfg(feature = "large_data_set")]
  const BUF_SIZE_X : usize = 1024;
  #[cfg(feature = "large_data_set")]
  const BUF_SIZE_Y : usize = 1024;

  const BUF_SIZE : usize = BUF_SIZE_X*BUF_SIZE_Y;

  // Input and output vectors
  let mut in_buf : Vec<i32> = vec![0; BUF_SIZE];
  let mut out_buf : Vec<i32> = vec![0; BUF_SIZE];
  let mut golden_out : Vec<i32> = vec![0; BUF_SIZE];

  // Initialise inputs
  let mut seed : u32 = 1;
  for y in 0..BUF_SIZE_Y {
    for x in 0 .. BUF_SIZE_X {
      in_buf[y * BUF_SIZE_X + x] = rand15(&mut seed) as i32
    }
  }

  // Use a single block of threads
  let dims =
    Dims {
      block_dim: Dim2 { x: prims::config::SIMT_LANES,
                        y: prims::config::SIMT_WARPS },
      grid_dim: Dim2 { x: BUF_SIZE_X / prims::config::SIMT_LANES,
                       y: BUF_SIZE_Y / prims::config::SIMT_WARPS }
    };
 
  // Kernel parameters
  let mut params =
    SimpleStencil {
      x_size  : BUF_SIZE_X,
      y_size  : BUF_SIZE_Y,
      in_buf  : in_buf.into(),
      out_buf : out_buf.into()
    };

  // Invoke kernel
  let params = nocl_run_kernel_verbose(dims, params);

  // Golden output
  for y in 0..BUF_SIZE_Y {
    for x in 0..BUF_SIZE_X {
      let ind = y * BUF_SIZE_X + x;
      let mut result = params.in_buf[ind];
      if x < BUF_SIZE_X - 1 { result += params.in_buf[y * BUF_SIZE_X + x + 1] }
      if x > 0              { result += params.in_buf[y * BUF_SIZE_X + x - 1] }
      if y < BUF_SIZE_Y - 1 { result += params.in_buf[(y+1) * BUF_SIZE_X + x] }
      if y > 0              { result += params.in_buf[(y-1) * BUF_SIZE_X + x] }
      golden_out[ind] = result;
    }
  }

  // Check result
  let mut ok = true;
  for i in 0..BUF_SIZE {
    ok = ok && params.out_buf[i] == golden_out[i]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
