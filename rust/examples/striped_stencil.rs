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
  let window_size = 3 * my.block_dim.x;

  // Data blocks to the left, middle, and right of current output
  let mut buffer = alloc::<i32>(shared, my.block_dim.y * window_size);

  // Offsets for left, middle, and right blocks
  let left = 0;
  let middle = my.block_dim.x;
  let right = 2 * my.block_dim.x;

  // Base index of current row
  let row_base = my.block_idx.y * my.block_dim.y * params.x_size +
        my.thread_idx.y * params.x_size;

  // Initially, left block is zeros
  buffer[my.thread_idx.y * window_size + left + my.thread_idx.x] = 0;

  // Initially, middle block is first block in row
  buffer[my.thread_idx.y * window_size + middle + my.thread_idx.x] =
    params.in_buf[row_base + my.thread_idx.x];

  // Visit every block in row
  for i in (0 .. params.x_size).step_by(my.block_dim.x) {

    // Index of value being computed by this thread
    let idx = row_base + i + my.thread_idx.x;

    // Fetch right block (won't diverge: row is multiple of block width)
    if i + my.block_dim.x == params.x_size {
      buffer[my.thread_idx.y * window_size + right + my.thread_idx.x] = 0
    }
    else {
      buffer[my.thread_idx.y * window_size + right + my.thread_idx.x] =
        params.in_buf[idx + my.block_dim.x]
    }

    syncthreads();

    // Fetch blocks above and below (won't diverge: conditioned on Y index)
    let above = if my.thread_idx.y != 0
      { buffer[(my.thread_idx.y-1)*window_size + middle + my.thread_idx.x] }
      else { if my.block_idx.y == 0 { 0 }
               else { params.in_buf[idx - params.x_size] } };

    let below = if my.thread_idx.y != my.block_dim.y-1
      { buffer[(my.thread_idx.y+1)*window_size + middle + my.thread_idx.x] }
      else { if my.block_idx.y == my.grid_dim.y-1 { 0 }
               else { params.in_buf[idx + params.x_size] } };

    // Shorthands, used often below
    let mx = middle + my.thread_idx.x;
    let by = my.thread_idx.y * window_size;

    // Write output
    params.out_buf[idx] =
        above
      + below
      + buffer[by + mx - 1]
      + buffer[by + mx]
      + buffer[by + mx + 1];

    syncthreads();

    // Shift buffer
    buffer[by + left + my.thread_idx.x] = buffer[by + mx];
    buffer[by + mx] = buffer[by + right + my.thread_idx.x];
  }
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
                        y: 4 },
      grid_dim: Dim2 { x: 1,
                       y: BUF_SIZE_Y / 4 }
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
      if y < BUF_SIZE_Y - 1 { result += params.in_buf[(y + 1)*BUF_SIZE_X + x] }
      if y > 0              { result += params.in_buf[(y - 1)*BUF_SIZE_X + x] }
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
