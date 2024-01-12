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

const SQUARE_SIZE: usize = prims::config::SIMT_LANES as usize;

struct Transpose {
  width  : usize,
  height : usize,
  input  : Box<[i32]>,
  output : Box<[i32]>
}

impl Code for Transpose {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut Transpose) {
  let mut square = alloc::<i32>(shared, (SQUARE_SIZE+1) * SQUARE_SIZE);

  // Origin of square within matrix
  let origin_x = my.block_idx.x * my.block_dim.x;
  let origin_y = my.block_idx.y * my.block_dim.x;

  // Load square
  for y in (my.thread_idx.y .. my.block_dim.x).step_by(my.block_dim.y) {
    square[y*(SQUARE_SIZE+1) + my.thread_idx.x] =
      params.input[(origin_y + y)*params.width + origin_x + my.thread_idx.x]
  }

  syncthreads();
    
  // Store square
  for y in (my.thread_idx.y .. my.block_dim.x).step_by(my.block_dim.y) {
    params.output[(origin_x + y)*params.height + origin_y + my.thread_idx.x] =
      square[my.thread_idx.x*(SQUARE_SIZE+1) + y]
  }
}

}

#[entry]
fn main() -> ! {
  nocl_init();

  // Matrix size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const WIDTH : usize = 256;
  #[cfg(not(feature = "large_data_set"))]
  const HEIGHT : usize = 64;
  #[cfg(feature = "large_data_set")]
  const WIDTH : usize = 512;
  #[cfg(feature = "large_data_set")]
  const HEIGHT : usize = 512;

  // Input and output matrix data
  let mut mat_in : Vec<i32> = vec![0; WIDTH*HEIGHT];
  let mut mat_out : Vec<i32> = vec![0; WIDTH*HEIGHT];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..HEIGHT {
    for j in 0..WIDTH {
      mat_in[i*WIDTH + j] = rand15(&mut seed) as i32
    }
  }

  // Number of loop iterations per block.  The number of iterations
  // times the block Y dimension must equal the block X dimension.
  const ITERS_PER_BLOCK : usize = 4;

  // Block/grid dimensions
  let dims =
    Dims {
      block_dim: Dim2 { x: SQUARE_SIZE,
                        y: SQUARE_SIZE / ITERS_PER_BLOCK },
      grid_dim: Dim2 { x: WIDTH / SQUARE_SIZE,
                       y: HEIGHT / SQUARE_SIZE }
    };
 
  // Kernel parameters
  let mut params =
    Transpose { width  : WIDTH,
                height : HEIGHT,
                input  : mat_in.into(),
                output : mat_out.into() };

  // Invoke kernel
  let params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let mut ok = true;
  for i in 0 .. HEIGHT {
    for j in 0 .. WIDTH {
      ok = ok && params.output[j*HEIGHT + i] ==
                 params.input[i*WIDTH + j];
    }
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
