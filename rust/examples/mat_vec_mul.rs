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

const BLOCK_SIZE: usize = prims::config::SIMT_LANES as usize;

struct MatVecMul<'t> {
  width   : usize,
  height  : usize,
  mat     : &'t [i32],
  vec_in  : &'t [i32],
  vec_out : &'t mut [i32]
}

impl Code for MatVecMul<'_> {

#[inline(always)]
fn run<'t> (my : &My, shared : &mut Mem, params: &mut MatVecMul<'t>) {
  let mut partial = alloc::<i32>(shared, BLOCK_SIZE);

  for y in (my.block_idx.x .. params.height).step_by(my.grid_dim.x) {
    // Row processed by this block
    let row = y * params.width;

    // Compute partial dot products
    let mut sum = 0;
    for x in (my.thread_idx.x .. params.width).step_by(my.block_dim.x) {
      sum += params.mat[row+x] * params.vec_in[x];
    }
    partial[my.thread_idx.x] = sum;
    syncthreads();

    // Final local reduction
    let mut i = my.block_dim.x >> 1;
    while i > 0 {
      if my.thread_idx.x < i {
        partial[my.thread_idx.x] += partial[my.thread_idx.x + i];
      }
      syncthreads();
      i = i >> 1
    }

    // Write dot product to global memory
    if my.thread_idx.x == 0 { params.vec_out[y] = partial[0] }

    nocl_converge()
  }
}

}

#[entry]
fn main() -> ! {
  // Matrix size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const WIDTH : usize = 128;
  #[cfg(not(feature = "large_data_set"))]
  const HEIGHT : usize = 64;
  #[cfg(feature = "large_data_set")]
  const WIDTH : usize = 1024;
  #[cfg(feature = "large_data_set")]
  const HEIGHT : usize = 1024;

  // Input and output matrix data
  let mut mat : NoCLAligned<[i32; WIDTH*HEIGHT]> =
        nocl_aligned([0; WIDTH*HEIGHT]);
  let mut vec_in : NoCLAligned<[i32; WIDTH]> =
        nocl_aligned([0; WIDTH]);
  let mut vec_out : NoCLAligned<[i32; HEIGHT]> =
        nocl_aligned([0; HEIGHT]);

  // Initialise inputs
  let mut seed : u32 = 1;
  for j in 0..WIDTH {
    vec_in.val[j] = rand15(&mut seed) as i32
  }
  for i in 0..HEIGHT {
    for j in 0..WIDTH {
      mat.val[i*WIDTH + j] = rand15(&mut seed) as i32
    }
  }

  // Block/grid dimensions
  let dims =
    Dims {
      block_dim: Dim2 { x: prims::config::SIMT_LANES,
                        y: 1 },
      grid_dim: Dim2 { x: prims::config::SIMT_WARPS,
                       y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    MatVecMul { width   : WIDTH,
                height  : HEIGHT,
                mat     : &mat.val[..],
                vec_in  : &vec_in.val[..],
                vec_out : &mut vec_out.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  for i in 0 .. HEIGHT {
    let mut sum = 0;
    for j in 0 .. WIDTH {
      sum += mat.val[i*WIDTH+j] * vec_in.val[j]
    }
    ok = ok && sum == vec_out.val[i]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
