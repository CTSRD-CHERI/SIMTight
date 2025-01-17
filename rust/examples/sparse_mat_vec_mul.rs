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

struct SparseMatVecMul {
  num_rows         : usize,
  num_cols         : usize,
  num_cols_per_row : usize,
  indices          : Buffer<usize>,
  data             : Buffer<i32>,
  x                : Buffer<i32>,
  y                : Buffer<i32>
}

impl Code for SparseMatVecMul {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut SparseMatVecMul) {
  let row = my.block_dim.x * my.block_idx.x + my.thread_idx.x;
  if row < params.num_rows {
    let mut dot = 0;
    for n in 0 .. params.num_cols_per_row {
      let col = params.indices[params.num_rows * n + row];
      let val = params.data[params.num_rows * n + row];
      dot += val * params.x[col];
    }
    params.y[row] = dot
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
  const WIDTH : usize = 2048;
  #[cfg(feature = "large_data_set")]
  const HEIGHT : usize = 2048;

  // Sparsity of matrix (power of two)
  const SPARSITY : usize = 8;
  const SAMPLES_PER_ROW : usize = WIDTH / SPARSITY;

  // Input and output matrix data
  let mut data : Vec<i32> = vec![0; SAMPLES_PER_ROW*HEIGHT];
  let mut indices : Vec<usize> = vec![0; SAMPLES_PER_ROW*HEIGHT];
  let mut data_t : Vec<i32> = vec![0; SAMPLES_PER_ROW*HEIGHT];
  let mut indices_t : Vec<usize> = vec![0; SAMPLES_PER_ROW*HEIGHT];
  let mut vec_in : Vec<i32> = vec![0; WIDTH*2];
  let mut vec_out : Vec<i32> = vec![0; HEIGHT];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..WIDTH {
    vec_in[i] = rand15(&mut seed) as i32
  }
  for r in 0..HEIGHT {
    vec_out[r] = 0;
    let mut offset = (rand15(&mut seed) as usize) & (2*SPARSITY - 1);
    let mut n = 0;
    while n < SAMPLES_PER_ROW {
      data[r*SAMPLES_PER_ROW + n] = (rand15(&mut seed) & 0xff) as i32;
      indices[r*SAMPLES_PER_ROW + n] = offset;
      n = n + 1;
      offset += (rand15(&mut seed) as usize) & (2*SPARSITY-1);
      if offset >= WIDTH { break }
    }
    while n < SAMPLES_PER_ROW {
      data[r*SAMPLES_PER_ROW + n] = 0;
      indices[r*SAMPLES_PER_ROW + n] = 0;
      n = n + 1
    }
  }

  // Get matrix in column-major order
  for r in 0 .. HEIGHT {
    for n in 0 .. SAMPLES_PER_ROW {
      data_t[n * HEIGHT + r] = data[r * SAMPLES_PER_ROW + n];
      indices_t[n * HEIGHT + r] = indices[r * SAMPLES_PER_ROW + n]
    }
  }

  // Block/grid dimensions
  const GROUPS : usize = HEIGHT / prims::config::SIMT_LANES;
  let dims =
    Dims {
      block_dim: Dim2 { x: prims::config::SIMT_LANES,
                        y: 1 },
      grid_dim: Dim2 { x: if GROUPS < prims::config::SIMT_WARPS
                            { prims::config::SIMT_WARPS } else { GROUPS },
                       y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    SparseMatVecMul { num_rows         : HEIGHT,
                      num_cols         : WIDTH,
                      num_cols_per_row : SAMPLES_PER_ROW,
                      indices          : indices_t.into(),
                      data             : data_t.into(),
                      x                : vec_in.into(),
                      y                : vec_out.into() };

  // Invoke kernel
  let params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let mut ok = true;
  for r in 0 .. HEIGHT {
    let mut sum = 0;
    for n in 0 .. SAMPLES_PER_ROW {
      let i = r*SAMPLES_PER_ROW + n;
      if data[i] != 0 { sum += data[i] * params.x[indices[i]] }
    }
    ok = ok && sum == params.y[r]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
