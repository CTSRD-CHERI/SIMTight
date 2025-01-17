// Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.

// Modified for Rust NoCL, November 2023.

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

const LOCAL_SIZE_LIMIT: usize = 1024 as usize;

struct BitonicSortLocal {
  length       : usize,
  sort_dir     : bool,
  d_srckey_arg : Buffer<u32>,
  d_srcval_arg : Buffer<u32>,
  d_dstkey_arg : Buffer<u32>,
  d_dstval_arg : Buffer<u32>
}

impl Code for BitonicSortLocal {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut BitonicSortLocal) {
  let mut l_key = alloc::<u32>(shared, LOCAL_SIZE_LIMIT);
  let mut l_val = alloc::<u32>(shared, LOCAL_SIZE_LIMIT);

  // Offset to the beginning of subBATCH and load data
  let d_srckey = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  let d_srcval = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  let d_dstkey = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  let d_dstval = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  l_key[my.thread_idx.x + 0] = params.d_srckey_arg[d_srckey+0];
  l_val[my.thread_idx.x + 0] = params.d_srcval_arg[d_srcval+0];
  l_key[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)] =
    params.d_srckey_arg[d_srckey + (LOCAL_SIZE_LIMIT / 2)];
  l_val[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)] =
    params.d_srcval_arg[d_srcval + (LOCAL_SIZE_LIMIT / 2)];

  let mut size = 2;
  while size < params.length {
    // Bitonic merge
    let dir = (my.thread_idx.x & (size / 2)) != 0;
    let mut stride = size / 2;
    while stride > 0 {
      syncthreads();
      let pos = 2 * my.thread_idx.x - (my.thread_idx.x & (stride - 1));
      let pos_plus = pos+stride;
      if (l_key[pos] > l_key[pos_plus]) == dir {
        l_key.swap(pos, pos_plus);
        l_val.swap(pos, pos_plus);
      }
      nocl_converge();
      stride = stride >> 1
    }
    size = size << 1
  }

  // dir == sortDir for the last bitonic merge step
  {
    let mut stride = params.length / 2;
    while stride > 0 {
      syncthreads();
      let pos = 2 * my.thread_idx.x - (my.thread_idx.x & (stride - 1));
      let pos_plus = pos + stride;
      if (l_key[pos] > l_key[pos_plus]) == params.sort_dir {
        l_key.swap(pos, pos_plus);
        l_val.swap(pos, pos_plus);
      }
      nocl_converge();
      stride = stride >> 1
    }
  }

  syncthreads();
  params.d_dstkey_arg[d_dstkey + 0] = l_key[my.thread_idx.x + 0];
  params.d_dstval_arg[d_dstval + 0] = l_val[my.thread_idx.x + 0];
  params.d_dstkey_arg[d_dstkey + (LOCAL_SIZE_LIMIT / 2)] =
    l_key[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)];
  params.d_dstval_arg[d_dstval + (LOCAL_SIZE_LIMIT / 2)] =
    l_val[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)];
}

}

#[entry]
fn main() -> ! {
  nocl_init();

  // Vector size for benchmarking
  const N : usize = LOCAL_SIZE_LIMIT;
  #[cfg(not(feature = "large_data_set"))]
  const BATCH : usize = 4;
  #[cfg(feature = "large_data_set")]
  const BATCH : usize = 32;

  // Input and output vectors
  let mut srckeys : Vec<u32> = vec![0; N*BATCH];
  let mut srcvals : Vec<u32> = vec![0; N*BATCH];
  let mut dstkeys : Vec<u32> = vec![0; N*BATCH];
  let mut dstvals : Vec<u32> = vec![0; N*BATCH];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0 .. N*BATCH {
    srckeys[i] = rand15(&mut seed);
    srcvals[i] = rand15(&mut seed)
  }

  // Use a single block of threads
  let dims =
    Dims {
      block_dim: Dim2 { x: LOCAL_SIZE_LIMIT/2,
                        y: 1 },
      grid_dim: Dim2 { x: BATCH, y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    BitonicSortLocal {
      length       : N,
      sort_dir     : true,
      d_srckey_arg : srckeys.into(),
      d_srcval_arg : srcvals.into(),
      d_dstkey_arg : dstkeys.into(),
      d_dstval_arg : dstvals.into()
    };

  // Invoke kernel
  let params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let mut ok = true;
  for b in 0..BATCH {
    for i in 0 .. N-1 {
      ok = ok && params.d_dstkey_arg[b*N+i] <= params.d_dstkey_arg[b*N+i+1]
    }
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
