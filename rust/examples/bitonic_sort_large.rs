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

const LOCAL_SIZE_LIMIT: usize = 4096 as usize;

#[inline(always)]
fn two_sort(keys : &mut [u32], vals : &mut [u32],
            a_idx : usize, b_idx : usize, dir : bool) {
  if (keys[a_idx] > keys[b_idx]) == dir {
    keys.swap(a_idx, b_idx);
    vals.swap(a_idx, b_idx);
  }
  nocl_converge()
}

struct BitonicSortLocal {
  d_srckey_arg : Box<[u32]>,
  d_srcval_arg : Box<[u32]>,
  d_dstkey_arg : Box<[u32]>,
  d_dstval_arg : Box<[u32]>
}

// Bottom-level bitonic sort
// Even / odd subarrays (of LOCAL_SIZE_LIMIT points) are
// sorted in opposite directions
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

  let global_id = my.block_dim.x * my.block_idx.x + my.thread_idx.x;
  let comparator_i = global_id & ((LOCAL_SIZE_LIMIT / 2) - 1);

  let mut size = 2;
  while size < LOCAL_SIZE_LIMIT {
    // Bitonic merge
    let dir = (comparator_i & (size / 2)) != 0;
    let mut stride = size / 2;
    while stride > 0 {
      syncthreads();
      let pos = 2 * my.thread_idx.x - (my.thread_idx.x & (stride - 1));
      two_sort(l_key, l_val, pos + 0, pos + stride, dir);
      stride = stride >> 1
    }
    size = size << 1
  }

  // Odd / even arrays of LOCAL_SIZE_LIMIT elements
  // sorted in opposite directions
  {
    let dir = (my.block_idx.x & 1) != 0;
    let mut stride = LOCAL_SIZE_LIMIT / 2;
    while stride > 0 {
      syncthreads();
      let pos = 2 * my.thread_idx.x - (my.thread_idx.x & (stride - 1));
      two_sort(l_key, l_val, pos + 0, pos + stride, dir);
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

struct BitonicMergeGlobal {
  length   : usize,
  size     : usize,
  stride   : usize,
  sort_dir : bool,
  d_key    : Box<[u32]>,
  d_val    : Box<[u32]>
}

impl Code for BitonicMergeGlobal {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut BitonicMergeGlobal) {
  let global_id = my.block_dim.x * my.block_idx.x + my.thread_idx.x;
  let global_comparator_i = global_id;
  let comparator_i = global_comparator_i & (params.length / 2 - 1);

  // Bitonic merge
  let dir = params.sort_dir ^ ( (comparator_i & (params.size / 2)) != 0 );
  let pos =
    2 * global_comparator_i - (global_comparator_i & (params.stride - 1));

  let mut key_a = params.d_key[pos + 0];
  let mut val_a = params.d_val[pos + 0];
  let mut key_b = params.d_key[pos + params.stride];
  let mut val_b = params.d_val[pos + params.stride];

  if ((key_a > key_b) == dir) {
    core::mem::swap(&mut key_a, &mut key_b);
    core::mem::swap(&mut val_a, &mut val_b);
  }
  nocl_converge();

  params.d_key[pos + 0] = key_a;
  params.d_val[pos + 0] = val_a;
  params.d_key[pos + params.stride] = key_b;
  params.d_val[pos + params.stride] = val_b;
}

}

struct BitonicMergeLocal {
  length       : usize,
  size         : usize,
  stride_arg   : usize,
  sort_dir     : bool,
  d_key_arg    : Box<[u32]>,
  d_val_arg    : Box<[u32]>
}

//Combined bitonic merge steps for
//'size' > LOCAL_SIZE_LIMIT and 'stride' = [1 .. LOCAL_SIZE_LIMIT / 2]
impl Code for BitonicMergeLocal {

#[inline(always)]
fn run (my : &My, shared : &mut Scratch, params: &mut BitonicMergeLocal) {
  let mut l_key = alloc::<u32>(shared, LOCAL_SIZE_LIMIT);
  let mut l_val = alloc::<u32>(shared, LOCAL_SIZE_LIMIT);

  let d_srckey = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  let d_srcval = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  let d_dstkey = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  let d_dstval = my.block_idx.x * LOCAL_SIZE_LIMIT + my.thread_idx.x;
  l_key[my.thread_idx.x + 0] = params.d_key_arg[d_srckey+0];
  l_val[my.thread_idx.x + 0] = params.d_val_arg[d_srcval+0];
  l_key[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)] =
    params.d_key_arg[d_srckey + (LOCAL_SIZE_LIMIT / 2)];
  l_val[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)] =
    params.d_val_arg[d_srcval + (LOCAL_SIZE_LIMIT / 2)];

  let global_id = my.block_dim.x * my.block_idx.x + my.thread_idx.x;
  let comparator_i = global_id & ((params.length / 2) - 1);
  let dir = params.sort_dir ^ ( (comparator_i & (params.size / 2)) != 0 );
  let mut stride = params.stride_arg;
  while stride > 0 {
    syncthreads();
    let pos = 2 * my.thread_idx.x - (my.thread_idx.x & (stride - 1));
    two_sort(l_key, l_val, pos + 0, pos + stride, dir);
    stride = stride >> 1
  }

  syncthreads();
  params.d_key_arg[d_dstkey + 0] = l_key[my.thread_idx.x + 0];
  params.d_val_arg[d_dstval + 0] = l_val[my.thread_idx.x + 0];
  params.d_key_arg[d_dstkey + (LOCAL_SIZE_LIMIT / 2)] =
    l_key[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)];
  params.d_val_arg[d_dstval + (LOCAL_SIZE_LIMIT / 2)] =
    l_val[my.thread_idx.x + (LOCAL_SIZE_LIMIT / 2)];
}

}

#[entry]
fn main() -> ! {
  nocl_init();

  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 1 << 13;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1 << 18;

  // Input and output vectors
  let mut srckeys : Vec<u32> = vec![0; N];
  let mut srcvals : Vec<u32> = vec![0; N];
  let mut dstkeys : Vec<u32> = vec![0; N];
  let mut dstvals : Vec<u32> = vec![0; N];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0 .. N {
    srckeys[i] = rand15(&mut seed);
    srcvals[i] = rand15(&mut seed)
  }

  // Blck & grid dimensions for all kernel invocations
  let dims =
    Dims {
      block_dim: Dim2 { x: LOCAL_SIZE_LIMIT/2,
                        y: 1 },
      grid_dim: Dim2 { x: N/LOCAL_SIZE_LIMIT, y: 1 }
    };

  // Launch BitonicSortLocal
  let mut dstkeys : Box<[u32]> = dstkeys.into();
  let mut dstvals : Box<[u32]> = dstvals.into();
  let mut bsl_params =
    BitonicSortLocal {
      d_srckey_arg : srckeys.into(),
      d_srcval_arg : srcvals.into(),
      d_dstkey_arg : dstkeys,
      d_dstval_arg : dstvals
    };
  let bsl_params = nocl_run_kernel_verbose(dims, bsl_params);
  let mut dstkeys = bsl_params.d_dstkey_arg;
  let mut dstvals = bsl_params.d_dstval_arg;

  let mut size = 2 * LOCAL_SIZE_LIMIT;
  while size <= N {
    let mut stride = size / 2;
    while stride > 0 {
      if stride >= LOCAL_SIZE_LIMIT {
        // Launch BitonicMergeGlobal
        let mut bmg_params =
          BitonicMergeGlobal {
            length   : N,
            size     : size,
            stride   : stride,
            sort_dir : true,
            d_key    : dstkeys,
            d_val    : dstvals
          };
        let bmg_params = nocl_run_kernel_verbose(dims, bmg_params);
        dstkeys = bmg_params.d_key;
        dstvals = bmg_params.d_val;
      }
      else {
        // Launch BitonicMergeLocal
        let mut bml_params =
          BitonicMergeLocal {
            length      : N,
            size        : size,
            stride_arg  : stride,
            sort_dir    : true,
            d_key_arg   : dstkeys,
            d_val_arg   : dstvals
          };
        let bml_params = nocl_run_kernel_verbose(dims, bml_params);
        dstkeys = bml_params.d_key_arg;
        dstvals = bml_params.d_val_arg;
        break
      }
      stride = stride >> 1
    }
    size = size << 1
  }

  // Check result
  let mut ok = true;
  for i in 0 .. N-1 {
    ok = ok && dstkeys[i] <= dstkeys[i+1]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
