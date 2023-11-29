// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

// Benchmark
// =========

const BLOCK_SIZE: usize = prims::config::SIMT_LANES as usize;

struct MatMul<'t> {
  a_width : usize,
  b_width : usize,
  a       : &'t [i32],
  b       : &'t [i32],
  c       : &'t mut [i32]
}

impl Code for MatMul<'_> {

#[inline(always)]
fn run<'t> (my : &My, shared : &mut Mem, params: &mut MatMul<'t>) {
  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  let mut a_sub = alloc::<i32>(shared, BLOCK_SIZE * BLOCK_SIZE);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  let mut b_sub = alloc::<i32>(shared, BLOCK_SIZE * BLOCK_SIZE);

  // Block index
  let bx = my.block_idx.x;
  let by = my.block_idx.y;

  // Thread index
  let tx = my.thread_idx.x;
  let ty = my.thread_idx.y;

  // Index of the first sub-matrix of A processed by the block
  let a_begin = params.a_width * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  let a_end = a_begin + params.a_width - 1;

  // Step size used to iterate through the sub-matrices of A
  let a_step = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  let b_begin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  let b_step = BLOCK_SIZE * params.b_width;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  let mut c_sub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  let mut a = a_begin;
  let mut b = b_begin;
  while a <= a_end {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    a_sub[ty*BLOCK_SIZE+tx] = params.a[a + params.a_width * ty + tx];
    b_sub[ty*BLOCK_SIZE+tx] = params.b[b + params.b_width * ty + tx];

    // Synchronize to make sure the matrices are loaded
    syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    for k in 0 .. BLOCK_SIZE {
      c_sub += a_sub[ty*BLOCK_SIZE+k] * b_sub[k*BLOCK_SIZE+tx]
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    syncthreads();

    a += a_step;
    b += b_step
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  let c = params.b_width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  params.c[c + params.b_width * ty + tx] = c_sub
}

}

#[entry]
fn main() -> ! {
  // Matrix size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const SIZE : usize = 64;
  #[cfg(feature = "large_data_set")]
  const SIZE : usize = 256;

  // Input and output matrix data
  let mut mat_a : NoCLAligned<[i32; SIZE*SIZE]> =
        nocl_aligned([0; SIZE*SIZE]);
  let mut mat_b : NoCLAligned<[i32; SIZE*SIZE]> =
        nocl_aligned([0; SIZE*SIZE]);
  let mut mat_c : NoCLAligned<[i32; SIZE*SIZE]> =
        nocl_aligned([0; SIZE*SIZE]);
  let mut mat_check : NoCLAligned<[i32; SIZE*SIZE]> =
        nocl_aligned([0; SIZE*SIZE]);

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..SIZE {
    for j in 0..SIZE {
      mat_a.val[i*SIZE+j] = (rand15(&mut seed) & 0xff) as i32;
      mat_b.val[i*SIZE+j] = (rand15(&mut seed) & 0xff) as i32
    }
  }

  // Block/grid dimensions
  let dims =
    Dims {
      block_dim: Dim2 { x: BLOCK_SIZE,
                        y: BLOCK_SIZE },
      grid_dim: Dim2 { x: SIZE / BLOCK_SIZE,
                       y: SIZE / BLOCK_SIZE }
    };
 
  // Kernel parameters
  let mut params =
    MatMul { a_width : SIZE,
             b_width : SIZE,
             a       : &mat_a.val[..],
             b       : &mat_b.val[..],
             c       : &mut mat_c.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  for i in 0 .. SIZE {
    for j in 0 .. SIZE {
      for k in 0 .. SIZE {
        mat_check.val[i*SIZE+j] += mat_a.val[i*SIZE+k] * mat_b.val[k*SIZE+j]
      }
    }
  }
  for i in 0 .. SIZE {
    for j in 0 .. SIZE {
      ok = ok && mat_check.val[i*SIZE+j] == mat_c.val[i*SIZE+j]
    }
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
