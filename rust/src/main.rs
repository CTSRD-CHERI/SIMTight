#![no_std]
#![no_main]
#![allow(unused)]

// External crates being used
// ==========================

extern crate panic_halt;
use riscv_rt::entry;

// Local imports
// =============

pub mod prims;
use prims::*;

pub mod nocl;
use nocl::*;

pub mod rand;
use rand::*;

// Benchmark
// =========

struct VecAdd<'t> {
  len    : usize,
  a      : &'t [u32],
  b      : &'t [u32],
  result : &'t mut[u32]
}

impl Code for VecAdd<'_> {
  #[inline(always)]
  fn run<'t> (my : &My, shared : &mut Mem, params: &mut VecAdd<'t>) {
    for i in (my.thread_idx.x .. params.len).step_by(my.block_dim.x) {
      params.result[i] = params.a[i] + params.b[i]
    }
  }
}

#[entry]
fn main() -> ! {
  // Vector size for benchmarking
  const N : usize = 3000;

  // Input and output vectors
  let mut a : NoCLAligned<[u32; N]> = nocl_aligned([0; N]);
  let mut b : NoCLAligned<[u32; N]> = nocl_aligned([0; N]);
  let mut result : NoCLAligned<[u32; N]> = nocl_aligned([0; N]);

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..N {
    a.val[i] = rand15(&mut seed);
    b.val[i] = rand15(&mut seed);
  }

  // Use a single block of threads
  let dims =
    Dims {
      block_dim: Dim2 { x: prims::config::SIMT_WARPS *
                           prims::config::SIMT_LANES,
                        y: 1 },
      grid_dim: Dim2 { x: 1, y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    VecAdd { len    : N,
             a      : &a.val[..],
             b      : &b.val[..],
             result : &mut result.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  for i in 0 .. N {
    ok = ok && result.val[i] == a.val[i] + b.val[i]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  putchar(0);
  loop {}
}
