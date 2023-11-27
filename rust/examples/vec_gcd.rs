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

// Euclid's algorithm
fn gcd(x_arg : u32, y_arg : u32) -> u32 {
  let mut x = x_arg;
  let mut y = y_arg;
  nocl_push();
  while x != y {
    nocl_push();
    if (x > y) {
      x = x-y
    }
    else {
      y = y-x
    }
    nocl_pop();
  }
  nocl_pop();
  return x;
}

struct VecGCD<'t> {
  len    : usize,
  a      : &'t [u32],
  b      : &'t [u32],
  result : &'t mut[u32]
}

impl Code for VecGCD<'_> {
  #[inline(always)]
  fn run<'t> (my : &My, shared : &mut Mem, params: &mut VecGCD<'t>) {
    for i in (my.thread_idx.x .. params.len).step_by(my.block_dim.x) {
      params.result[i] = gcd(params.a[i], params.b[i])
    }
  }
}

#[entry]
fn main() -> ! {
  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 100;
  #[cfg(feature = "large_data_set")]
  const N : usize = 100000;

  // Input and output vectors
  let mut a : NoCLAligned<[u32; N]> = nocl_aligned([0; N]);
  let mut b : NoCLAligned<[u32; N]> = nocl_aligned([0; N]);
  let mut result : NoCLAligned<[u32; N]> = nocl_aligned([0; N]);

  // Initialise inputs
  let mut seed : u32 = 100;
  for i in 0..N {
    a.val[i] = 1 + (rand15(&mut seed) & 0xff);
    b.val[i] = 1 + (rand15(&mut seed) & 0xff);
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
    VecGCD { len    : N,
             a      : &a.val[..],
             b      : &b.val[..],
             result : &mut result.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  for i in 0 .. N {
    ok = ok && result.val[i] == gcd(a.val[i], b.val[i])
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
