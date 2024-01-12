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

struct VecAdd {
  a      : Box<[u32]>,
  b      : Box<[u32]>,
  result : Box<[u32]>
}

impl Code for VecAdd {
  #[inline(always)]
  fn run (my : &My, shared : &mut Scratch, params: &mut VecAdd) {
    for i in (my.thread_idx.x .. params.result.len()).step_by(my.block_dim.x) {
      params.result[i] = params.a[i] + params.b[i]
    }
  }
}

#[entry]
fn main() -> ! {
  nocl_init();

  // Vector size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const N : usize = 3000;
  #[cfg(feature = "large_data_set")]
  const N : usize = 1000000;

  // Input and output data
  let mut a : Vec<u32> = vec![0; N];
  let mut b : Vec<u32> = vec![0; N];
  let mut result : Vec<u32> = vec![0; N];

  // Initialise inputs
  let mut seed : u32 = 1;
  for i in 0..N {
    a[i] = rand15(&mut seed);
    b[i] = rand15(&mut seed);
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
    VecAdd {
      a      : a.into(),
      b      : b.into(),
      result : result.into()
    };

  // Invoke kernel
  let params = nocl_run_kernel_verbose(dims, params);

  // Check result
  let mut ok = true;
  for i in 0 .. N {
    ok = ok && params.result[i] == params.a[i] + params.b[i]
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
