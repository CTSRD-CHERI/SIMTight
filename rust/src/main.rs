#![no_std]
#![no_main]

use riscv_rt::entry;

extern crate panic_halt;

pub mod prims;
use prims::*;

#[entry]
fn main() -> ! {
  sim_emit(0xdeadbeef);

  putstr("hello\n");
  puthex(0xcafebabe);
  putstr("\n");

  // The end
  putchar(0);
  loop {}
}
