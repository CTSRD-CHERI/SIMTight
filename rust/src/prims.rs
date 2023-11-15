// Emit word to console in simulation
pub fn sim_emit(word: u32) -> () {
  unsafe {
    #[cfg(target_arch = "riscv32")]
    core::arch::asm!(
      "csrw 0x800, {w}",
         w = in(reg) word,
    );
  }
}

// Can we a send byte to the UART?
pub fn uart_can_put() -> bool {
  let can_put : u32;
  unsafe {
    #[cfg(target_arch = "riscv32")]
    core::arch::asm!(
      "csrrw {result}, 0x802, zero",
         result = out(reg) can_put,
    );
  }
  can_put != 0
}

// Send a byte to the UART, assuming uart_can_put() returned true.
pub fn uart_put(byte: u8) -> () {
  unsafe {
    #[cfg(target_arch = "riscv32")]
    core::arch::asm!(
      "csrw 0x803, {b}",
         b = in(reg) byte,
    );
  }
}

// Send a byte to the UART
pub fn putchar(byte: u8) -> () {
  while !uart_can_put() {};
  uart_put(byte)
}

// Send a string to the UART
pub fn putstr(s : &str) -> () {
  for c in s.bytes() {
    putchar(c)
  }
}

// Print an unsigned integer to the UART in hex format
pub fn puthex(input : u32) -> () {
  let mut x = input;
  for _i in 0..8 {
    let nibble = (x >> 28) as u8;
    putchar(if nibble > 9 {(b'a' - 10) + nibble} else {b'0' + nibble});
    x = x << 4;
  }
}
