// Primitives provided by SIMTight hardware

pub mod config;

// Emit word to console in simulation
pub fn sim_emit(word: u32) -> () {
  unsafe {
    core::arch::asm!(
      "csrw 0x800, {w}",
         w = in(reg) word,
    );
  }
}

// Can we send a byte to the UART?
pub fn uart_can_put() -> bool {
  let can_put : u32;
  unsafe {
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
    x = x << 4
  }
}

// Can we get a byte from the UART?
pub fn uart_can_get() -> bool {
  let can_get : u32;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0x804, zero",
         result = out(reg) can_get,
    );
  }
  can_get != 0
}

// Get a byte from the UART.
// Assumes uart_can_get() is true.
pub fn uart_get() -> u8 {
  let byte : u32;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0x805, zero",
         result = out(reg) byte,
    );
  }
  byte as u8
}

// Read a byte from the UART.
pub fn getchar() -> u8 {
  while !uart_can_get() {};
  uart_get()
}

// Assert condition.
// Print message and loop on failure.
pub fn assert(cond : bool, s : &str) -> () {
  if !cond { putstr(s); putchar(b'\n'); loop {} }
}

// Get id of calling thread.
#[inline(always)]
pub fn hart_id() -> u32 {
 let id : u32;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0xf14, zero",
         result = out(reg) id,
    );
  }
  id
}

// Increase SIMT nesting level
#[inline(always)]
pub fn simt_push() -> () {
  unsafe {
    // Custom instruction
    // Opcode: 0000000 rs2 rs1 000 rd 0001001, with rd=0, rs1=0, rs2=0
    core::arch::asm!(
      ".word 0x00050009"
    )
  }
}

// Decrease SIMT nestling level
#[inline(always)]
pub fn simt_pop() -> () {
  unsafe {
    // Custom instruction
    // Opcode: 0000000 rs2 rs1 001 rd 0001001, with rd=0, rs1=0, rs2=0
    core::arch::asm!(
      ".word 0x00051009"
    )
  }
}

// Mark a convergence point
#[inline(always)]
pub fn simt_converge() -> () {
  simt_pop();
  simt_push()
}

// Barrier synchonrisation.
// Assumes all threads in warp have converged.
#[inline(always)]
pub fn simt_barrier() -> () {
  unsafe {
    core::arch::asm!(
      "csrw 0x830, zero"
    )
  }
}

// Terminate current warp with success.
// Assumes all threads in warp have converged.
pub fn simt_terminate_success() -> () {
  unsafe {
    core::arch::asm!(
      "csrw 0x830, 3"
    )
  }
}

// Terminate current warp with failure.
// Assumes all threads in warp have converged.
pub fn simt_terminate_failure() -> () {
  unsafe {
    core::arch::asm!(
      "csrw 0x830, 1"
    )
  }
}

// Get address of kernel structure to execute.
pub fn simt_get_kernel_addr() -> usize {
  let addr : usize;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0x831, zero",
         result = out(reg) addr,
    );
  }
  addr
}

// Assert condition on SIMT core.
// Emit in simulation and loop on failure.
pub fn simt_assert(cond : bool) {
  if !cond { sim_emit(0xdeadbeef); loop {} }
}

// Can issue command to SIMT core?
pub fn simt_host_can_put() -> bool {
  let can_put : u32;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0x820, zero",
         result = out(reg) can_put,
    );
  }
  can_put != 0
}

// Set address of kernel sturucture to execute.
// Assumes simt_host_can_put() is true.
pub fn simt_host_set_kernel_addr(addr : usize) -> () {
  unsafe {
    core::arch::asm!(
      "csrw 0x826, {a}",
         a = in(reg) addr,
    );
  }
}

// Set number of warps per block.
// Assumes simt_host_can_put() is true.
// (A block is a group of threads that synchronise on a barrier.)
// (A value of 0 indicates all warps.)
pub fn simt_host_set_warps_per_block(n : u32) {
  unsafe {
    core::arch::asm!(
      "csrw 0x827, {num}",
         num = in(reg) n,
    );
  }
}

// Start a kernel on the SIMT core.
// Assumes simt_host_can_put() is true.
pub fn simt_host_start_kernel(pc : usize) {
  unsafe {
    core::arch::asm!(
      "csrw 0x823, {addr}",
         addr = in(reg) pc,
    );
  }
}

// Ask for the value of the given stat counter.
// Assumes simt_host_can_put() is true.
pub fn simt_host_ask_stat(id : u32) {
  unsafe {
    core::arch::asm!(
      "csrw 0x828, {x}",
         x = in(reg) id,
    );
  }
}

// Can receive response from SIMT core?
pub fn simt_host_can_get() -> bool {
 let can_get : u32;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0x824, zero",
         result = out(reg) can_get,
    );
  }
  can_get != 0
}

// Receive response from SIMT core.
// Assumes simt_host_can_get() is true.
pub fn simt_host_get() -> u32 {
  let get : u32;
  unsafe {
    core::arch::asm!(
      "csrrw {result}, 0x825, zero",
         result = out(reg) get,
    );
  }
  get
}

// Memory fence
pub fn fence() -> () {
  unsafe {
    core::arch::asm!(
      "fence rw, rw",
    );
  }
}

// Cache line flush
pub fn cache_flush_line(addr : u32) -> () {
 unsafe {
    // Custom instruction
    // Opcode: 0000000 rs2 rs1 000 rd 0001000, with rd=0, rs1=x10, rs2=0
    core::arch::asm!(
      "mv x10, {a}
       .word 0x00050008",
         a = in(reg) addr,
         out("x10") _
    )
  }
}

// Full cache flush
pub fn cache_flush_full() -> () {
  // Flush each line
  let num_lines : u32 = 1 << config::SBDCACHE_LOG_LINES;
  for i in 0..num_lines {
    let addr : u32 = i * config::DRAMBEAT_BYTES;
    cache_flush_line(addr);
  }
  // Issue a fence to ensure flush is complete
  fence()
}

// Send null to UART and jump to boot loader
pub fn end() -> ! {
  putchar(0);
  unsafe {
    core::arch::asm!(
      "jr {base}",
         base = in(reg) config::MEM_BASE,
    )
  }
  loop {}
}
