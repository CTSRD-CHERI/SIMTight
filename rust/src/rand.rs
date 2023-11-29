pub fn rand15(seed : &mut u32) -> u32 {
  *seed = (*seed * 1664525 + 1013904223) & 0x7fffffff;
  *seed >> 16
}
