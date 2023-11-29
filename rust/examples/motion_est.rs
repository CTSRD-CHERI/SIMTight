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

const RADIUS: usize = 4;

struct MotionEst<'t> {
  frame_width   : usize,
  frame_height  : usize,
  prev_frame    : &'t [i32],
  current_frame : &'t [i32],

   // Origin and dimensions of region being processed
  region_origin_x   : usize,
  region_origin_y   : usize,
  region_log_width  : usize,
  region_log_height : usize,

  // Output SAD per motion vector per pixel block
  sads : &'t mut [i32]
}

impl Code for MotionEst<'_> {

#[inline(always)]
fn run<'t> (my : &My, shared : &mut Mem, params: &mut MotionEst<'t>) {
  let region_width = 1 << params.region_log_width;
  let region_height = 1 << params.region_log_height;
  let mut current = alloc::<i32>(shared, region_width * region_height);
  let prev_width = region_width + 2*prims::config::SIMT_LANES;
  let mut prev = alloc::<i32>(shared,
    (region_height + 2 * RADIUS) * prev_width);

  // Load current frame's region
  for y in 0 .. region_height {
    nocl_push();
    for x in (my.thread_idx.x .. region_width).step_by(my.block_dim.x) {
      let fy = params.region_origin_y + y;
      let fx = params.region_origin_x + x;
      current[y*region_width + x] =
        params.current_frame[fy*params.frame_width + fx]
    }
    nocl_pop()
  }
  syncthreads();

  // Load previous frame's region (and required surroundings)
  for y in 0 .. region_height + 2*RADIUS {
    nocl_push();
    for x in (my.thread_idx.x .. region_width +
                2*prims::config::SIMT_LANES).step_by(my.block_dim.x) {
      let fy = params.region_origin_y as i32 + ((y as i32) - (RADIUS as i32));
      let fx = params.region_origin_x as i32 +
        ((x as i32) - (prims::config::SIMT_LANES as i32));
      let outside = (fy < 0) | (fy >= params.frame_height as i32) |
                    (fx < 0) | (fx >= params.frame_width as i32);
      nocl_push();
      if outside {
        prev[y*prev_width + x] = 0
      }
      else {
        prev[y*prev_width + x] =
          params.prev_frame[(fy as usize) * params.frame_width +
                            (fx as usize)]
      }
      nocl_pop();
    }
    nocl_pop();
  }
  syncthreads();

  // Compute all SADs
  let num_blocks_x = region_width >> 2;
  let num_blocks_y = region_height >> 2;
  let num_blocks = num_blocks_x * num_blocks_y;
  let outputs_per_block = (2*RADIUS+1) * (2*RADIUS+1);
  let num_outputs = num_blocks * outputs_per_block;
  for i in (my.thread_idx.x .. num_outputs).step_by(my.block_dim.x) {
    // Which block in current frame are we processing?
    let block_id = i / outputs_per_block;
    // Which motion vector are we computing?
    let vec_id = i - block_id * outputs_per_block;

    // Origin of current block
    let block_id_x = block_id & (num_blocks_x - 1);
    let block_id_y = block_id >> (params.region_log_width - 2);
    let current_x = block_id_x << 2;
    let current_y = block_id_y << 2;

    // Origin of previous block
    let vec_id_y = vec_id / (2*RADIUS+1);
    let vec_id_x = vec_id - vec_id_y * (2*RADIUS+1);
    let prev_x = current_x - RADIUS + prims::config::SIMT_LANES + vec_id_x;
    let prev_y = current_y + vec_id_y;

    // Compute SAD for current motion vector
    let mut sad : i32 = 0;
    for y in 0..4 {
      for x in 0..4 {
        let mut diff = current[(current_y+y)*region_width+current_x+x] -
                       prev[(prev_y+y)*prev_width + prev_x+x];
        nocl_push();
        if diff < 0 { diff = -diff };
        nocl_pop();
        sad += diff
      }
    }
    params.sads[i] = sad
  }
}

}

#[entry]
fn main() -> ! {
  // Matrix size for benchmarking
  #[cfg(not(feature = "large_data_set"))]
  const LOG_WIDTH : usize = 3;
  #[cfg(not(feature = "large_data_set"))]
  const LOG_HEIGHT : usize = 3;
  #[cfg(feature = "large_data_set")]
  const LOG_WIDTH : usize = 6;
  #[cfg(feature = "large_data_set")]
  const LOG_HEIGHT : usize = 6;

  const WIDTH : usize = 1 << LOG_WIDTH;
  const HEIGHT : usize = 1 << LOG_HEIGHT;

  // Number of SADs being computed
  // One per motion vector per block
  const NUM_OUTPUTS : usize = (WIDTH/4)*(HEIGHT/4)*(2*RADIUS+1)*(2*RADIUS+1);

  // Input frames and output SADs
  let mut current_frame : NoCLAligned<[i32; WIDTH*HEIGHT]> =
        nocl_aligned([0; WIDTH*HEIGHT]);
  let mut prev_frame : NoCLAligned<[i32; WIDTH*HEIGHT]> =
        nocl_aligned([0; WIDTH*HEIGHT]);
  let mut sads : NoCLAligned<[i32; NUM_OUTPUTS]> =
        nocl_aligned([0; NUM_OUTPUTS]);

  // Prepare inputs
  let mut seed : u32 = 1;
  for y in 0..HEIGHT {
    for x in 0..WIDTH {
      current_frame.val[y*WIDTH + x] = (rand15(&mut seed) & 0xff) as i32;
      prev_frame.val[y*WIDTH + x] = (rand15(&mut seed) & 0xff) as i32
    }
  }

  // Block/grid dimensions
  let dims =
    Dims {
      block_dim: Dim2 { x: prims::config::SIMT_LANES *
                           prims::config::SIMT_WARPS,
                        y: 1 },
      grid_dim: Dim2 { x: 1, y: 1 }
    };
 
  // Kernel parameters
  let mut params =
    MotionEst { frame_width       : WIDTH,
                frame_height      : HEIGHT,
                region_origin_x   : 0,
                region_origin_y   : 0,
                region_log_width  : LOG_WIDTH,
                region_log_height : LOG_HEIGHT,
                current_frame     : &current_frame.val[..],
                prev_frame        : &prev_frame.val[..],
                sads              : &mut sads.val[..] };

  // Invoke kernel
  nocl_run_kernel_verbose(&dims, &mut params);

  // Check result
  let mut ok = true;
  let mut out_count = 0;
  for cy in (0 .. HEIGHT).step_by(4) {
    for cx in (0 .. WIDTH).step_by(4) {
      for py in (cy as i32 - RADIUS as i32) ..
                (cy as i32 + (RADIUS as i32) + 1) {
        for px in (cx as i32 - RADIUS as i32) ..
                  (cx as i32 + RADIUS as i32 +1) {
          let mut sad = 0;
          for y in 0..4 {
            for x in 0..4 {
              let mut diff = current_frame.val[(cy+y) * WIDTH + cx + x];
              if (py + (y as i32) >= 0 && py + (y as i32) < (HEIGHT as i32) &&
                  px + (x as i32) >= 0 && px + (x as i32) < (WIDTH as i32)) {
                diff = diff - prev_frame.val[((py as usize) + y) *
                         WIDTH + (px as usize) + x];
              }
              if diff < 0 { diff = -diff };
              sad += diff
            }
          }
          ok = ok && sads.val[out_count] == sad;
          out_count += 1
        }
      }
    }
  }

  // Display result
  putstr("Self test: ");
  putstr(if ok { "PASSED" } else { "FAILED" });
  putchar(b'\n');

  // The end
  end()
}
