/**
 * A simple test stencil computation that computes the sum of each point 
 * and its four direct neighbours in a 2D grid.
 *
 * Author: Paul Metzger 
 */

#include <NoCL.h>

#define DEBUG false

#define STENCIL_REACH 1

void populate_in_buf(int *in_buf, int x_size, int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x) {
      // Elements on the border of the iteration space
      // have a value of 0.
      if (x == 0 || y == 0 || x == x_size - 1 || y == y_size - 1) {
        in_buf[y * y_size + x] = 0;
      } else {
        in_buf[y * y_size + x] = 1;
      }
    }
  }
}

// Generate a 'golden output' to check if the output computed
// by the GPU kernel is correct.
void generate_golden_output(int *in_buf, int *golden_out, int x_size, int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x) {
      if (x == 0 || y == 0 || x == x_size - 1 || y == y_size - 1) {
        continue; // Skip over border values
      } else {
        golden_out[y * y_size + x] = in_buf[y * y_size + x] + 
                                     in_buf[y * y_size + x + 1] +
                                     in_buf[y * y_size + x - 1] + 
                                     in_buf[(y + 1) * y_size + x] +
                                     in_buf[(y - 1) * y_size + x];
      }
    }
  }
}

// Check if the results computed by the GPU kernel match
// the golden output.
bool check_output(int *out_buf, int *golden_buf, int buf_size) {
  for (int i = 0; i < buf_size; ++i) {
    if (out_buf[i] != golden_buf[i]) return false;
  }
  return true;
}

struct SimpleStencil : Kernel {
  int x_size = 0;
  int y_size = 0;
  int *out_buf, *in_buf;

  void kernel() {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && y > 0 && x < x_size - 1 && y < y_size - 1) {
      out_buf[y * y_size + x] = in_buf[y * y_size + x] +
                                in_buf[y * y_size + x + 1] + 
                                in_buf[y * y_size + x - 1] + 
                                in_buf[(y + 1) * y_size + x] +
                                in_buf[(y - 1) * y_size + x];
    }
  }
};

int main() {
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int problem_size_x = 512;
  int problem_size_y = 512;
  if (isSim) {
    problem_size_x = 32;
    problem_size_y = 32;
  }

  const int buf_size_x = problem_size_x + 2 * STENCIL_REACH;
  const int buf_size_y = problem_size_x + 2 * STENCIL_REACH;
  const int buf_size = buf_size_x * buf_size_y;

  simt_aligned int in_buf[buf_size];
  simt_aligned int out_buf[buf_size];
  int golden_out_buf[buf_size];

  // Prepare buffers
  // Zero out the ouput buffers
  for (int i = 0; i < buf_size; ++i) {
    out_buf[i] = 0;
    golden_out_buf[i] = 0;
  }
  populate_in_buf(in_buf, buf_size_x, buf_size_y);
  
  // Generate the golden output to check if
  // the results computed by the GPU kernel are correct (see below).
  generate_golden_output(in_buf, golden_out_buf, buf_size_x, buf_size_y);

  // Do computation on the GPU
  SimpleStencil k;
  k.blockDim.x = SIMTLanes;
  k.blockDim.y = SIMTWarps;
  k.gridDim.x  = problem_size_x / SIMTLanes + (buf_size_x % SIMTLanes ? 1 : 0);
  k.gridDim.y  = problem_size_y / SIMTWarps + (buf_size_y % SIMTWarps ? 1 : 0);
  k.x_size     = buf_size_x;
  k.y_size     = buf_size_y;
  k.out_buf    = out_buf;
  k.in_buf     = in_buf;
  if (DEBUG) puts("Kernel running... ");
  noclRunKernelAndDumpStats(&k);
  if (DEBUG) puts("Done\n");

  // Check result
  bool ok = check_output(out_buf, golden_out_buf, buf_size_x * buf_size_y);
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
