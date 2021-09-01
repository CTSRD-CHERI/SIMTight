/**
 * A simple stencil computation that computes the sum of each point 
 * and its four direct neighbours in a 2D grid.
 * The implementation accesses memory in an aligned way and uses shared memory.
 * Each thread computes a single element in the output buffer.
 *
 * Author: Paul Metzger 
 */

#include <NoCL.h>

#define DEBUG false

void populate_in_buf(int *in_buf, int x_size, int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x) in_buf[y * x_size + x] = x * y;
  }
}

// Generate a 'golden output' to check if the output computed
// by the GPU kernel is correct.
void generate_golden_output(int *in_buf, int *golden_out, int x_size, int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x) {
      const int ind = y * x_size + x;

      int result = in_buf[ind];
      if (x < x_size - 1) result += in_buf[y * x_size + x + 1];
      if (x > 0)          result += in_buf[y * x_size + x - 1];
      if (y < y_size - 1) result += in_buf[(y + 1) * x_size + x];
      if (y > 0)          result += in_buf[(y - 1) * x_size + x];
      golden_out[ind] = result;
    }
  }
}

// Check if the results computed by the GPU kernel match
// the golden output.
bool check_output(int *out_buf, int *golden_buf, int buf_size) {
  for (int i = 0; i < buf_size; ++i) {
    if (out_buf[i] != golden_buf[i]) {
      puts("Detected an error at index: "); puthex(i); putchar('\n');
      puts("Expected value: "); puthex(golden_buf[i]); putchar('\n');
      puts("Computed value: "); puthex(out_buf[i]);    putchar('\n');
      return false;
    }
  }
  return true;
}

struct SimpleStencil : Kernel {
  int x_size, y_size;
  int *out_buf, *in_buf;

  void kernel() {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int ind = y * x_size + x;

    // Load values into local memory
    auto c = shared.array<int>(blockDim.y, blockDim.x);
    c[threadIdx.y][threadIdx.x] = in_buf[ind];
    __syncthreads();

    int result = c[threadIdx.y][threadIdx.x];
    if (x < x_size - 1) {
      if (threadIdx.x == blockDim.x - 1) result += in_buf[ind + 1];
      else result += c[threadIdx.y][threadIdx.x + 1];
    }
    noclConverge();

    if (x > 0) {
      if (threadIdx.x == 0) result += in_buf[ind - 1];
      else result += c[threadIdx.y][threadIdx.x - 1];
    }
    noclConverge();

    if (y < y_size - 1) {
      if (threadIdx.y == blockDim.y - 1) result += in_buf[(y + 1) * x_size + x];
      else result += c[threadIdx.y + 1][threadIdx.x];
    }
    noclConverge();

    if (y > 0) {
      if (threadIdx.y == 0) result += in_buf[(y - 1) * x_size + x];
      else result += c[threadIdx.y - 1][threadIdx.x];
    }
    noclConverge();

    out_buf[ind] = result;
  }
};

int main() {
  // Are we in a simulation?
  bool isSim = getchar();

  // Problem size
  int buf_size_x = 1024;
  int buf_size_y = 1024;
  if (isSim) {
    buf_size_x = 64;
    buf_size_y = 64;
  }
  
  // Check problem size
  if ((buf_size_x % SIMTLanes) != 0 || (buf_size_y % SIMTLanes) != 0) {
    puts("Error buf_size_x must be a multiple of SIMTLanes and buf_size_y must be a multiple of SIMTWarps");
    return 1;
  }

  const int buf_size = buf_size_x * buf_size_y;
  simt_aligned int in_buf[buf_size];
  simt_aligned int out_buf[buf_size];
  int golden_out_buf[buf_size];

  // Prepare buffers
  // Zero out the ouput buffer. This makes it easier to spot bugs.
  for (int i = 0; i < buf_size; ++i) out_buf[i] = 0;
  populate_in_buf(in_buf, buf_size_x, buf_size_y);
  
  // Generate a golden output to check if
  // the results computed by the GPU kernel are correct (see below).
  generate_golden_output(in_buf, golden_out_buf, buf_size_x, buf_size_y);

  // Run the GPU kernel
  SimpleStencil k;
  k.blockDim.x = SIMTLanes; 
  k.blockDim.y = SIMTWarps;
  k.gridDim.x  = buf_size_x / SIMTLanes;
  k.gridDim.y  = buf_size_y / SIMTWarps;
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
