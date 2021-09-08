/*

A simple stencil computation that computes the sum of each point and
its four direct neighbours in a 2D grid.  A single thread block
computes multiple rows of the output, minimising fetching of data from
global memory.

Authors: Paul Metzger, Matthew Naylor

*/

#include <NoCL.h>

#define DEBUG false

void populate_in_buf(int *in_buf, int x_size, int y_size) {
  for (int y = 0; y < y_size; ++y) {
    for (int x = 0; x < x_size; ++x)
      in_buf[y * x_size + x] = x * y;
  }
}

// Generate a 'golden output' to check if the output computed
// by the GPU kernel is correct.
void generate_golden_output(int *in_buf, int *golden_out,
                            int x_size, int y_size) {
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
      puts("Detected an error at index: ");
      puthex(i);
      putchar('\n');
      puts("Expected value: ");
      puthex(golden_buf[i]);
      putchar('\n');
      puts("Computed value: ");
      puthex(out_buf[i]);
      putchar('\n');
      return false;
    }
  }
  return true;
}

struct SimpleStencil : Kernel {
  unsigned x_size;
  unsigned y_size;
  unsigned rowsPerBlock;
  int *out_buf, *in_buf;

  void kernel() {
    // Data blocks to the left, middle, and right of current output
    auto buffer = shared.array<int>(rowsPerBlock, 3 * blockDim.x);

    // Offsets for left, middle, and right blocks
    const int left = 0;
    const int middle = blockDim.x;
    const int right = 2 * blockDim.x;

    // Initialise row buffer
    for (int j = 0; j < rowsPerBlock; j++) {
      // Base index of current row
      int rowBase = blockIdx.y * rowsPerBlock * x_size + j * x_size;

      // Initially, left block is zeros
      buffer[j][left + threadIdx.x] = 0;

      // Initially, middle block is first block in row
      buffer[j][middle + threadIdx.x] = in_buf[rowBase + threadIdx.x];
    }

    // Visit every block in row
    for (int i = 0; i < x_size; i += blockDim.x) {

      // Fetch right blocks
      for (int j = 0; j < rowsPerBlock; j++) {
        // Base index of current row
        int rowBase = blockIdx.y * rowsPerBlock * x_size + j * x_size;

        // Index of value being computed by this thread
        int idx = rowBase + i + threadIdx.x;
 
        // Fetch right block (won't diverge: row is multiple of block width)
        if (i + blockDim.x == x_size)
          buffer[j][right + threadIdx.x] = 0;
        else
          buffer[j][right + threadIdx.x] = in_buf[idx + blockDim.x];
      }

      // Compute outputs
      for (int j = 0; j < rowsPerBlock; j++) {
        // Base index of current row
        int rowBase = blockIdx.y * rowsPerBlock * x_size + j * x_size;

        // Index of value being computed by this thread
        int idx = rowBase + i + threadIdx.x;

        // Fetch blocks above and below (won't diverge: block height is 1)
        int above = j != 0 ? buffer[j-1][middle + threadIdx.x] :
                      blockIdx.y == 0 ? 0 : in_buf[idx - x_size];
        int below = j != rowsPerBlock-1 ? buffer[j+1][middle + threadIdx.x] :
                      blockIdx.y == gridDim.y-1 ? 0 : in_buf[idx + x_size];

        // Write output
        out_buf[idx] =
            above
          + below
          + buffer[j][middle + threadIdx.x - 1]
          + buffer[j][middle + threadIdx.x]
          + buffer[j][middle + threadIdx.x + 1];
      }

      // Shift buffer
      for (int j = 0; j < rowsPerBlock; j++) {
        buffer[j][left + threadIdx.x] = buffer[j][middle + threadIdx.x];
        buffer[j][middle + threadIdx.x] = buffer[j][right + threadIdx.x];
      }
    }
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

  // Ensure that the problem size is a multiple of SIMTLanes and SIMTWarps
  if ((buf_size_x % SIMTLanes) != 0 || (buf_size_y % SIMTWarps) != 0) {
    puts("Error: buf_size_x must be a multiple of SIMTLanes "
         "and buf_size_y must be a multiple of SIMTWarps");
    return 1;
  }

  const int buf_size = buf_size_x * buf_size_y;
  simt_aligned int in_buf[buf_size];
  simt_aligned int out_buf[buf_size];
  int golden_out_buf[buf_size];

  // Prepare buffers
  // Zero out the ouput buffers
  for (int i = 0; i < buf_size; ++i) out_buf[i] = 0;
  populate_in_buf(in_buf, buf_size_x, buf_size_y);

  // Generate the golden output to check if
  // the results computed by the GPU kernel are correct (see below).
  generate_golden_output(in_buf, golden_out_buf, buf_size_x, buf_size_y);

  // Do computation on the GPU
  SimpleStencil k;
  k.rowsPerBlock = 2;
  k.blockDim.x   = SIMTLanes; 
  k.blockDim.y   = 1;
  k.gridDim.x    = 1;
  k.gridDim.y    = buf_size_y/k.rowsPerBlock;
  k.x_size       = buf_size_x;
  k.y_size       = buf_size_y;
  k.out_buf      = out_buf;
  k.in_buf       = in_buf;
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
