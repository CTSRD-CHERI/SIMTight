/*

A simple stencil computation that computes the sum of each point and
its four direct neighbours in a 2D grid.  Each thread block
computes multiple rows of the output, minimising fetching of data from
global memory.

Authors: Paul Metzger, Matthew Naylor

*/

#include <NoCL.h>
#include <Rand.h>

#define DEBUG false

void populate_in_buf(int *in_buf, int x_size, int y_size) {
  uint32_t seed = 1;
  for (int y = 0; y < y_size; ++y)
    for (int x = 0; x < x_size; ++x)
      in_buf[y * x_size + x] = rand15(&seed);
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
  int *out_buf, *in_buf;

  // Shared local memory: data blocks to the left, middle, and
  // right of current output
  Array2D<int> buffer;

  void init() {
    declareShared(&buffer, blockDim.y, 3 * blockDim.x);
  }

  void kernel() {
    // Offsets for left, middle, and right blocks
    const int left = 0;
    const int middle = blockDim.x;
    const int right = 2 * blockDim.x;

    // Base index of current row
    int rowBase = blockIdx.y * blockDim.y * x_size + threadIdx.y * x_size;

    // Initially, left block is zeros
    buffer[threadIdx.y][left + threadIdx.x] = 0;

    // Initially, middle block is first block in row
    buffer[threadIdx.y][middle + threadIdx.x] = in_buf[rowBase + threadIdx.x];

    // Visit every block in row
    for (int i = 0; i < x_size; i += blockDim.x) {

      // Index of value being computed by this thread
      int idx = rowBase + i + threadIdx.x;
 
      // Fetch right block (won't diverge: row is multiple of block width)
      if (i + blockDim.x == x_size)
        buffer[threadIdx.y][right + threadIdx.x] = 0;
      else
        buffer[threadIdx.y][right + threadIdx.x] = in_buf[idx + blockDim.x];

      __syncthreads();

      // Fetch blocks above and below (won't diverge: conditioned on Y index)
      int above = threadIdx.y != 0 ?
                    buffer[threadIdx.y-1][middle + threadIdx.x] :
                      blockIdx.y == 0 ? 0 : in_buf[idx - x_size];
      int below = threadIdx.y != blockDim.y-1 ?
                    buffer[threadIdx.y+1][middle + threadIdx.x] :
                      blockIdx.y == gridDim.y-1 ? 0 : in_buf[idx + x_size];

      // Middle plus x, used often below
      int mx = middle + threadIdx.x;

      // Write output
      out_buf[idx] =
          above
        + below
        + buffer[threadIdx.y][mx - 1]
        + buffer[threadIdx.y][mx]
        + buffer[threadIdx.y][mx + 1];

      __syncthreads();

      // Shift buffer
      buffer[threadIdx.y][left + threadIdx.x] = buffer[threadIdx.y][mx];
      buffer[threadIdx.y][mx] = buffer[threadIdx.y][right + threadIdx.x];
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
  k.blockDim.x   = SIMTLanes; 
  k.blockDim.y   = 4;
  k.gridDim.x    = 1;
  k.gridDim.y    = buf_size_y/k.blockDim.y;
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
