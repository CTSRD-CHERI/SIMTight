// Transcribed from "Efficient Sparse Matrix-Vector Multiplication on
// CUDA" by Bell and Garland, NVIDIA Corporation.

#include <NoCL.h>
#include <Rand.h>

// Kernel for sparse matrix vector multipliation on ELLPACK format
// One thread per matrix row
struct SparseMatVecMul : Kernel {
  int num_rows;
  int num_cols;
  int num_cols_per_row;
  int* indices;
  int* data;
  int* x;
  int* y;

  void kernel() {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
      int dot = 0;
      for (int n = 0; n < num_cols_per_row; n++) {
        int col = indices[num_rows * n + row];
        int val = data[num_rows * n + row];
        dot += val * x[col];
      }
      y[row] = dot;
    }
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector and matrix dimensions for benchmarking
  // Should be powers of two
  int width = isSim ? 256 : 2048;
  int height = isSim ? 64 : 2048;

  // Sparsity of matrix (power of two)
  int sparsity = 8;
  int samplesPerRow = width / sparsity;

  // Input and outputs
  simt_aligned int data[samplesPerRow * height],
                   indices[samplesPerRow * height],
                   dataT[samplesPerRow * height],
                   indicesT[samplesPerRow * height],
                   vecIn[width*2], vecOut[height];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < width; i++)
    vecIn[i] = rand15(&seed) & 0xff;
  for (int r = 0; r < height; r++) {
    vecOut[r] = 0;
    int offset = rand15(&seed) & (2*sparsity - 1);
    int n = 0;
    while (n < samplesPerRow) {
      data[r*samplesPerRow + n] = rand15(&seed) & 0xff;
      indices[r*samplesPerRow + n] = offset;
      n++;
      offset += rand15(&seed) & (2*sparsity-1);
      if (offset >= width) break;
    }
    while (n < samplesPerRow) {
      data[r*samplesPerRow + n] = 0;
      indices[r*samplesPerRow + n] = 0;
      n++;
    }
  }

  // Get matrix in column-major order
  for (int r = 0; r < height; r++)
    for (int n = 0; n < samplesPerRow; n++) {
      dataT[n * height + r] = data[r * samplesPerRow + n];
      indicesT[n * height + r] = indices[r * samplesPerRow + n];
    }

  // Instantiate kernel
  SparseMatVecMul k;

  // One thread per row
  int groups = height / SIMTLanes;
  k.blockDim.x = SIMTLanes;
  k.gridDim.x = groups < SIMTWarps ? SIMTWarps : groups;

  // Assign parameters
  k.num_rows = height;
  k.num_cols = width;
  k.num_cols_per_row = samplesPerRow;
  k.indices = indicesT;
  k.data = dataT;
  k.x = vecIn;
  k.y = vecOut;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int r = 0; r < height; r++) {
    int sum = 0;
    for (int n = 0; n < samplesPerRow; n++) {
      int i = r*samplesPerRow + n;
      if (data[i] != 0) sum += data[i] * vecIn[indices[i]];
    }
    ok = ok && sum == vecOut[r];
  }

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
