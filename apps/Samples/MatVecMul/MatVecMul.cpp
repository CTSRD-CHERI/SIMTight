#include <NoCL.h>
#include <Rand.h>

// Kernel for matrix-vector multipliation
template <int BlockSize> struct MatVecMul : Kernel {
  int width, height;
  int *mat, *vecIn, *vecOut;
 
  // Partial dot products stored in shared local memory
  int* partial;

  INLINE void init() {
    declareShared(&partial, BlockSize);
  }
 
  INLINE void kernel() {
    for (int y = blockIdx.x; y < height; y += gridDim.x) {
      // Row processed by this block
      int* row = mat + y * width;

      // Compute partial dot products
      int sum = 0;
      for (int x = threadIdx.x; x < width; x += blockDim.x)
        sum += row[x] * vecIn[x];
      partial[threadIdx.x] = sum;
      __syncthreads();

      // Final local reduction
      for (int i = blockDim.x >> 1; i > 0; i >>= 1)  {
        if (threadIdx.x < i)
          partial[threadIdx.x] += partial[threadIdx.x + i];
        __syncthreads();
      }

      // Write dot product to global memory
      if (threadIdx.x == 0) vecOut[y] = partial[0];

      noclConverge();
    }
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector and matrix dimensions for benchmarking
  int width = isSim ? 128 : 1024;
  int height = isSim ? 64 : 1024;

  // Input and outputs
  simt_aligned int mat[height*width], vecIn[width], vecOut[height];

  // Initialise inputs
  uint32_t seed = 1;
  for (int j = 0; j < width; j++)
    vecIn[j] = rand15(&seed) & 0xff;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++)
      mat[i*width+j] = rand15(&seed) & 0xff;
  }

  // Instantiate kernel
  MatVecMul<SIMTLanes> k;

  // One block of threads per matrix row
  k.blockDim.x = SIMTLanes;
  k.gridDim.x = SIMTWarps;

  // Assign parameters
  k.width = width;
  k.height = height;
  k.mat = mat;
  k.vecIn = vecIn;
  k.vecOut = vecOut;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int i = 0; i < height; i++) {
    int sum = 0;
    for (int j = 0; j < width; j++)
      sum += mat[i*width+j] * vecIn[j];
    ok = ok && sum == vecOut[i];
  }

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
