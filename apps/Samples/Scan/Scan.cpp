#include <NoCL.h>
#include <Rand.h>

// Kernel for computing the parallel prefix sum (inclusive scan)
// Simple (non-work-efficient) version based on one from "GPU Gems 3"
template <int BlockSize> struct Scan : Kernel {
  int len;
  int *in, *out;

  // Shared local memory
  int* tempIn;
  int* tempOut;

  void init() {
    declareShared(&tempIn, BlockSize);
    declareShared(&tempOut, BlockSize);
  }

  void kernel() {
    // Shorthand for local thread id
    int t = threadIdx.x;

    for (int x = 0; x < len; x += blockDim.x) {
      // Load data
      tempOut[t] = in[x+t];
      __syncthreads();

      // Local scan
      for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        swap(tempIn, tempOut);
        if (t >= offset)
          tempOut[t] = tempIn[t] + tempIn[t - offset];
        else
          tempOut[t] = tempIn[t];
        __syncthreads();
      }

      // Store data
      int acc = x > 0 ? out[x-1] : 0;
      out[x+t] = tempOut[t] + acc;
    }
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  // Should divide evenly by SIMT thread count
  int N = isSim ? 4096 : 1024000;

  // Input and output vectors
  simt_aligned int in[N], out[N];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    in[i] = rand15(&seed);
  }

  // Instantiate kernel
  Scan<SIMTWarps * SIMTLanes> k;

  // Use a single block of threads
  k.blockDim.x = SIMTWarps * SIMTLanes;

  // Assign parameters
  k.len = N;
  k.in = in;
  k.out = out;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  int acc = 0;
  for (int i = 0; i < N; i++) {
    acc += in[i];
    ok = ok && out[i] == acc;
  }

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
