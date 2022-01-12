#include <NoCL.h>
#include <Rand.h>

// Kernel for computing 256-bin histograms
struct Histogram : Kernel {
  int len;
  unsigned char* input;
  int* bins;

  void kernel() {
    // Store histogram bins in shared local memory
    int* histo = shared.array<int, 256>();

    // Initialise bins
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      histo[i] = 0;

    __syncthreads();

    // Update bins
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      atomicAdd(&histo[input[i]], 1);

    __syncthreads();

    // Write bins to global memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
      bins[i] = histo[i];
  }
};


int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int N = isSim ? 3000 : 1000000;

  // Input and output vectors
  nocl_aligned unsigned char input[N];
  nocl_aligned int bins[256];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++)
    input[i] = rand15(&seed) & 0xff;

  // Instantiate kernel
  Histogram k;

  // Use single block of threads
  k.blockDim.x = SIMTLanes * SIMTWarps;

  // Assign parameters
  k.len = N;
  k.input = input;
  k.bins = bins;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  int goldenBins[256];
  for (int i = 0; i < 256; i++) goldenBins[i] = 0;
  for (int i = 0; i < N; i++) goldenBins[input[i]]++;
  for (int i = 0; i < 256; i++)
    ok = ok && bins[i] == goldenBins[i];

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
