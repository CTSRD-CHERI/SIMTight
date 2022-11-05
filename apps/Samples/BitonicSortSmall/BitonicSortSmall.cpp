// Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.

// Modified for NoCL, November 2022.

#include <NoCL.h>
#include <Rand.h>

// Size of arrays being sorted
#define LOCAL_SIZE_LIMIT 1024

// Sort two key/value pairs
inline void twoSort(unsigned *keyA, unsigned* valA,
                    unsigned *keyB, unsigned* valB, unsigned dir)
{
  if ((*keyA > *keyB) == dir) {
    unsigned t;
    t = *keyA; *keyA = *keyB; *keyB = t;
    t = *valA; *valA = *valB; *valB = t;
  }
  noclConverge();
}

// Monolithic bitonic sort kernel for short arrays fitting into local mem
struct BitonicSortLocal : Kernel {
  unsigned *d_DstKey_arg;
  unsigned *d_DstVal_arg;
  unsigned *d_SrcKey_arg;
  unsigned *d_SrcVal_arg;
  unsigned arrayLength;
  unsigned sortDir;

  void kernel() {
    unsigned* l_key = shared.array<unsigned, LOCAL_SIZE_LIMIT>();
    unsigned* l_val = shared.array<unsigned, LOCAL_SIZE_LIMIT>();

    // Offset to the beginning of subbatch and load data
    unsigned* d_SrcKey =
      d_SrcKey_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_SrcVal =
      d_SrcVal_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_DstKey =
      d_DstKey_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    unsigned* d_DstVal =
      d_DstVal_arg + blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
    l_key[threadIdx.x + 0] = d_SrcKey[0];
    l_val[threadIdx.x + 0] = d_SrcVal[0];
    l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] =
      d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
    l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] =
      d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

    for (unsigned size = 2; size < arrayLength; size <<= 1) {
      // Bitonic merge
      unsigned dir = ((threadIdx.x & (size / 2)) != 0);
      for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        unsigned pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        twoSort(
          &l_key[pos + 0], &l_val[pos + 0],
          &l_key[pos + stride], &l_val[pos + stride], dir);
      }
    }

    // dir == sortDir for the last bitonic merge step
    {
      for(unsigned stride = arrayLength / 2; stride > 0; stride >>= 1){
        __syncthreads();
        unsigned pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        twoSort(
          &l_key[pos + 0], &l_val[pos + 0],
          &l_key[pos + stride], &l_val[pos + stride], sortDir);
      }
    }

    __syncthreads();
    d_DstKey[0] = l_key[threadIdx.x + 0];
    d_DstVal[0] = l_val[threadIdx.x + 0];
    d_DstKey[(LOCAL_SIZE_LIMIT / 2)] =
      l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
    d_DstVal[(LOCAL_SIZE_LIMIT / 2)] =
      l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Array size and number of arrays for benchmarking
  int N = LOCAL_SIZE_LIMIT;
  int batch = isSim ? 4 : 32;

  // Input and output vectors
  simt_aligned unsigned srcKeys[N*batch], srcVals[N*batch];
  simt_aligned unsigned dstKeys[N*batch], dstVals[N*batch];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N*batch; i++) {
    srcKeys[i] = rand15(&seed);
    srcVals[i] = rand15(&seed);
  }

  // Instantiate kernel
  BitonicSortLocal k;

  // Use a single block of threads per array
  k.blockDim.x = LOCAL_SIZE_LIMIT / 2;
  k.gridDim.x = batch;

  // Assign parameters
  k.d_DstKey_arg = dstKeys;
  k.d_DstVal_arg = dstVals;
  k.d_SrcKey_arg = srcKeys;
  k.d_SrcVal_arg = srcVals;
  k.arrayLength = N;
  k.sortDir = 1;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int b = 0; b < batch; b++)
    for (int i = 0; i < N-1; i++)
      ok = ok && dstKeys[b*N+i] <= dstKeys[b*N+i+1];

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
