// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Modified for NoCL, April 2021.

#include <NoCL.h>
#include <Rand.h>

// Matrix multiplication C = A * B
// (wA is A's width and wB is B's width)
template <int BlockSize> struct MatMul : Kernel {
  int *A, *B, *C;
  int wA, wB;

  void kernel() {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    auto As = shared.array<int, BlockSize, BlockSize>();
  
    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    auto Bs = shared.array<int, BlockSize, BlockSize>();

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BlockSize * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BlockSize;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BlockSize * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BlockSize * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    int Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {

      // Load the matrices from device memory
      // to shared memory; each thread loads
      // one element of each matrix
      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];

      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two matrices together;
      // each thread computes one element
      // of the block sub-matrix
      for (int k = 0; k < BlockSize; ++k) {
        Csub += As[ty][k] * Bs[k][tx];
      }

      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BlockSize * by + BlockSize * bx;
    C[c + wB * ty + tx] = Csub;
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Matrix dimensions for benchmarking
  // (Must be a multiple of SIMTLanes)
  int size = isSim ? 32 : 256;

  // Input and outputs
  simt_aligned int matA[size*size], matB[size*size],
                   matC[size*size], matCheck[size*size];

  // Initialise matrices
  uint32_t seed = 1;
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++) {
      matA[i*size+j] = rand15(&seed) & 0xff;
      matB[i*size+j] = rand15(&seed) & 0xff;
      matCheck[i*size+j] = 0;
    }

  // Instantiate kernel
  MatMul<SIMTLanes> k;

  // One block of threads per matrix tile
  k.blockDim.x = SIMTLanes;
  k.blockDim.y = SIMTLanes;
  k.gridDim.x = size / SIMTLanes;
  k.gridDim.y = size / SIMTLanes;

  // Assign parameters
  k.wA = size;
  k.wB = size;
  k.A = matA;
  k.B = matB;
  k.C = matC;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        matCheck[i*size+j] += matA[i*size+k] * matB[k*size+j];
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      ok = ok && matCheck[i*size+j] == matC[i*size+j];

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
