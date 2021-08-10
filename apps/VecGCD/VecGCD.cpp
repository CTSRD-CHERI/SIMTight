#include <NoCL.h>
#include <Rand.h>

// Euclid's algorithm
int gcd(int x, int y) {
  noclPush();
    while (x != y) {
      noclPush();
        if (x > y)
          x = x-y;
        else
          y = y-x;
      noclPop();
    }
  noclPop();
  return x;
}

// Euclid's algorithm on vectors
struct VecGCD : Kernel {
  int len;
  int *a, *b, *result;

  void kernel() {
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      result[i] = gcd(a[i], b[i]);
  }
};

int main()
{
  // Are we in simulation?
  bool isSim = getchar();

  // Vector size for benchmarking
  int N = isSim ? 100 : 100000;

  // Input and output vectors
  simt_aligned int a[N], b[N], result[N];

  // Initialise inputs
  uint32_t seed = 100;
  for (int i = 0; i < N; i++) {
    a[i] = 1 + (rand15(&seed) & 0xff);
    b[i] = 1 + (rand15(&seed) & 0xff);
  }

  // Instantiate kernel
  VecGCD k;

  // Use single block of threads
  k.blockDim.x = SIMTLanes * SIMTWarps;

  // Assign parameters
  k.len = N;
  k.a = a;
  k.b = b;
  k.result = result;

  // Invoke kernel
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  for (int i = 0; i < N; i++)
    ok = ok && result[i] == gcd(a[i], b[i]);

  // Display result
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
