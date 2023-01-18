#include <NoCL.h>
#include <Rand.h>
#include <FastZero.h>


int main()
{

  uint32_t N = 64 * 2;
  simt_aligned int a[N];

  // Initialise inputs
  uint32_t seed = 1;
  for (int i = 0; i < N; i++) {
    a[i] = rand15(&seed);
  }

  fastZero(a, N);
  // Display result
  puts("FINISHED\n");

  return 0;
}
