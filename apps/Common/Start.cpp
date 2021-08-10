#include <Config.h>
#include <Pebbles/CSRs/UART.h>

#if EnableCHERI
#include <cheriintrin.h>
#include <cheri_init_globals.h>
#endif

extern "C" void _start();
extern int main();

// Send byte over UART (blocking)
INLINE void putByte(uint32_t byte)
{
  while (!pebblesUARTCanPut()) {}
  pebblesUARTPut(byte);
}

void _start()
{
  // Initialise .captable
  #if EnableCHERI
    void* almighty = cheri_ddc_get();

    // TODO: Constrain capabilities
    cheri_init_globals_3(almighty, almighty, almighty);
  #endif

  // Invoke application
  main();

  // Send terminating null to host
  putByte('\0');
}
