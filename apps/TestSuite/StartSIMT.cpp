#include <Config.h>
#include <Pebbles/CSRs/UART.h>
#include <Pebbles/CSRs/SIMTHost.h>

#if EnableCHERI
#include <cheriintrin.h>
#endif

extern "C" void _start();

// Send byte over UART (blocking)
INLINE void putByte(uint32_t byte)
{
  while (!pebblesUARTCanPut()) {}
  pebblesUARTPut(byte);
}

// SIMT test suite entry point
void _start()
{

  // Start kernel on SIMT core
  while (! pebblesSIMTCanPut()) {}
  #if EnableCHERI
    void* startAddr;
    asm volatile ("cllc %0, _test_start" : "=&C"(startAddr));
    pebblesSIMTStartKernel(cheri_address_get(startAddr));
  #else
    uint32_t startAddr;
    asm volatile ("la %0, _test_start" : "=r"(startAddr));
    pebblesSIMTStartKernel(startAddr);
  #endif

  // Wait for kernel response
  while (!pebblesSIMTCanGet()) {}
  int resp = pebblesSIMTGet();

  // Send kernel response to host
  putByte(resp);

  // Restart boot loader
  asm volatile("jr %0" : : "r"(MemBase));
}
