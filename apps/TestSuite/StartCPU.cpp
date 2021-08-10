#include <Config.h>
#include <Pebbles/CSRs/UART.h>
#include <Pebbles/CSRs/SIMTHost.h>

extern "C" void _start();

// CPU test suite entry point
void _start()
{
  // Run assembley tests on CPU core
  asm volatile("j _test_start");
}
