// Boot loader
#include <Config.h>
#include <Pebbles/CSRs/Sim.h>
#include <Pebbles/CSRs/UART.h>
#include <Pebbles/CSRs/InstrMem.h>
#include <Pebbles/CSRs/SIMTHost.h>
#include <Pebbles/Instrs/CacheMgmt.h>

#if EnableCHERI
#include <cheriintrin.h>
#endif

// Receive byte from UART (blocking)
INLINE uint32_t getByte()
{
  while (!pebblesUARTCanGet()) {}
  return pebblesUARTGet();
}

// Receive 32-bit word from UART (blocking)
INLINE uint32_t getWord()
{
  uint32_t w;
  w = getByte();
  w |= getByte() << 8;
  w |= getByte() << 16;
  w |= getByte() << 24;
  return w;
}

// Send byte over UART (blocking)
INLINE void putByte(uint32_t byte)
{
  while (!pebblesUARTCanPut()) {}
  pebblesUARTPut(byte);
}

// Write instruction to SIMT core
INLINE void writeInstrToSIMTCore(uint32_t addr, uint32_t instr)
{
  while (! pebblesSIMTCanPut()) {}
  pebblesSIMTWriteInstr(addr, instr);
}

// Boot loader
int main()
{
  #if EnableCHERI
    void* almighty = cheri_ddc_get();
  #endif

  // Receive code from host (blocking)
  while (1) {
    uint32_t addr = getWord();
    if (addr == 0xffffffff) break;
    uint32_t data = getWord();
    pebblesInstrMemWrite(addr, data);
    writeInstrToSIMTCore(addr, data);
  }

  // Receive data from host (blocking)
  while (1) {
    uint32_t addr = getWord();
    if (addr == 0xffffffff) break;
    uint32_t data = getWord();
    #if EnableCHERI
      volatile uint32_t* ptr = (uint32_t*) cheri_address_set(almighty, addr);
    #else
      volatile uint32_t* ptr = (uint32_t*) addr;
    #endif
    *ptr = data;
  }

  // Perform cache flush so data is visible globally, e.g. to SIMT core
  pebblesCacheFlushFull();

  // Call the application's start function
  #if EnableCHERI
    int (*appStart)() = cheri_address_set(almighty,
      MemBase + MaxBootImageBytes);
  #else
    int (*appStart)() = (int (*)()) (MemBase + MaxBootImageBytes);
  #endif
  appStart();

  // Restart boot loader
  #if EnableCHERI
    uint32_t* memBase = (uint32_t*) cheri_address_set(almighty, MemBase);
    asm volatile("cjr %0" : : "C"(memBase));
  #else
    asm volatile("jr %0" : : "r"(MemBase));
  #endif

  // Unreachable
  return 0;
}
