#ifndef _HOSTLINK_H_
#define _HOSTLINK_H_

#include <HostLinkBase.h>

class HostLink : public HostLinkBase {

  public:

  // Assuming the boot loader is running on the CPU
  // ----------------------------------------------

  // Load application code and data onto the SoC
  void boot(const char* codeFilename, const char* dataFilename) {
    MemFileReader code(codeFilename);
    MemFileReader data(dataFilename);

    // Send code
    uint32_t addr, word;
    while (code.getWord(&addr, &word)) {
      uart->putWord(addr);
      uart->putWord(word);
    }
    uart->putWord(0xffffffff);  // Terminator

    // Send data
    while (data.getWord(&addr, &word)) {
      uart->putWord(addr);
      uart->putWord(word);
    }
    uart->putWord(0xffffffff);  // Terminator
    uart->flush();
  }

};

#endif
