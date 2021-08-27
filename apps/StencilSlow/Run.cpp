#include <HostLink.h>

int main()
{
  HostLink hostLink;
  hostLink.boot("code.v", "data.v");
  hostLink.uart->putByte(IsSimulation);
  hostLink.dump();
  return 0;
}
