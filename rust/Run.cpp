#include <HostLink.h>

int main(int argc, char* argv[])
{
  if (argc == 2) {
    char codeFilename[1024];
    char dataFilename[1024];
    snprintf(codeFilename, sizeof(codeFilename), "%s-code.v", argv[1]);
    snprintf(dataFilename, sizeof(dataFilename), "%s-data.v", argv[1]);
    HostLink hostLink;
    hostLink.boot(codeFilename, dataFilename);
    hostLink.dump();
  }
  else {
    printf("Usage: Run [EXAMPLE]\n");
    return -1;
  }
  return 0;
}
