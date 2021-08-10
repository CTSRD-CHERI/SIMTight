#include <stdio.h>
#include <string.h>
#include <HostLink.h>

int main(int argc, char** argv)
{
  HostLink hostLink;

  bool failed = false;
  for (int i = 1; i < argc; i++) {
    char codeFile[1024];
    char dataFile[1024];
    int n = strlen(argv[i]);
    if (argv[i][n-1] == 'S' && argv[i][n-2] == '.') argv[i][n-2] = '\0';
    printf("  %-24s", argv[i]); fflush(stdout);
    snprintf(codeFile, sizeof(codeFile), "%s.cpu.code.v", argv[i]);
    snprintf(dataFile, sizeof(dataFile), "%s.cpu.data.v", argv[i]);
    hostLink.boot(codeFile, dataFile);
    uint8_t result = hostLink.uart->getByte();
    if (result == 1)
      printf("ok\n");
    else {
      printf("FAILED(%d)\n", result >> 1);
      failed = true;
    }
  }

  return failed ? -1 : 0;
}
