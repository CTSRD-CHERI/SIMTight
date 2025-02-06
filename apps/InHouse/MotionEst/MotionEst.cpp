#include <NoCL.h>
#include <Rand.h>

// Search radius in pixels
// Must be less than SIMTLanes
#define RADIUS 4

// Motion estimation for 4x4 pixel blocks
struct MotionEst : Kernel {
  // Input frames and dimensions
  unsigned* currentFrame;
  unsigned* prevFrame;
  int frameWidth, frameHeight;

  // Origin and dimensions of region being processed
  int regionOriginX, regionOriginY;
  int regionLogWidth, regionLogHeight;
  
  // Output SAD per motion vector per pixel block
  unsigned int* sads;

  // Shared local memory: current frame's region being processed
  Array2D<unsigned> current;

  // Shared local memory: preiovus frame's region being processed
  Array2D<unsigned> prev;

  INLINE void init() {
    int regionWidth = 1 << regionLogWidth;
    int regionHeight = 1 << regionLogHeight;
    declareShared(&current, regionHeight, regionWidth);
    declareShared(&prev, regionHeight + 2*RADIUS, regionWidth + 2*SIMTLanes);
  }

  INLINE void kernel() {
    // Region dimensions
    int regionWidth = 1 << regionLogWidth;
    int regionHeight = 1 << regionLogHeight;

      // Load current frame's region
    for (int y = 0; y < regionHeight; y++) {
      for (int x = threadIdx.x; x < regionWidth; x += blockDim.x) {
        int fy = regionOriginY + y;
        int fx = regionOriginX + x;
        current[y][x] = currentFrame[fy * frameWidth + fx];
      }
      noclConverge();
    }
    __syncthreads();

    // Load previous frame's region (and required surroundings)
    for (int y = 0; y < regionHeight + 2*RADIUS; y++) {
      for (int x = threadIdx.x; x < regionWidth + 2*SIMTLanes;
                                x += blockDim.x) {
        int fy = regionOriginY + (y-RADIUS);
        int fx = regionOriginX + (x-SIMTLanes);
        bool outside = (fy < 0) | (fy >= frameHeight) |
                       (fx < 0) | (fx >= frameWidth);
        noclPush();
        if (outside)
          prev[y][x] = 0;
        else
          prev[y][x] = prevFrame[fy * frameWidth + fx];
        noclPop();
      }
      noclConverge();
    }
    __syncthreads();

    // Compute all SADs
    int numBlocksX = regionWidth >> 2;
    int numBlocksY = regionHeight >> 2;
    int numBlocks = numBlocksX * numBlocksY;
    int outputsPerBlock = (2*RADIUS+1) * (2*RADIUS+1);
    int numOutputs = numBlocks * outputsPerBlock;
    for (int i = threadIdx.x; i < numOutputs; i += blockDim.x) {
      // Which block in current frame are we processing?
      int blockId = i / outputsPerBlock;
      // Which motion vector are we computing?
      int vecId = i - blockId * outputsPerBlock;

      // Origin of current block
      int blockIdX = blockId & (numBlocksX - 1);
      int blockIdY = blockId >> (regionLogWidth - 2);
      int currentX = blockIdX << 2;
      int currentY = blockIdY << 2;

      // Origin of previous block
      int vecIdY = vecId / (2*RADIUS+1);
      int vecIdX = vecId - vecIdY * (2*RADIUS+1);
      int prevX = currentX - RADIUS + SIMTLanes + vecIdX;
      int prevY = currentY + vecIdY;

      // Compute SAD for current motion vector
      unsigned sad = 0;
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
          int diff = current[currentY+y][currentX+x] -
                     prev[prevY+y][prevX+x];
          noclPush();
          if (diff < 0) diff = -diff;
          noclPop();
          sad += diff;
        }
      sads[i] = sad;
    }
  }
};

int main() {
  // Are we in a simulation?
  bool isSim = getchar();

  // Problem size
  int logWidth = isSim ? 3 : 6;
  int logHeight = isSim ? 3 : 6;
  int width = 1 << logWidth;
  int height = 1 << logHeight;

  // Number of SADs being computed
  // One per motion vector per block
  int numOutputs = (width/4)*(height/4)*(2*RADIUS+1)*(2*RADIUS+1);

  // Input frames and output SADs
  nocl_aligned unsigned currentFrame[width*height],
                        prevFrame[width*height],
                        sads[numOutputs];

  // Prepare inputs
  uint32_t seed = 1;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      currentFrame[y*width+x] = rand15(&seed) & 0xff;
      prevFrame[y*width+x] = rand15(&seed) & 0xff;
    }

  // Run the kernel
  MotionEst k;
  k.blockDim.x = SIMTLanes * SIMTWarps; 
  k.frameWidth = width;
  k.frameHeight = height;
  k.regionOriginX = 0;
  k.regionOriginY = 0;
  k.regionLogWidth = logWidth;
  k.regionLogHeight = logHeight;
  k.currentFrame = currentFrame;
  k.prevFrame = prevFrame;
  k.sads = sads;
  noclRunKernelAndDumpStats(&k);

  // Check result
  bool ok = true;
  int outCount = 0;
  for (int cy = 0; cy < height; cy += 4)
    for (int cx = 0; cx < width; cx += 4)
      for (int py = cy-RADIUS; py < cy+RADIUS+1; py++)
        for (int px = cx-RADIUS; px < cx+RADIUS+1; px++) {
          unsigned sad = 0;
          for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
              int diff = currentFrame[(cy + y) * width + cx + x];
              if (py + y >= 0 && py + y < height &&
                  px + x >= 0 && px + x < width)
                diff = diff - prevFrame[(py + y) * width + px + x];
              if (diff < 0) diff = -diff;
              sad += diff;
            }
          ok = ok && sads[outCount] == sad;
          outCount++;
        }
      
  puts("Self test: ");
  puts(ok ? "PASSED" : "FAILED");
  putchar('\n');

  return 0;
}
