#include "libNextDoor.hpp"
#include "main.cu"

#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64

static NextDoorData<LayerSample, FastGCNApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  return appMain<LayerSample, FastGCNApp>(argc, argv, checkAdjacencyMatrixResult<LayerSample, FastGCNApp>);
}

#ifdef PYTHON_2
#include "nextDoorModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorModule.cu"
#endif