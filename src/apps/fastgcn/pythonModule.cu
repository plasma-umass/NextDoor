#include <libNextDoor.hpp>
#include <main.cu>
#include "fastgcnSampling.cu"

static NextDoorData<FastGCNSample, FastGCNSamplingApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  return appMain<FastGCNSample, FastGCNSamplingApp>(argc, argv, checkAdjacencyMatrixResult<FastGCNSample, FastGCNSamplingApp>);
}

#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif