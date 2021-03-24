#include <libNextDoor.hpp>
#include <main.cu>
#include "randomWalks.cu"

typedef DeepWalkApp DeepWalkSampling;

//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME DeepWalkSamplingPy2
  #define INIT_FUNC_NAME initDeepWalkSamplingPy2
  const char* moduleName = "DeepWalkSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME DeepWalkSamplingPy3
  #define INIT_FUNC_NAME PyInit_DeepWalkSamplingPy3
  const char* moduleName = "DeepWalkSamplingPy3";
#endif

//Declare NextDoorData
static NextDoorData<DummySample, DeepWalkApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<DummySample, DeepWalkApp>(argc, argv, checkSampledVerticesResult<DummySample, DeepWalkSampling>);
}

//Include NextDoor Modules
#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif