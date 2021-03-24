#include <libNextDoor.hpp>
#include <main.cu>
#include "randomWalks.cu"

typedef PPRApp PPRSampling;

//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME PPRSamplingPy2
  #define INIT_FUNC_NAME initPPRSamplingPy2
  const char* moduleName = "PPRSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME PPRSamplingPy3
  #define INIT_FUNC_NAME PyInit_PPRSamplingPy3
  const char* moduleName = "PPRSamplingPy3";
#endif

//Declare NextDoorData
static NextDoorData<DummySample, PPRApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<DummySample, PPRApp>(argc, argv, checkSampledVerticesResult<DummySample, PPRSampling>);
}

//Include NextDoor Modules
#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif