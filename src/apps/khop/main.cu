#include <libNextDoor.hpp>
#include <main.cu>
#include "khop.cu"

typedef KHopApp KHopSampling;

//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME KHopSamplingPy2
  #define INIT_FUNC_NAME initKHopSamplingPy2
  const char* moduleName = "KHopSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME KHopSamplingPy3
  #define INIT_FUNC_NAME PyInit_KHopSamplingPy3
  const char* moduleName = "KHopSamplingPy3";
#endif

//Declare NextDoorData
static NextDoorData<KHopSample, KHopSampling> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<KHopSample, KHopSampling>(argc, argv, checkSampledVerticesResult<KHopSample, KHopSampling>);
}

//Include NextDoor Modules
#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif