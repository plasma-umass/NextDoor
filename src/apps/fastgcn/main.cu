#include <libNextDoor.hpp>
#include <main.cu>
#include "fastgcnSampling.cu"

//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME FastGCNSamplingPy2
  #define INIT_FUNC_NAME initFastGCNSamplingPy2
  const char* moduleName = "FastGCNSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME FastGCNSamplingPy3
  #define INIT_FUNC_NAME PyInit_FastGCNSamplingPy3
  const char* moduleName = "FastGCNSamplingPy3";
#endif

//Declare NextDoorData
static NextDoorData<FastGCNSample, FastGCNSamplingApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<FastGCNSample, FastGCNSamplingApp>(argc, argv, checkAdjacencyMatrixResult<FastGCNSample, FastGCNSamplingApp>);
}

//Include NextDoor Modules
#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif