#include <libNextDoor.hpp>
#include <main.cu>
#include "ladiessampling.cu"
//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME LADIESSamplingPy2
  #define INIT_FUNC_NAME initLADIESSamplingPy2
  const char* moduleName = "LADIESSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME LADIESSamplingPy3
  #define INIT_FUNC_NAME PyInit_LADIESSamplingPy3
  const char* moduleName = "LADIESSamplingPy3";
#endif

static NextDoorData<ImportanceSample, ImportanceSampleApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  return appMain<ImportanceSample, ImportanceSampleApp>(argc, argv, checkAdjacencyMatrixResult<ImportanceSample, ImportanceSampleApp>);
}

#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif