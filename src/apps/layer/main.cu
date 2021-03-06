#include <libNextDoor.hpp>
#include <main.cu>
#include "layerSampling.cu"

//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME LayerSamplingPy2
  #define INIT_FUNC_NAME initLayerSamplingPy2
  const char* moduleName = "LayerSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME LayerSamplingPy3
  #define INIT_FUNC_NAME PyInit_LayerSamplingPy3
  const char* moduleName = "LayerSamplingPy3";
#endif

//Declare NextDoorData
static NextDoorData<LayerSample, LayerSamplingApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<LayerSample, LayerSamplingApp>(argc, argv, nullptr);
}

//Include NextDoor Modules
#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif