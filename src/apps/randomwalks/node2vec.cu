#include <libNextDoor.hpp>
#include <main.cu>
#include "randomWalks.cu"

typedef Node2VecApp Node2VecSampling;

//Different Modules Names for different Python versions 
#ifdef PYTHON_2
  #define MODULE_NAME Node2VecSamplingPy2
  #define INIT_FUNC_NAME initNode2VecSamplingPy2
  const char* moduleName = "Node2VecSamplingPy2";
#endif

#ifdef PYTHON_3
  #define MODULE_NAME Node2VecSamplingPy3
  #define INIT_FUNC_NAME PyInit_Node2VecSamplingPy3
  const char* moduleName = "Node2VecSamplingPy3";
#endif

//Declare NextDoorData
static NextDoorData<Node2VecSample, Node2VecSampling> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<Node2VecSample, Node2VecSampling>(argc, argv, checkSampledVerticesResult<Node2VecSample, Node2VecSampling>);
}

//Include NextDoor Modules
#ifdef PYTHON_2
#include "nextDoorPythonModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorPythonModule.cu"
#endif