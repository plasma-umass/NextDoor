#include "sample.hpp"
#include "csr.hpp"
#include "utils.hpp"
#include "sampler.cuh"
#include "rand_num_gen.cuh"

#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#ifndef __NEXTDOOR_HPP__
#define __NEXTDOOR_HPP__

template<class SampleType>
struct NextDoorData {
  std::vector<SampleType> samples;
  std::vector<VertexID_t> hFinalSamples;
  std::vector<VertexID_t> initialContents;
  SampleType* dOutputSamples;

  /*Outputs for Matrix*/
  std::vector<EdgePos_t> hFinalSamplesCSRRow;
  std::vector<EdgePos_t> hFinalSamplesCSRCol;
  std::vector<float> hFinalSamplesCSRVal;
  EdgePos_t* dFinalSamplesCSRRow;
  float* dFinalSamplesCSRVal;
  EdgePos_t* dFinalSamplesCSRCol;
  /******************/
  VertexID_t* dSamplesToTransitMapKeys;
  VertexID_t* dSamplesToTransitMapValues;
  VertexID_t* dTransitToSampleMapKeys;
  VertexID_t* dTransitToSampleMapValues;
  EdgePos_t* dSampleInsertionPositions;
  EdgePos_t* dNeighborhoodSizes;
  curandState* dCurandStates;
  size_t maxThreadsPerKernel;
  VertexID_t* dFinalSamples;
  VertexID_t* dInitialSamples;
  int INVALID_VERTEX;
  int maxBits;
  GPUCSRPartition gpuCSRPartition;
};

CSR* loadGraph(Graph& graph, char* graph_file, char* graph_type, char* graph_format);
GPUCSRPartition transferCSRToGPU(CSR* csr);
template<class SampleType>
bool allocNextDoorDataOnGPU(CSR* csr, NextDoorData<SampleType>& data);
template<class SampleType>
bool doSampling(CSR* csr, GPUCSRPartition gpuCSRPartition, NextDoorData<SampleType>& data, int nruns);
template<class SampleType>
std::vector<VertexID_t>& getFinalSamples(NextDoorData<SampleType>& data);

#endif