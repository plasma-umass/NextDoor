#include <curand.h>
#include <curand_kernel.h>
#include <vector>


#include "csr.hpp"
#include "utils.hpp"
#include "rand_num_gen.cuh"

#ifndef __NEXTDOOR_HPP__
#define __NEXTDOOR_HPP__

template<class SampleType, typename App>
struct NextDoorData {
  CSR* csr;
  std::vector<SampleType> samples;
  std::vector<VertexID_t> hFinalSamples;
  std::vector<VertexID_t> initialContents;
  std::vector<VertexID_t> initialTransitToSampleValues;
  std::vector<int> devices;

  //Per Device Data.
  std::vector<SampleType*> dOutputSamples;
  std::vector<VertexID_t*> dSamplesToTransitMapKeys;
  std::vector<VertexID_t*> dSamplesToTransitMapValues;
  std::vector<VertexID_t*> dTransitToSampleMapKeys;
  std::vector<VertexID_t*> dTransitToSampleMapValues;
  std::vector<EdgePos_t*> dSampleInsertionPositions;
  std::vector<EdgePos_t*> dNeighborhoodSizes;
  std::vector<curandState*> dCurandStates;
  std::vector<size_t> maxThreadsPerKernel;
  std::vector<VertexID_t*> dFinalSamples;
  std::vector<VertexID_t*> dInitialSamples;
  int INVALID_VERTEX;
  int maxBits;
  GPUCSRPartition gpuCSRPartition;
};

CSR* loadGraph(Graph& graph, char* graph_file, char* graph_type, char* graph_format);
GPUCSRPartition transferCSRToGPU(CSR* csr);
template<class SampleType, typename App>
bool allocNextDoorDataOnGPU(CSR* csr, NextDoorData<SampleType, App>& data);
template<class SampleType, typename App>
bool doSampleParallelSampling(CSR* csr, GPUCSRPartition gpuCSRPartition, NextDoorData<SampleType, App>& nextDoorData);
template<class SampleType, typename App>
std::vector<VertexID_t>& getFinalSamples(NextDoorData<SampleType, App>& data);
int getFinalSampleSize();

#endif