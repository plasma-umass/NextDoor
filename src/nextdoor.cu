#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <unordered_set>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <string.h>
#include <assert.h>
#include <tuple>
#include <queue>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

#include "sample.hpp"

#ifndef __NEXTDOOR_CU__
#define __NEXTDOOR_CU__

typedef VertexID VertexID_t;

//TODO-List:
//[] Divide main() function in several small functions.
//[] Divide the code in several include files that can be included in the API.
//[] In GPU Kernels, do refactoring and move them to other places.
//[] Use vectors instead of dynamic arrays and new.
//[] Convert these vectors to a new array type that does not do initialization of data.
//[] Use MKL or cuSPARSE to do the matrix transpose or sorting
//[] A configuration that specifies all the parameters.
//[] Use Templates for cleaner code of Sampler

//Supported:
//citeseer.graph
// const int N = 3312;
// const int N_EDGES = 9074;
//micro.graph
//const int N = 100000;
//const int N_EDGES = 2160312;
//rmat.graph
// const int N = 1024;
// const int N_EDGES = 29381;
//ego-facebook
// const int N = 4039;
// const int N_EDGES = 88244;
//ego-twitter
//const int N = 81306;
//const int N_EDGES = 2420766;
//ego-gplus
//const int N = 107614;
//const int N_EDGES = 13652253;
//soc-pokec-relationships
//const int N = 1632803;
//const int N_EDGES = 30480021;
//soc-LiveJournal1
//const int N = 4847571;
//const int N_EDGES = 68556521;

//Not supportred:
//com-orkut.ungraph
// const int N = 3072441;
// const int N_EDGES = 117185083;

#include "csr.hpp"
#include "utils.hpp"
#include "sampler.cuh"
#include "rand_num_gen.cuh"
#include "libNextDoor.hpp"

using namespace utils;
using namespace GPUUtils;

#define CHECK_RESULT

//For mico, 512 works best
const size_t N_THREADS = 256;

//TODO try for larger random walks to improve results

#define WARP_HOP

const int ALL_NEIGHBORS = -1;

const bool useGridKernel = true;
const bool useSubWarpKernel = false;
const bool useThreadBlockKernel = false;
const bool combineTwoSampleStores = true;

enum TransitKernelTypes {
  GridKernel = 1,
  ThreadBlockKernel = 2,
  SubWarpKernel = 3,
  IdentityKernel = 4,
  NumKernelTypes = 4
};

/**User Defined Functions**/
enum SamplingType {
  IndividualNeighborhood,
  CollectiveNeighborhood
};

enum OutputFormat {
  SampledVertices,
  AdjacencyMatrix
};

/************Application Functions********** 
__host__ __device__ int stepSize(int k);

template<class SampleType> 
__device__ inline
VertexID next(int step, CSRPartition* csr, const VertexID* transit, 
              const VertexID sampleID, SampleType* sample,
              const float maxWeight,
              const CSR::Edge* transitEdges, const float* transitEdgeWeights,
              const EdgePos_t numEdges, const EdgePos_t neighbrID, 
              curandState* state);
template<class SampleType, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool DECREASE_GM_LOADS>
__device__ inline
VertexID nextCached(int step, const VertexID transit, 
              const VertexID sampleID, SampleType* sample,
              const float maxWeight,
              const CSR::Edge* transitEdges, const float* transitEdgeWeights,
              const EdgePos_t numEdges, const EdgePos_t neighbrID, 
              curandState* state, VertexID_t* cachedEdges, float* cachedWeights,
              bool* globalLoadBV);
__host__ __device__ int steps();
__host__ __device__ int samplingType();
__host__ __device__ bool hasExplicitTransits();
template<class SampleType>
__device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, const int transitIdx, curandState* randState);
template<class SampleType>
__host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID);
__host__ __device__ OutputFormat outputFormat();
__host__ __device__ EdgePos_t (CSR* graph);
__host__ __device__ EdgePos_t initialSampleSize(CSR* graph);
template<class SampleType>
__host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample);
*********************/

__constant__ char csrPartitionBuff[sizeof(CSRPartition)];

template<typename App>
__host__ __device__
EdgePos_t newNeighborsSize(int hop, EdgePos_t num_edges)
{
  return (App().stepSize(hop) == ALL_NEIGHBORS) ? num_edges : (EdgePos_t)App().stepSize(hop);
}

template<typename App>
__host__ __device__
EdgePos_t stepSizeAtStep(int step)
{
  if (step == -1)
    return 0;

  if (App().samplingType() == IndividualNeighborhood) {
    EdgePos_t n = App().initialSampleSize(nullptr);
    for (int i = 0; i <= step; i++) {
      n = n * App().stepSize(i);
    }

    return n;
  } else if (App().samplingType() == CollectiveNeighborhood) {
    EdgePos_t n = 0;
    for (int i = 0; i <= step; i++) {
      n += App().stepSize(i);
    }

    return n;
  }

  return -1;
}

template<typename App>
__host__ __device__ int numberOfTransits(int step) {
  if (App().samplingType() == CollectiveNeighborhood) {
    return App().stepSize(step);
  } else if (App().samplingType() == IndividualNeighborhood) {
    return stepSizeAtStep<App>(step);
  }
  assert(false);
  return -1;
}

template<class SamplingType, typename App>
__global__ void samplingKernel(const int step, GPUCSRPartition graph, const size_t threadsExecuted, const size_t currExecutionThreads,
                               const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, SamplingType* samples, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               curandState* randStates)
{
  EdgePos_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  //__shared__ VertexID newNeigbhors[N_THREADS];

  if (threadId >= currExecutionThreads)
    return;
  
  curandState* randState = &randStates[threadId];

  threadId += threadsExecuted;
  EdgePos_t transitIdx = threadId/App().stepSize(step);
  EdgePos_t transitNeighborIdx = threadId % App().stepSize(step);
  VertexID_t sampleIdx = transitToSamplesValues[transitIdx];
  assert(sampleIdx < NumSamples);
  VertexID_t transit = transitToSamplesKeys[transitIdx];
  VertexID_t neighbor = invalidVertex;
  graph.device_csr = (CSRPartition*)&csrPartitionBuff[0];

  if (transit != invalidVertex) {
    // if (graph.device_csr->has_vertex(transit) == false)
    //   printf("transit %d\n", transit);
    assert(graph.device_csr->has_vertex(transit));

    EdgePos_t numTransitEdges = graph.device_csr->get_n_edges_for_vertex(transit);
    
    if (numTransitEdges != 0) {
      const CSR::Edge* transitEdges = graph.device_csr->get_edges(transit);
      const float* transitEdgeWeights = graph.device_csr->get_weights(transit);
      const float maxWeight = graph.device_csr->get_max_weight(transit);

      neighbor = App().next(step, graph.device_csr, &transit, sampleIdx, &samples[sampleIdx], maxWeight, transitEdges, transitEdgeWeights, 
                      numTransitEdges, transitNeighborIdx, randState);
#if 0
      //search if neighbor has already been selected.
      //we can do that in register if required
      newNeigbhors[threadIdx.x] = neighbor;

      bool found = false;
      for (int i = 0; i < N_THREADS; i++) {
        if (newNeigbhors[i] == neighbor) {
          found = true;
          // break;
        }
      }

      __syncwarp();
      if (found) {
        neighbor = App().next(step, transit, sample, transitEdges, numTransitEdges, 
          transitNeighborIdx, randState);;
      }
#endif
    }
  }

  __syncwarp();

  EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);

  //TODO: templatize over hasExplicitTransits()
  if (step != App().steps() - 1) {
    //No need to store at last step
    if (App().hasExplicitTransits()) {
      VertexID_t transit = App().stepTransits(step, sampleIdx, samples[sampleIdx], transitIdx, randState);
      samplesToTransitValues[threadId] = transit;
    } else {
      samplesToTransitValues[threadId] = neighbor;
    }
    samplesToTransitKeys[threadId] = sampleIdx;
  }

  EdgePos_t insertionPos = 0;

  if (numberOfTransits<App>(step) > 1) {    
    //insertionPos = finalSampleSizeTillPreviousStep + transitNeighborIdx; //
    if (step == 0) {
      insertionPos = transitNeighborIdx;
    } else {
      EdgePos_t numTransits = numberOfTransits<App>(step);
      size_t finalSampleSizeTillPreviousStep = 0;
      size_t neighborsToSampleAtStep = 1;
      for (int _s = 0; _s < step; _s++) {
        neighborsToSampleAtStep *= App().stepSize(_s);
        finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
      }
      insertionPos = finalSampleSizeTillPreviousStep + utils::atomicAdd(&sampleInsertionPositions[sampleIdx], 1);
    }
  } else {
    insertionPos = step;
  }

  // if (insertionPo

  // if (insertionPos < finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
  // }
  assert(finalSampleSize > 0);
  if (insertionPos >= finalSampleSize) {
    printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sampleIdx);
  }
  assert(insertionPos < finalSampleSize);
  if (numberOfTransits<App>(step) == 1 and combineTwoSampleStores) {
    if (step % 2 == 1) {
      finalSamples[sampleIdx*finalSampleSize + insertionPos - 1] = transit;
      finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
    } else if (step == App().steps() - 1) {
      finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
    }
  }
  else {
    finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
  }

  // if (sample == 100) {
  //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
  // }
}

template<class SampleType, typename App, int THREADS, bool COALESCE_CURAND_LOAD>
__global__ void identityKernel(const int step, GPUCSRPartition graph, const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               curandState* randStates, const int* kernelTypeForTransit)
{
  __shared__ unsigned char shMemCuRand[sizeof(curandState)*THREADS];

  int threadId = threadIdx.x + blockDim.x * blockIdx.x;

  curandState* curandSrcPtr;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMemCuRand[0];

    int* randStatesAsInts = (int*)randStates;
  
    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = *curandSrcPtr;

  //__shared__ VertexID newNeigbhors[N_THREADS];
  EdgePos_t transitIdx;
  EdgePos_t transitNeighborIdx;
  VertexID_t transit;
  int kernelTy;

  if (threadId >= transitToSamplesSize)
    return;
  
  transitIdx = threadId/App().stepSize(step);
  transitNeighborIdx = threadId % App().stepSize(step);
  transit = transitToSamplesKeys[transitIdx];
  kernelTy = kernelTypeForTransit[transit];
  
  if ((useGridKernel && kernelTy == TransitKernelTypes::GridKernel) || 
      (useSubWarpKernel && kernelTy == TransitKernelTypes::SubWarpKernel) || 
      (useThreadBlockKernel && kernelTy == TransitKernelTypes::ThreadBlockKernel)) {
    return;
  }

  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];
  VertexID_t sampleIdx = transitToSamplesValues[transitIdx];
  assert(sampleIdx < NumSamples);
  VertexID_t neighbor = invalidVertex;

  if (transit != invalidVertex) {
    // if (graph.device_csr->has_vertex(transit) == false)
    //   printf("transit %d\n", transit);
    assert(csr->has_vertex(transit));

    EdgePos_t numTransitEdges = csr->get_n_edges_for_vertex(transit);
    
    if (numTransitEdges != 0) {
      const CSR::Edge* transitEdges = csr->get_edges(transit);
      const float* transitEdgeWeights = csr->get_weights(transit);
      const float maxWeight = csr->get_max_weight(transit);

      neighbor = App().next(step, csr, &transit, sampleIdx, &samples[sampleIdx], maxWeight, transitEdges, transitEdgeWeights, 
                      numTransitEdges, transitNeighborIdx, &localRandState);
#if 0
      //search if neighbor has already been selected.
      //we can do that in register if required
      newNeigbhors[threadIdx.x] = neighbor;

      bool found = false;
      for (int i = 0; i < N_THREADS; i++) {
        if (newNeigbhors[i] == neighbor) {
          found = true;
          // break;
        }
      }

      __syncwarp();
      if (found) {
        neighbor = next(step, transit, sample, transitEdges, numTransitEdges, 
          transitNeighborIdx, randState);;
      }
#endif
    }
  }

  __syncwarp();

//  EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);

  if (step != App().steps() - 1) {
    //No need to store at last step
    if (App().hasExplicitTransits()) {
      VertexID_t transit = App().stepTransits(step, sampleIdx, samples[sampleIdx], transitIdx, &localRandState);
      samplesToTransitValues[threadId] = transit;
    } else {
      samplesToTransitValues[threadId] = neighbor;
    }
    samplesToTransitKeys[threadId] = sampleIdx;
  }
  
  EdgePos_t insertionPos = 0; 
  if (false && numberOfTransits<App>(step) > 1) {    
    insertionPos = utils::atomicAdd(&sampleInsertionPositions[sampleIdx], 1);
  } else {
    insertionPos = step;
  }

  // if (insertionPos < finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
  // }
  assert(finalSampleSize > 0);
  // if (insertionPos >= finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sample);
  // }
  assert(insertionPos < finalSampleSize);

  if (combineTwoSampleStores) {
    if (step % 2 == 1) {
      int2 *ptr = (int2*)&finalSamples[sampleIdx*finalSampleSize + insertionPos - 1];
      int2 res;
      res.x = transit;
      res.y = neighbor;
      *ptr = res;
      //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
    } else if (step == App().steps() - 1) {
      finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
    }
  } else {
    finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
  }
  // if (sample == 100) {
  //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
  // }
  //TODO: We do not need atomic instead store indices of transit in another array,
  //wich can be accessed based on sample and transitIdx.
}

template<class SampleType, typename App, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool COALESCE_GL_LOADS, int TRANSITS_PER_THREAD, bool COALESCE_CURAND_LOAD>
__global__ void subWarpKernel(const int step, GPUCSRPartition graph, const VertexID_t invalidVertex,
                              const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                              const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                              VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                              VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                              curandState* randStates, const int* kernelTypeForTransit, const VertexID_t* subWarpKernelTBPositions, 
                              const EdgePos_t subWarpKernelTBPositionsNum)
{  
  // __shared__ unsigned char shMemAlloc[sizeof(curandState)*THREADS];
  // __shared__ EdgePos_t shSubWarpPositions[SUBWARPS_IN_TB*TRANSITS_PER_THREAD];
  const int SUBWARPS_IN_TB = THREADS/LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
  const int EDGE_CACHE_SIZE = (CACHE_EDGES ? CACHE_SIZE * sizeof(CSR::Edge) : 0);
  const int WEIGHT_CACHE_SIZE = (CACHE_WEIGHTS ? CACHE_SIZE * sizeof(float) : 0);
  const int TOTAL_CACHE_SIZE = MAX(WEIGHT_CACHE_SIZE + EDGE_CACHE_SIZE, 1); 
  const int CACHE_SIZE_PER_SUBWARP = CACHE_SIZE/SUBWARPS_IN_TB;

  union unionShMem {
    struct {
      EdgePos_t shSubWarpPositions[SUBWARPS_IN_TB*TRANSITS_PER_THREAD];
      unsigned char edgeAndWeightCache[TOTAL_CACHE_SIZE];
    };
    unsigned char shMemAlloc[sizeof(curandState)*THREADS];
  };
  __shared__ unionShMem shMem;
  
  const int threadId = threadIdx.x + blockDim.x * blockIdx.x;

  const int subWarpThreadIdx = threadId % LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
  const int subWarp = threadId / LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
  const int subWarpIdxInTB = threadIdx.x/LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
  const int startSubWarpIdxInTB = (blockIdx.x*blockDim.x)/LoadBalancing::LoadBalancingThreshold::SubWarpLevel;

  EdgePos_t* edgesInShMem = (EdgePos_t*) (CACHE_EDGES ? &shMem.edgeAndWeightCache[CACHE_SIZE_PER_SUBWARP*subWarpIdxInTB] : nullptr);
  float* edgeWeightsInShMem = (float*) (CACHE_WEIGHTS ? (&shMem.edgeAndWeightCache[EDGE_CACHE_SIZE + CACHE_SIZE_PER_SUBWARP*subWarpIdxInTB]): nullptr);
  bool* globalLoadBV = nullptr;

  curandState* curandSrcPtr;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMem.shMemAlloc[0];

    int* randStatesAsInts = (int*)randStates;
  
    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = *curandSrcPtr;
  
  for (int _subWarpIdx = threadIdx.x; _subWarpIdx < SUBWARPS_IN_TB * TRANSITS_PER_THREAD; _subWarpIdx += blockDim.x) {
    if (_subWarpIdx + startSubWarpIdxInTB * TRANSITS_PER_THREAD >= subWarpKernelTBPositionsNum) {
      continue;
    }
    shMem.shSubWarpPositions[_subWarpIdx] = subWarpKernelTBPositions[_subWarpIdx + startSubWarpIdxInTB * TRANSITS_PER_THREAD];
  }

  __syncthreads();
  bool invalidateCache;
  VertexID_t currTransit = invalidVertex;

  invalidateCache = true;
  EdgePos_t numTransitEdges;
  CSR::Edge* glTransitEdges;
  float* glTransitEdgeWeights;
  float maxWeight;

  for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++) {
    EdgePos_t subWarpIdx = TRANSITS_PER_THREAD * subWarp + transitI;
    if (subWarpIdx >= subWarpKernelTBPositionsNum) {
      continue;
    }

    EdgePos_t transitStartPos = shMem.shSubWarpPositions[subWarpIdxInTB * TRANSITS_PER_THREAD + transitI];
    EdgePos_t transitIdx = transitStartPos + subWarpThreadIdx;
    EdgePos_t transitNeighborIdx = 0;
    VertexID_t transit = transitIdx < NumSamples ? transitToSamplesKeys[transitIdx] : -1;
    // if ((uint64_t)(transitToSamplesKeys + transitIdx) % 32 != 0) {
    //   printf("unaligned %p %p %d %d\n", transitToSamplesKeys + transitIdx, transitToSamplesKeys, transitIdx, transitStartPos);
    // }    
    VertexID_t firstThreadTransit = __shfl_sync(FULL_WARP_MASK, transit, 0, LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
    __syncwarp();

    invalidateCache = currTransit != firstThreadTransit;
    currTransit = firstThreadTransit;

    CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];
    
    int tmpReadVertexData;

    if (invalidateCache) {
      const CSR::Vertex* transitVertex = csr->get_vertex(currTransit);
      if (subWarpThreadIdx < sizeof(CSR::Vertex)/sizeof(int)) {
        tmpReadVertexData = ((const int*)transitVertex)[subWarpThreadIdx];
      }
    }
    
    __syncwarp();

    const EdgePos_t startEdgeIdx = __shfl_sync(FULL_WARP_MASK, tmpReadVertexData, 1, LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
    const EdgePos_t endEdgeIdx = __shfl_sync(FULL_WARP_MASK, tmpReadVertexData, 2, LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
    
    if (invalidateCache) {
      int maxWeightBuff = __shfl_sync(FULL_WARP_MASK, tmpReadVertexData, 3, LoadBalancing::LoadBalancingThreshold::SubWarpLevel);      
      maxWeight = *((float*)&maxWeightBuff);
      numTransitEdges = (endEdgeIdx != -1) ? (endEdgeIdx - startEdgeIdx + 1) : 0; 
     
      glTransitEdges = (CSR::Edge*)((startEdgeIdx != -1) ? csr->get_edges() + startEdgeIdx : nullptr);
      glTransitEdgeWeights = (float*)((startEdgeIdx != -1) ? csr->get_weights() + startEdgeIdx : nullptr);
    }

    if (false) {
      //shMem.edgeAndWeightCache[threadIdx.x%32] = numTransitEdges + (int32_t)maxWeight + (int32_t)glTransitEdges + (int32_t)glTransitEdgeWeights;
      continue;
    }

    if (CACHE_EDGES && invalidateCache) {
      for (int e = subWarpThreadIdx; e < min((EdgePos_t)CACHE_SIZE_PER_SUBWARP, numTransitEdges); 
           e += LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
        edgesInShMem[e] = -1;
      }
    }

    if (CACHE_WEIGHTS && invalidateCache) {
      for (int e = subWarpThreadIdx; e < min((EdgePos_t)CACHE_SIZE_PER_SUBWARP, numTransitEdges); 
           e += LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
        edgeWeightsInShMem[e] = -1;
      }
    }

    __syncwarp();

    if (firstThreadTransit != transit)
      continue;

    // int kernelTy = kernelTypeForTransit[transit];
    // if (kernelTy != TransitKernelTypes::SubWarpKernel) {
    //   printf("threadId %d transitIdx %d kernelTy %d\n", threadId, transitIdx, kernelTy);
    // }
    //assert(kernelTypeForTransit[firstThreadTransit] == TransitKernelTypes::SubWarpKernel);
    VertexID_t sampleIdx = transitToSamplesValues[transitIdx];
    assert(sampleIdx < NumSamples);
    VertexID_t neighbor = invalidVertex;

    // if (graph.device_csr->has_vertex(transit) == false)
    //   printf("transit %d\n", transit);
    assert(csr->has_vertex(transit));
    
    if (numTransitEdges != 0) {
      neighbor = App().template nextCached<SampleType, CACHE_SIZE_PER_SUBWARP, CACHE_EDGES, CACHE_WEIGHTS, COALESCE_GL_LOADS>(step, transit, sampleIdx, &samples[sampleIdx], maxWeight, 
                                                                                    glTransitEdges, glTransitEdgeWeights, 
                                                                                    numTransitEdges, transitNeighborIdx, &localRandState,
                                                                                    edgesInShMem, edgeWeightsInShMem,
                                                                                    globalLoadBV);
    }

    // __syncwarp();

    //EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);

    if (step != App().steps() - 1) {
      //No need to store at last step
      samplesToTransitKeys[transitIdx] = sampleIdx;
      if (App().hasExplicitTransits()) {
        VertexID_t transit = App().stepTransits(step, sampleIdx, samples[sampleIdx], transitIdx, &localRandState);
        samplesToTransitValues[threadId] = transit;
      } else {
        samplesToTransitValues[threadId] = neighbor;
      }
    }
    
    EdgePos_t insertionPos = 0; 
    if (false && numberOfTransits<App>(step) > 1) {    
      insertionPos = utils::atomicAdd(&sampleInsertionPositions[sampleIdx], 1);
    } else {
      insertionPos = step;
    }

    // if (insertionPos < finalSampleSize) {
    //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
    // }
    assert(finalSampleSize > 0);
    assert(insertionPos < finalSampleSize);
    if (combineTwoSampleStores) {
      if (step % 2 == 1) {
        int2 *ptr = (int2*)&finalSamples[sampleIdx*finalSampleSize + insertionPos - 1];
        int2 res;
        res.x = transit;
        res.y = neighbor;
        *ptr = res;
        //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
      } else if (step == App().steps() - 1) {
        finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
      }
    } else {
      finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
    }
    // if (sample == 100) {
    //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
    // }
    //TODO: We do not need atomic instead store indices of transit in another array,
    //wich can be accessed based on sample and transitIdx.
  }
}

template<int CACHE_SIZE, bool COALESCE_GL_LOADS, typename T>
__device__ inline VertexID_t cacheAndGet(EdgePos_t id, const T* transitEdges, T* cachedEdges, bool* globalLoadBV)
{
  VertexID_t e;
  if (id >= CACHE_SIZE) {
    return transitEdges[id];
  }
  
  if (COALESCE_GL_LOADS) {
    e = cachedEdges[id];
    if (e == -1)
      cachedEdges[id] = -2;

    int subWarpThreadIdx = threadIdx.x % LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
    //int subWarp = threadIdx.x / LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
    for (int i = subWarpThreadIdx; i < CACHE_SIZE; i += LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
      if (cachedEdges[i] == -2) {
        cachedEdges[i] = transitEdges[i];
      }
    }
    
    e = cachedEdges[id];
  } else {
    e = cachedEdges[id];

    if (e == -1) {
      e = transitEdges[id];
      cachedEdges[id] = e;
    }
  }

  return e;
}

template<class SampleType, typename App, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool COALESCE_GL_LOADS, int TRANSITS_PER_THREAD, bool COALESCE_CURAND_LOAD>
__global__ void threadBlockKernel(const int step, GPUCSRPartition graph, const VertexID_t invalidVertex,
                                  const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                                  const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                                  VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                                  VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                                  curandState* randStates, const int* kernelTypeForTransit, 
                                  const VertexID_t* threadBlockKernelTBPositions, 
                                  const EdgePos_t threadBlockKernelTBPositionsNum)
{
  //TODO: This works with thread block size of 32 only and NEEDS to be optimized.
  #define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE*sizeof(CSR::Edge) : 0)
  #define WEIGHT_CACHE_SIZE (CACHE_WEIGHTS ? CACHE_SIZE*sizeof(float) : 0)
  #define CURAND_SHMEM_SIZE (sizeof(curandState)*THREADS)
  // #define COALESCE_GL_LOADS_SHMEM_SIZE ()

  __shared__ unsigned char shMemAlloc[MAX(EDGE_CACHE_SIZE+WEIGHT_CACHE_SIZE, CURAND_SHMEM_SIZE)];
  
  //__shared__ bool globalLoadBV[COALESCE_GL_LOADS ? CACHE_SIZE : 1];
  bool* globalLoadBV;
  __shared__ VertexID_t numEdgesInShMem;
  __shared__ bool invalidateCache;
  __shared__ VertexID_t transitForTB;
  __shared__ CSR::Edge* glTransitEdges;
  __shared__ float* glTransitEdgeWeights;
  __shared__ float maxWeight;
  __shared__ EdgePos_t mapStartPos;

  CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)&shMemAlloc[0] : nullptr;
  float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMemAlloc[EDGE_CACHE_SIZE] : nullptr;
  
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  
  curandState* curandSrcPtr;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMemAlloc[0];

    int* randStatesAsInts = (int*)randStates;
  
    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = *curandSrcPtr;

  //__shared__ VertexID newNeigbhors[N_THREADS];
  //if (threadIdx.x == 0) printf("blockIdx.x %d\n", blockIdx.x);
  //shRandStates[threadIdx.x] = randStates[threadId];  
  //__syncthreads();
  
  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];

  for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++) {
    EdgePos_t transitIdx = 0;
    EdgePos_t transitNeighborIdx = 0;//threadId % stepSize(step); //TODO: Correct this for k-hop
    if (TRANSITS_PER_THREAD * blockIdx.x + transitI >= threadBlockKernelTBPositionsNum) {
      continue;
    }
    if (threadIdx.x == 0) {
      mapStartPos = threadBlockKernelTBPositions[TRANSITS_PER_THREAD * blockIdx.x + transitI];
    }
    __syncthreads();
    transitIdx = mapStartPos + threadIdx.x; //threadId/stepSize(step);
    VertexID_t transit = transitToSamplesKeys[transitIdx];

    if (threadIdx.x == 0) {
      invalidateCache = transitForTB != transit || transitI == 0;
      transitForTB = transit;
    }
    if (threadIdx.x == 0 && invalidateCache) {
      //assert(graph.device_csr->has_vertex(transit));
      //TODO: fuse below functions into one to decrease reads
      numEdgesInShMem = csr->get_n_edges_for_vertex(transit);
      glTransitEdges = (CSR::Edge*)csr->get_edges(transit);
      glTransitEdgeWeights = (float*)csr->get_weights(transit);
      maxWeight = csr->get_max_weight(transit);
    }

    __syncthreads();

    if (CACHE_EDGES && invalidateCache) {
      for (int i = threadIdx.x; i < min(CACHE_SIZE, numEdgesInShMem); i += blockDim.x) {
        edgesInShMem[i] = -1;//glTransitEdges[i];
      }
    }
  
    if (CACHE_WEIGHTS && invalidateCache) {
      for (int i = threadIdx.x; i < min(CACHE_SIZE, numEdgesInShMem); i += blockDim.x) {
        edgeWeightsInShMem[i] = -1;//glTransitEdgeWeights[i];
      }
    }

    __syncthreads();

    if (transit == transitForTB) {
      // if (threadIdx.x == 0 && kernelTypeForTransit[transit] != TransitKernelTypes::GridKernel) {
      //   printf("transit %d transitIdx %d gridDim.x %d\n", transit, transitIdx, gridDim.x);
      // }
      // assert (kernelTypeForTransit[transit] == TransitKernelTypes::GridKernel);

      VertexID_t sampleIdx = transitToSamplesValues[transitIdx];

      assert(sampleIdx < NumSamples);
      VertexID_t neighbor = invalidVertex;
      // if (graph.device_csr->has_vertex(transit) == false)
      //   printf("transit %d\n", transit);
      neighbor = App().template nextCached<SampleType, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, 0>(step, transit, sampleIdx, &samples[sampleIdx], maxWeight, 
                                                              glTransitEdges, glTransitEdgeWeights, 
                                                              numEdgesInShMem, transitNeighborIdx, &localRandState,
                                                              edgesInShMem, edgeWeightsInShMem,
                                                              &globalLoadBV[0]);
      __syncwarp();

      //EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);

      if (step != App().steps() - 1) {
        //No need to store at last step
        samplesToTransitKeys[transitIdx] = sampleIdx; //TODO: Update this for khop to transitIdx + transitNeighborIdx
        if (App().hasExplicitTransits()) {
          VertexID_t transit = App().stepTransits(step, sampleIdx, samples[sampleIdx], transitIdx, &localRandState);
          samplesToTransitValues[transitIdx] = transit;
        } else {
          samplesToTransitValues[transitIdx] = neighbor;
        }
      }
      
      EdgePos_t insertionPos = 0; 
      if (false && numberOfTransits<App>(step) > 1) {
        //insertionPos = utils::atomicAdd(&sampleInsertionPositions[sample], 1);
      } else {
        insertionPos = step;
      }

      // if (insertionPos < finalSampleSize) {
      //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
      // }
      assert(finalSampleSize > 0);
      if (insertionPos >= finalSampleSize) {
        printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sampleIdx);
      }
      assert(insertionPos < finalSampleSize);

      if (combineTwoSampleStores) {
        if (step % 2 == 1) {
          int2 *ptr = (int2*)&finalSamples[sampleIdx*finalSampleSize + insertionPos - 1];
          int2 res;
          res.x = transit;
          res.y = neighbor;
          *ptr = res;
          //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
        } else if (step == App().steps() - 1) {
          finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
        }
      } else {
        finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
      }
      // if (sample == 100) {
      //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
      // }
      //TODO: We do not need atomic instead store indices of transit in another array,
      //wich can be accessed based on sample and transitIdx.
    }
  }
}

template<class SampleType, typename App, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool COALESCE_GL_LOADS, int TRANSITS_PER_THREAD, bool COALESCE_CURAND_LOAD>
__global__ void gridKernel(const int step, GPUCSRPartition graph, const VertexID_t invalidVertex,
                           const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                           const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                           VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                           VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                           curandState* randStates, const int* kernelTypeForTransit, const VertexID_t* gridKernelTBPositions, 
                           const EdgePos_t gridKernelTBPositionsNum)
{
  #define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE*sizeof(CSR::Edge) : 0)
  #define WEIGHT_CACHE_SIZE (CACHE_WEIGHTS ? CACHE_SIZE*sizeof(float) : 0)
  #define CURAND_SHMEM_SIZE (sizeof(curandState)*THREADS)
  // #define COALESCE_GL_LOADS_SHMEM_SIZE ()

  __shared__ unsigned char shMemAlloc[MAX(EDGE_CACHE_SIZE+WEIGHT_CACHE_SIZE, CURAND_SHMEM_SIZE)];
  
  //__shared__ bool globalLoadBV[COALESCE_GL_LOADS ? CACHE_SIZE : 1];
  bool* globalLoadBV;
  __shared__ VertexID_t numEdgesInShMem;
  __shared__ bool invalidateCache;
  __shared__ VertexID_t transitForTB;
  __shared__ CSR::Edge* glTransitEdges;
  __shared__ float* glTransitEdgeWeights;
  __shared__ float maxWeight;
  __shared__ EdgePos_t mapStartPos;

  CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)&shMemAlloc[0] : nullptr;
  float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMemAlloc[EDGE_CACHE_SIZE] : nullptr;
  
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  
  curandState* curandSrcPtr;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMemAlloc[0];

    int* randStatesAsInts = (int*)randStates;
  
    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = *curandSrcPtr;

  //__shared__ VertexID newNeigbhors[N_THREADS];
  //if (threadIdx.x == 0) printf("blockIdx.x %d\n", blockIdx.x);
  //shRandStates[threadIdx.x] = randStates[threadId];  
  //__syncthreads();
  
  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];

  for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++) {
    EdgePos_t transitIdx = 0;
    EdgePos_t transitNeighborIdx = 0;//threadId % stepSize(step); //TODO: Correct this for k-hop
    if (TRANSITS_PER_THREAD * blockIdx.x + transitI >= gridKernelTBPositionsNum) {
      continue;
    }
    if (threadIdx.x == 0) {
      mapStartPos = gridKernelTBPositions[TRANSITS_PER_THREAD * blockIdx.x + transitI];
    }
    __syncthreads();
    transitIdx = mapStartPos + threadIdx.x; //threadId/stepSize(step);
    VertexID_t transit = transitToSamplesKeys[transitIdx];

    if (threadIdx.x == 0) {
      invalidateCache = transitForTB != transit || transitI == 0;
      transitForTB = transit;
    }
    if (threadIdx.x == 0 && invalidateCache) {
      //assert(graph.device_csr->has_vertex(transit));
      //TODO: fuse below functions into one to decrease reads
      numEdgesInShMem = csr->get_n_edges_for_vertex(transit);
      glTransitEdges = (CSR::Edge*)csr->get_edges(transit);
      glTransitEdgeWeights = (float*)csr->get_weights(transit);
      maxWeight = csr->get_max_weight(transit);
    }

    __syncthreads();

    if (CACHE_EDGES && invalidateCache) {
      for (int i = threadIdx.x; i < min(CACHE_SIZE, numEdgesInShMem); i += blockDim.x) {
        edgesInShMem[i] = -1;//glTransitEdges[i];
      }
    }
  
    if (CACHE_WEIGHTS && invalidateCache) {
      for (int i = threadIdx.x; i < min(CACHE_SIZE, numEdgesInShMem); i += blockDim.x) {
        edgeWeightsInShMem[i] = -1;//glTransitEdgeWeights[i];
      }
    }

    __syncthreads();

    if (transit == transitForTB) {
      // if (threadIdx.x == 0 && kernelTypeForTransit[transit] != TransitKernelTypes::GridKernel) {
      //   printf("transit %d transitIdx %d gridDim.x %d\n", transit, transitIdx, gridDim.x);
      // }
      // assert (kernelTypeForTransit[transit] == TransitKernelTypes::GridKernel);

      VertexID_t sampleIdx = transitToSamplesValues[transitIdx];

      assert(sampleIdx < NumSamples);
      VertexID_t neighbor = invalidVertex;
      // if (graph.device_csr->has_vertex(transit) == false)
      //   printf("transit %d\n", transit);
      neighbor = App().template nextCached<SampleType, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, 0>(step, transit, sampleIdx, &samples[sampleIdx], maxWeight, 
                                                              glTransitEdges, glTransitEdgeWeights, 
                                                              numEdgesInShMem, transitNeighborIdx, &localRandState,
                                                              edgesInShMem, edgeWeightsInShMem,
                                                              &globalLoadBV[0]);
      __syncwarp();

      //EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);

      if (step != App().steps() - 1) {
        //No need to store at last step
        samplesToTransitKeys[transitIdx] = sampleIdx; //TODO: Update this for khop to transitIdx + transitNeighborIdx
        if (App().hasExplicitTransits()) {
          VertexID_t transit = App().stepTransits(step, sampleIdx, samples[sampleIdx], transitIdx, &localRandState);
          samplesToTransitValues[transitIdx] = transit;
        } else {
          samplesToTransitValues[transitIdx] = neighbor;
        }
      }
      
      EdgePos_t insertionPos = 0; 
      if (false && numberOfTransits<App>(step) > 1) {
        //insertionPos = utils::atomicAdd(&sampleInsertionPositions[sample], 1);
      } else {
        insertionPos = step;
      }

      // if (insertionPos < finalSampleSize) {
      //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
      // }
      assert(finalSampleSize > 0);
      if (insertionPos >= finalSampleSize) {
        printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sampleIdx);
      }
      assert(insertionPos < finalSampleSize);

      if (combineTwoSampleStores) {
        if (step % 2 == 1) {
          int2 *ptr = (int2*)&finalSamples[sampleIdx*finalSampleSize + insertionPos - 1];
          int2 res;
          res.x = transit;
          res.y = neighbor;
          *ptr = res;
          //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
        } else if (step == App().steps() - 1) {
          finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
        }
      } else {
        finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
      }
      // if (sample == 100) {
      //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
      // }
      //TODO: We do not need atomic instead store indices of transit in another array,
      //wich can be accessed based on sample and transitIdx.
    }
  }
}

template<typename App>
__global__ void collectiveNeighbrsSize(const int step, GPUCSRPartition graph, 
                                       const VertexID_t invalidVertex,
                                       VertexID_t* initialSamples, 
                                       VertexID_t* finalSamples, 
                                       const size_t finalSampleSize, 
                                       EdgePos_t* sampleNeighborhoodPos,
                                       EdgePos_t* sumNeighborhoodSizes)
{
  //Assign one thread block to a sample
  __shared__ EdgePos_t neighborhoodSize;

  if (threadIdx.x == 0) {
    neighborhoodSize = 0;
  }

  __syncthreads();

  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];  
  VertexID_t sampleIdx = blockIdx.x;
  EdgePos_t numTransits = App().initialSampleSize(nullptr);
  //EdgePos_t numTransitsInPrevStep = numberOfTransits(step - 1);

  //TODO: Assuming step is 0
  for (int transitIdx = threadIdx.x; transitIdx < numTransits; transitIdx += blockDim.x) {
    VertexID_t transit = initialSamples[sampleIdx*App().initialSampleSize(nullptr) + transitIdx];
    if (transit != invalidVertex) {
      ::atomicAdd(&neighborhoodSize, csr->get_n_edges_for_vertex(transit)); 
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    sampleNeighborhoodPos[sampleIdx] = ::atomicAdd(sumNeighborhoodSizes, neighborhoodSize);
  }
}

template<typename App>
__global__ void collectiveNeighborhood(const int step, GPUCSRPartition graph, 
                                       const VertexID_t invalidVertex,
                                       VertexID_t* initialSamples, 
                                       VertexID_t* finalSamples, 
                                       const size_t finalSampleSize, 
                                       EdgePos_t* sampleNeighborhoodCSRRows,
                                       VertexID_t* sampleNeighborhoodCSRCols,
                                       EdgePos_t* sampleNeighborhoodPos,
                                       EdgePos_t* sumNeighborhoodSizes)
{
  //Assign one thread block to a sample
  EdgePos_t insertionPos = 0;
  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];  
  VertexID_t sampleIdx = blockIdx.x;
  EdgePos_t numTransits = App().initialSampleSize(nullptr);
  //EdgePos_t numTransitsInPrevStep = numberOfTransits(step - 1);

  //TODO: Assuming step is 0
  //Copy edges from graph, vertex by vertex
  for (int transitIdx = 0; transitIdx < numTransits; transitIdx++) {
    VertexID_t transit = initialSamples[sampleIdx*App().initialSampleSize(nullptr) + transitIdx];
    EdgePos_t nEdges = csr->get_n_edges_for_vertex(transit);
    const CSR::Edge* edges = csr->get_edges(transit);
    
    sampleNeighborhoodCSRRows[sampleIdx*App().initialSampleSize(nullptr) + transitIdx] = insertionPos;

    for (int e = threadIdx.x; e < nEdges; e += blockDim.x) {
      EdgePos_t pos = sampleNeighborhoodPos[sampleIdx] + insertionPos + e;
      sampleNeighborhoodCSRCols[pos] = edges[e];
    }

    insertionPos += nEdges;
    __syncthreads();
  }
}

template<class SampleType, typename App, bool StoreAsMap>
__global__ void explicitTransitsKernel(const int step, GPUCSRPartition graph, 
                                     const VertexID_t invalidVertex,
                                     const size_t threadsExecuted, 
                                     const size_t currExecutionThreads,
                                     const size_t totalThreads,
                                     SampleType* samples,
                                     const size_t NumSamples,
                                     VertexID_t* samplesToTransitKeys,
                                     VertexID_t* samplesToTransitValues,
                                     curandState* randStates)
{
  //Number of threads executed are: Num of Samples * Number of Transits
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  //__shared__ VertexID newNeigbhors[N_THREADS];
  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];
  if (threadId >= currExecutionThreads)
    return;
  
  curandState* randState = &randStates[threadId];
  threadId += threadsExecuted;
  EdgePos_t numTransits = numberOfTransits<App>(step - 1);
  EdgePos_t sampleIdx = threadId/numTransits;
  EdgePos_t transitIdx = threadId % numTransits;
  if (App().samplingType() == CollectiveNeighborhood) {
    assert(!App().hasExplicitTransits());
  } else {
    VertexID_t transit = App().stepTransits(step, sampleIdx, samples[sampleIdx], transitIdx, randState);
    samplesToTransitValues[threadId] = transit;

    if (StoreAsMap) {
      samplesToTransitKeys[threadId] = sampleIdx;
    }
  }
}

template<class SampleType, typename App>
__global__ void sampleParallelKernel(const int step, GPUCSRPartition graph, 
                                     const VertexID_t invalidVertex,
                                     const size_t threadsExecuted, 
                                     const size_t currExecutionThreads,
                                     const size_t totalThreads,
                                     VertexID_t* initialSamples,
                                     SampleType* samples,
                                     const size_t NumSamples,
                                     VertexID_t* finalSamplesCSRCol, 
                                     EdgePos_t* finalSamplesCSRRow,
                                     float* finalSamplesCSRVal,
                                     VertexID_t* finalSamples,
                                     const size_t finalSampleSize, 
                                     VertexID_t* explicitTransits,
                                     EdgePos_t* sampleInsertionPositions,
                                     curandState* randStates)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  //__shared__ VertexID newNeigbhors[N_THREADS];
  graph.device_csr = (CSRPartition*)&csrPartitionBuff[0];
  if (threadId >= currExecutionThreads)
    return;

  curandState* randState = &randStates[threadId];
  threadId += threadsExecuted;
  EdgePos_t numTransits = numberOfTransits<App>(step);
  EdgePos_t numTransitsInPrevStep = numberOfTransits<App>(step - 1);
  VertexID_t sampleIdx = threadId / numTransits;
  VertexID_t* transits = nullptr;
  EdgePos_t numTransitsInNeghbrhood = 0;
  //TODO: Template this kernel based on the sampling type
  if (App().samplingType() == CollectiveNeighborhood) {
    assert(!App().hasExplicitTransits());
    numTransitsInNeghbrhood = numberOfTransits<App>(step);
    if (step == 0) {
      transits = &initialSamples[sampleIdx*App().initialSampleSize(nullptr)];
    } else {
      size_t verticesAddTillPreviousStep = stepSizeAtStep<App>(step - 2);
      //printf("verticesAddTillPreviousStep %ld\n", verticesAddTillPreviousStep); 

      transits = &finalSamples[sampleIdx*finalSampleSize + verticesAddTillPreviousStep];
    }
  } else {
    if (step == 0) {
      EdgePos_t transitIdx = threadId % numTransits;
      transits = &initialSamples[sampleIdx*App().initialSampleSize(nullptr) + transitIdx];
    } else if (App().hasExplicitTransits()) {
      transits = &explicitTransits[sampleIdx*numTransitsInPrevStep + (threadId % numTransits) % numTransitsInPrevStep];
    } else {
      transits = &finalSamples[sampleIdx*finalSampleSize + (step - 1) * numTransits + (threadId % numTransits) % numTransitsInPrevStep];
    }

    numTransitsInNeghbrhood = 1;
  }

  VertexID_t neighbor = invalidVertex;
  VertexID_t neighbrID = threadId % App().stepSize(step) ;//(threadId % numTransits) % numTransitsInPrevStep;
  VertexID_t transitID = (threadId % numTransits) / App().stepSize(step);

  if (*transits == invalidVertex) {
    return;
  }
  
  EdgePos_t numTransitEdges = 0;

  for (int i = 0; i < numTransitsInNeghbrhood; i++) {
    assert(graph.device_csr->has_vertex(transits[i]));
    numTransitEdges += graph.device_csr->get_n_edges_for_vertex(transits[i]);
  }
  
  if (numTransitEdges != 0) {
    const CSR::Edge* transitEdges = (App().samplingType() == CollectiveNeighborhood) ? nullptr : graph.device_csr->get_edges(*transits);
    const float* transitEdgeWeights = (App().samplingType() == CollectiveNeighborhood) ? nullptr : graph.device_csr->get_weights(*transits);
    const float maxWeight = (App().samplingType() == CollectiveNeighborhood) ? 0.0 : graph.device_csr->get_max_weight(*transits);

    neighbor = App().next(step, graph.device_csr, transits, sampleIdx, &samples[sampleIdx], maxWeight, transitEdges, transitEdgeWeights, 
                    numTransitEdges, neighbrID, randState);
#if 0
    //search if neighbor has already been selected.
    //we can do that in register if required
    newNeigbhors[threadIdx.x] = neighbor;

    bool found = false;
    for (int i = 0; i < N_THREADS; i++) {
      if (newNeigbhors[i] == neighbor) {
        found = true;
        // break;
      }
    }

    __syncwarp();
    if (found) {
      neighbor = next(step, transit, sample, transitEdges, numTransitEdges, 
        transitNeighborIdx, randState);;
    }
#endif
  }
  
  EdgePos_t insertionPos = 0; 

  size_t finalSampleSizeTillPreviousStep = 0;
  size_t neighborsToSampleAtStep = 1;
  for (int _s = 0; _s < step; _s++) {
    neighborsToSampleAtStep *= App().stepSize(_s);
    finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
  }

  

  // if (insertionPos < finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
  // }
  // assert(finalSampleSize > 0);
  // if (insertionPos >= finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sample);
  // }
  // assert(insertionPos < finalSampleSize);
  
  if (App().outputFormat() == AdjacencyMatrix && App().samplingType() == CollectiveNeighborhood) {
    finalSamples[sampleIdx*finalSampleSize + stepSizeAtStep<App>(step - 1) + neighbrID] = neighbor;
  } else if (App().outputFormat() == SampledVertices && App().samplingType() == IndividualNeighborhood) {
    if (numberOfTransits<App>(step) > 1) {    
      insertionPos = finalSampleSizeTillPreviousStep + (threadId % numTransits);
    } else {
      insertionPos = step;
    }
    finalSamples[sampleIdx*finalSampleSize + insertionPos] = neighbor;
  }
  // if (sample == 100) {
  //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
  // }
  //TODO: We do not need atomic instead store indices of transit in another array,
  //wich can be accessed based on sample and transitIdx.
}

template<int TB_THREADS>
__global__ void partitionTransitsInKernels(int step, EdgePos_t* uniqueTransits, EdgePos_t* uniqueTransitCounts, 
                                           EdgePos_t* transitPositions,
                                           EdgePos_t uniqueTransitCountsNum, VertexID_t invalidVertex,
                                           EdgePos_t* gridKernelTransits, EdgePos_t* gridKernelTransitsNum,
                                           EdgePos_t* threadBlockKernelTransits, EdgePos_t* threadBlockKernelTransitsNum,
                                           EdgePos_t* subWarpKernelTransits, EdgePos_t* subWarpKernelTransitsNum,
                                           EdgePos_t* identityKernelTransits, EdgePos_t* identityKernelTransitsNum,
                                           int* kernelTypeForTransit, VertexID_t* transitToSamplesKeys) 
{
  //__shared__ EdgePos_t insertionPosOfThread[TB_THREADS];
  const int SHMEM_SIZE = 7*TB_THREADS;
  // __shared__ EdgePos_t trThreadBlocks[TB_THREADS];
  // __shared__ EdgePos_t trStartPos[TB_THREADS];
  typedef cub::BlockScan<int, TB_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ EdgePos_t shGridKernelTransits[SHMEM_SIZE];
  //__shared__ EdgePos_t warpsLastThreadVals;
  __shared__ EdgePos_t threadToTransitPrefixSum[TB_THREADS];
  __shared__ EdgePos_t threadToTransitPos[TB_THREADS];
  __shared__ VertexID_t threadToTransit[TB_THREADS];
  __shared__ EdgePos_t totalThreadGroups;
  __shared__ EdgePos_t threadGroupsInsertionPos;
//  __shared__ EdgePos_t gridKernelTransitsIter;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    totalThreadGroups = 0;
  //  gridKernelTransitsIter = 0;
  }

  for (int i = threadIdx.x; i < SHMEM_SIZE; i+= blockDim.x) {
    shGridKernelTransits[i] = 0;
  }

  __syncthreads();
  
  VertexID_t transit = uniqueTransits[threadId];
  EdgePos_t trCount = (threadId >= uniqueTransitCountsNum || transit == invalidVertex) ? -1: uniqueTransitCounts[threadId];
  EdgePos_t trPos = (threadId >= uniqueTransitCountsNum || transit == invalidVertex) ? -1: transitPositions[threadId];

  int kernelType = -1;
  EdgePos_t numThreadGroups = 0;
  if (trCount >= LoadBalancing::LoadBalancingThreshold::GridLevel) {    
    kernelType = TransitKernelTypes::GridKernel;
  } else if (trCount > LoadBalancing::LoadBalancingThreshold::BlockLevel) {
    kernelType = TransitKernelTypes::ThreadBlockKernel;
    // numThreadGroups = 0;
    // threadToTransitPos[threadIdx.x] = 0;
  } else if (trCount >= LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
    kernelType = TransitKernelTypes::SubWarpKernel;
    
    // numThreadGroups = 0;
    // threadToTransitPos[threadIdx.x] = 0;
  } // else {
  //   kernelType = TransitKernelTypes::IdentityKernel;
  //   // numThreadGroups = 0;
  //   // threadToTransitPos[threadIdx.x] = 0;
  // }
  
  if (threadId < uniqueTransitCountsNum && transit != invalidVertex) {
    kernelTypeForTransit[transit] = kernelType;
  }

  __syncthreads();

  for (int kTy = 1; kTy < TransitKernelTypes::SubWarpKernel + 1; kTy++) {
    EdgePos_t* glKernelTransitsNum, *glKernelTransits;
    const int threadGroupSize = (kTy == TransitKernelTypes::GridKernel) ? LoadBalancing::LoadBalancingThreshold::GridLevel : 
                                (kTy == TransitKernelTypes::ThreadBlockKernel ? LoadBalancing::LoadBalancingThreshold::BlockLevel : 
                                (kTy == TransitKernelTypes::SubWarpKernel ? LoadBalancing::LoadBalancingThreshold::SubWarpLevel : -1));

    if (kTy == TransitKernelTypes::GridKernel && useGridKernel) {
      if (kernelType == TransitKernelTypes::GridKernel) {
        numThreadGroups = DIVUP(trCount, LoadBalancing::LoadBalancingThreshold::GridLevel);
        threadToTransitPos[threadIdx.x] = trPos;
        threadToTransit[threadIdx.x] = transit;
      } else {
        numThreadGroups = 0;
        threadToTransitPos[threadIdx.x] = 0;
        threadToTransit[threadIdx.x] = -1;
      } 
      glKernelTransitsNum = gridKernelTransitsNum;
      glKernelTransits = gridKernelTransits;
    } else if (kTy == TransitKernelTypes::ThreadBlockKernel && useThreadBlockKernel) {
      if (kernelType == TransitKernelTypes::ThreadBlockKernel) {
        numThreadGroups = DIVUP(trCount, LoadBalancing::LoadBalancingThreshold::BlockLevel);
        threadToTransitPos[threadIdx.x] = trPos;
        threadToTransit[threadIdx.x] = transit;
      } else {
        numThreadGroups = 0;
        threadToTransitPos[threadIdx.x] = 0;
        threadToTransit[threadIdx.x] = -1;
      }       
      glKernelTransitsNum = threadBlockKernelTransitsNum;
      glKernelTransits = threadBlockKernelTransits;
    } else if (kTy == TransitKernelTypes::SubWarpKernel && useSubWarpKernel) {
      if (kernelType == TransitKernelTypes::SubWarpKernel) {
        numThreadGroups = DIVUP(trCount, LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
        threadToTransitPos[threadIdx.x] = trPos;
        threadToTransit[threadIdx.x] = transit;
        //printf("blockIdx.x %d threadIdx.x %d transit %d trCount %d numThreadgroups %d\n", threadIdx.x, transit, trCount, numThreadGroups);
      } else {
        numThreadGroups = 0;
        threadToTransitPos[threadIdx.x] = 0;
        threadToTransit[threadIdx.x] = -1;
      }       
      glKernelTransitsNum = subWarpKernelTransitsNum;
      glKernelTransits = subWarpKernelTransits;
    } else {
      continue;
    }

    //Get all grid kernel transits
    EdgePos_t prefixSumThreadData = 0;
    BlockScan(temp_storage).ExclusiveSum(numThreadGroups, prefixSumThreadData);
    
    __syncthreads();

    if (threadIdx.x == blockDim.x - 1) {
      totalThreadGroups = prefixSumThreadData + numThreadGroups;
      threadGroupsInsertionPos = ::atomicAdd(glKernelTransitsNum, totalThreadGroups);
    }

    threadToTransitPrefixSum[threadIdx.x] = prefixSumThreadData;
    
    __syncthreads();
    
    // if (totalThreadGroups != 0 and numThreadGroups != 0) {
    //   printf("threadIdx.x %d blockIdx.x %d tr %d trPos %d numThreadGroups %d totalThreadGroups %d prefixSumThreadData %d\n", threadIdx.x, blockIdx.x, transit, trPos, numThreadGroups, totalThreadGroups, prefixSumThreadData);
    // }
    
    for (int tgIter = 0; tgIter < totalThreadGroups; tgIter += SHMEM_SIZE) {
      for (int i = threadIdx.x; i < SHMEM_SIZE; i+= blockDim.x) {
        shGridKernelTransits[i] = 0;
      }
    
      __syncthreads();
      
      int prefixSumIndex = prefixSumThreadData - tgIter;
      if (prefixSumIndex < 0 && prefixSumIndex + numThreadGroups > 0) {
        prefixSumIndex = 0;
      }
      if (numThreadGroups > 0) {
        if (prefixSumIndex >= 0 && prefixSumIndex < SHMEM_SIZE)
          shGridKernelTransits[prefixSumIndex] = threadIdx.x;
      }
      
      __syncthreads();

      for (int tbs = threadIdx.x; tbs < DIVUP(min(SHMEM_SIZE, totalThreadGroups - tgIter), TB_THREADS)*TB_THREADS; tbs += blockDim.x) {
        int d;
        if (tbs < TB_THREADS) {
          d = (tbs < totalThreadGroups) ? shGridKernelTransits[tbs] : 0;
        } else if (threadIdx.x == 0) {
          d = (tbs < totalThreadGroups) ? max(shGridKernelTransits[tbs], shGridKernelTransits[tbs-1]): 0;
        } else {
          d = (tbs < totalThreadGroups) ? shGridKernelTransits[tbs] : 0;
        }
        
        __syncthreads();
        BlockScan(temp_storage).InclusiveScan(d, d, cub::Max());
        __syncthreads();

        if (tbs < totalThreadGroups)
          shGridKernelTransits[tbs] = d;
          
        __syncthreads();

        
        int previousTrPrefixSum = (tbs < totalThreadGroups && shGridKernelTransits[tbs] >= 0) ? threadToTransitPrefixSum[shGridKernelTransits[tbs]] : 0;

        if (tbs + tgIter < totalThreadGroups) {
          // if (step == 1) {
          //   printf("blockIdx.x %d shGridKernelTransits[tbs] %d tbs %d\n", blockIdx.x, shGridKernelTransits[tbs], tbs);
          // }
          EdgePos_t startPos = threadToTransitPos[shGridKernelTransits[tbs]];
          EdgePos_t pos = startPos + threadGroupSize*(tbs  + tgIter - previousTrPrefixSum);
          VertexID_t transit = threadToTransit[shGridKernelTransits[tbs]];
          glKernelTransits[threadGroupsInsertionPos + tbs + tgIter] = pos;
          assert(kernelTypeForTransit[transit] == kTy);
          // if (transitToSamplesKeys[pos] != transit) {
          //   printf("blockIdx.x %d shGridKernelTransits[tbs] %d tbs %d tgIter %d startPos %d pos %d expectedTr %d threadTr %d\n", blockIdx.x, shGridKernelTransits[tbs], tbs, tgIter, startPos, pos, transitToSamplesKeys[pos], transit);
          // }
          assert(transitToSamplesKeys[pos] == transit);
        }
      }

      __syncthreads();
    }

    // if (threadIdx.x==0){
    //   for (int i = 0; i < totalThreadGroups; i++) {
    //    // printf("blockIdx.x %d gridKernelTransits[%d] %d step %d\n", blockIdx.x, i, gridKernelTransits[threadGroupsInsertionPos + i], step);
    //   }
    // }

    __syncthreads();
  }

  // if (threadIdx.x+blockIdx.x*blockDim.x==0) {
  //   printf("subWarpKernelTransitsNum %d\n", *subWarpKernelTransitsNum);
  // }
  #if 0
  int done = 0;
  int startCopyingIteration = prefixSumThreadData/SHMEM_SIZE;
  int endCopyingIteration = (prefixSumThreadData + numThreadGroups)/SHMEM_SIZE;

  __syncthreads();

  for (int tbs = 0; tbs < gridTotalTBs; tbs += SHMEM_SIZE) {
    if (trPos >= 0 && numThreadBlocks > 0 && done < numThreadBlocks && tbs/SHMEM_SIZE >= startCopyingIteration && tbs/SHMEM_SIZE <= endCopyingIteration) {
      int todo;
      for (todo = 0; todo < min(numThreadBlocks-done, SHMEM_SIZE); todo++) {
        int idx = prefixSumThreadData + done - tbs + todo;
        if (idx >= SHMEM_SIZE) {
          break;
        }
        if (idx < 0 || idx >= SHMEM_SIZE) {
          printf("idx %d prefixSum %d done %d tbs %d todo %d\n", idx, prefixSumThreadData, done, tbs, todo);
        }
        shGridKernelTransits[idx] = trPos + LoadBalancing::LoadBalancingThreshold::GridLevel*(todo+done);
      }
      done += todo;
    }

    __syncthreads();

    for (EdgePos_t i = threadIdx.x; i < min(SHMEM_SIZE, gridTotalTBs - tbs); i+=blockDim.x) {
      gridKernelTransits[gridInsertionPos + tbs + i] = shGridKernelTransits[i];
    }
    __syncthreads();
  }
  #endif

  // if (threadIdx.x == 0) {
  //   for (EdgePos_t i = 0; i < gridTotalTBs; i+=1) {
  //     printf("%d %d, %d\n", blockIdx.x, i, gridKernelTransits[gridInsertionPos + i]);
  //   }
  // }
}

__global__ void init_curand_states(curandState* states, size_t num_states)
{
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if (thread_id < num_states)
    curand_init(thread_id, 0, 0, &states[thread_id]);
}

CSR* loadGraph(Graph& graph, char* graph_file, char* graph_type, char* graph_format)
{
  CSR* csr;

   //Load Graph
   if (strcmp(graph_type, "adj-list") == 0) {
    if (strcmp(graph_format, "text") == 0) {
      graph.load_from_adjacency_list(graph_file);
      //Convert graph to CSR format
      csr = new CSR(graph.get_vertices().size(), graph.get_n_edges());
      csr_from_graph (csr, graph);
      return csr;
    }
    else {
      printf ("graph_format '%s' not supported for graph_type '%s'\n", 
              graph_format, graph_type);
      return nullptr;
    }
  } else if (strcmp(graph_type, "edge-list") == 0) {
    if (strcmp(graph_format, "binary") == 0) {
      graph.load_from_edge_list_binary(graph_file, true);
      csr = new CSR(graph.get_vertices().size(), graph.get_n_edges());
      csr_from_graph (csr, graph);
      return csr;
    } else if (strcmp(graph_format, "text") == 0) {
      FILE* fp = fopen (graph_file, "r");
      if (fp == nullptr) {
        std::cout << "File '" << graph_file << "' not found" << std::endl;
        return nullptr;
      }
      graph.load_from_edge_list_txt(fp, true);
      fclose (fp);
      csr = new CSR(graph.get_vertices().size(), graph.get_n_edges());
      csr_from_graph (csr, graph);
      return csr;
    } else {
      printf ("graph_format '%s' not supported for graph_type '%s'\n", 
              graph_format, graph_type);
      return nullptr;
    }
  } else {
    printf("Incorrect graph file type '%s'\n", graph_type);
    return nullptr;
  }

  return nullptr;
}

GPUCSRPartition transferCSRToGPU(CSR* csr)
{
  //Assume that whole graph can be stored in GPU Memory.
  //Hence, only one Graph Partition is created.
  CSRPartition full_partition = CSRPartition (0, csr->get_n_vertices() - 1, 0, csr->get_n_edges() - 1, 
                                              csr->get_vertices(), csr->get_edges(), csr->get_weights());
  
  //Copy full graph to GPU
  GPUCSRPartition gpuCSRPartition;
  CSRPartition deviceCSRPartition = copyPartitionToGPU(full_partition, gpuCSRPartition);
  gpuCSRPartition.device_csr = (CSRPartition*)csrPartitionBuff;
  CHK_CU(cudaMemcpyToSymbol(csrPartitionBuff, &deviceCSRPartition, sizeof(CSRPartition)));
  return gpuCSRPartition;
}

template<typename App>
int getFinalSampleSize()
{
  size_t finalSampleSize = 0;
  size_t neighborsToSampleAtStep = 1;
  for (int step = 0; step < App().steps(); step++) {
    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      neighborsToSampleAtStep = App().stepSize(step);
    } else {
      neighborsToSampleAtStep *= App().stepSize(step);
    }

    finalSampleSize += neighborsToSampleAtStep;
  }

  return finalSampleSize;
}

template<typename SampleType, typename App>
bool allocNextDoorDataOnGPU(CSR* csr, NextDoorData<SampleType>& data)
{
  //Initially each sample contains only one vertex
  //Allocate one sample for each vertex
  int maxV = 0;
  for (int sampleIdx = 0; sampleIdx < App().numSamples(csr); sampleIdx++) {
    SampleType sample = App().template initializeSample<SampleType>(csr, sampleIdx);
    data.samples.push_back(sample);
    auto initialVertices = App().initialSample(sampleIdx, csr, data.samples[data.samples.size() - 1]);
    if ((EdgePos_t)initialVertices.size() != App().initialSampleSize(csr)) {
      //We require that number of vertices in sample initially are equal to the initialSampleSize
      printf ("initialSampleSize '%d' != initialSample(%d).size() '%ld'\n", 
              App().initialSampleSize(csr), sampleIdx, initialVertices.size());
      abort();
    }

    data.initialContents.insert(data.initialContents.end(), initialVertices.begin(), initialVertices.end());
    for (auto v : initialVertices)
      data.initialTransitToSampleValues.push_back(sampleIdx);
  }

  for (auto vertex : csr->iterate_vertices()) {
    maxV = (maxV < vertex) ? vertex : maxV;
  }
  //Size of each sample output
  size_t maxNeighborsToSample = App().initialSampleSize(csr); //TODO: Set initial vertices
  for (int step = 0; step < App().steps() - 1; step++) {
    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      maxNeighborsToSample = max((long)App().stepSize(step), maxNeighborsToSample);
    } else {
      maxNeighborsToSample *= App().stepSize(step);
    }
  }

  int finalSampleSize = getFinalSampleSize<App>();
  std::cout << "Final Size of each sample: " << finalSampleSize << std::endl;
  std::cout << "Maximum Neighbors Sampled at each step: " << maxNeighborsToSample << std::endl;
  std::cout << "Number of Samples: " << App().numSamples(csr) << std::endl;
  data.INVALID_VERTEX = csr->get_n_vertices();
  int maxBits = 0;
  while ((data.INVALID_VERTEX >> maxBits) != 0) {
    maxBits++;
  }
  
  data.maxBits = maxBits;
  
  // size_t free = 0, total = 0;
  // CHK_CU(cudaMemGetInfo(&free, &total));
  // printf("free memory %ld nextDoorData.samples.size() %ld maxNeighborsToSample %ld\n", free, data.samples.size(), maxNeighborsToSample);
  const size_t numSamples = data.samples.size();

  //Allocate storage and copy initial samples on GPU
  CHK_CU(cudaMalloc(&data.dInitialSamples, sizeof(VertexID_t)*data.initialContents.size()));
  CHK_CU(cudaMemcpy(data.dInitialSamples, &data.initialContents[0], 
                    sizeof(VertexID_t)*data.initialContents.size(), cudaMemcpyHostToDevice));

  //Allocate storage for samples on GPU
  if (sizeof(SampleType) > 0) {
    CHK_CU(cudaMalloc(&data.dOutputSamples, sizeof(SampleType)*numSamples));
    CHK_CU(cudaMemcpy(data.dOutputSamples, &data.samples[0], sizeof(SampleType)*numSamples, 
                      cudaMemcpyHostToDevice));
  }

  //Allocate storage for final samples on GPU
  data.hFinalSamples = std::vector<VertexID_t>(finalSampleSize*numSamples);

  //TODO: Do not need this when output is adjacency matrix
  CHK_CU(cudaMalloc(&data.dFinalSamples, sizeof(VertexID_t)*data.hFinalSamples.size()));
  gpu_memset(data.dFinalSamples, data.INVALID_VERTEX, data.hFinalSamples.size());
  //Samples to Transit Map

  CHK_CU(cudaMalloc(&data.dSamplesToTransitMapKeys, sizeof(VertexID_t)*numSamples*maxNeighborsToSample));

  CHK_CU(cudaMalloc(&data.dSamplesToTransitMapValues, sizeof(VertexID_t)*numSamples*maxNeighborsToSample));

  //Transit to Samples Map
  CHK_CU(cudaMalloc(&data.dTransitToSampleMapKeys, sizeof(VertexID_t)*numSamples*maxNeighborsToSample));
  CHK_CU(cudaMalloc(&data.dTransitToSampleMapValues, sizeof(VertexID_t)*numSamples*maxNeighborsToSample));

  //Same as initial values of samples for first iteration
  CHK_CU(cudaMemcpy(data.dTransitToSampleMapKeys, &data.samples[0], sizeof(VertexID_t)*numSamples, 
                    cudaMemcpyHostToDevice));
  CHK_CU(cudaMemcpy(data.dTransitToSampleMapValues, &data.samples[0], sizeof(VertexID_t)*numSamples, 
                    cudaMemcpyHostToDevice));
  //Insertion positions per transit vertex for each sample
  CHK_CU(cudaMalloc(&data.dSampleInsertionPositions, sizeof(EdgePos_t)*numSamples));
  size_t curandDataSize = maxNeighborsToSample*numSamples*sizeof(curandState);
  const size_t curandSizeLimit = 5L*1024L*1024L*8*sizeof(curandState);
  if (curandDataSize < curandSizeLimit) {
    data.maxThreadsPerKernel = maxNeighborsToSample*numSamples;
  } else {
    data.maxThreadsPerKernel = curandSizeLimit/sizeof(curandState);
    curandDataSize = curandSizeLimit;
  }
  printf("Maximum Threads Per Kernel: %ld\n", data.maxThreadsPerKernel);
  CHK_CU(cudaMalloc(&data.dCurandStates, curandDataSize));
  init_curand_states<<<thread_block_size(data.maxThreadsPerKernel, 256UL), 256UL>>> (data.dCurandStates, data.maxThreadsPerKernel);
  CHK_CU(cudaDeviceSynchronize());

  if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
    CHK_CU(cudaMalloc(&data.dNeighborhoodSizes, sizeof(EdgePos_t)*numSamples));
  }

  return true;
}

template<class SampleType>
void freeDeviceData(NextDoorData<SampleType>& data) 
{
  CHK_CU(cudaFree(data.dSamplesToTransitMapKeys));
  CHK_CU(cudaFree(data.dSamplesToTransitMapValues));
  CHK_CU(cudaFree(data.dTransitToSampleMapKeys));
  CHK_CU(cudaFree(data.dTransitToSampleMapValues));
  CHK_CU(cudaFree(data.dSampleInsertionPositions));
  CHK_CU(cudaFree(data.dCurandStates));
  CHK_CU(cudaFree(data.dFinalSamples));
  CHK_CU(cudaFree(data.gpuCSRPartition.device_vertex_array));
  CHK_CU(cudaFree(data.gpuCSRPartition.device_edge_array));
  CHK_CU(cudaFree(data.gpuCSRPartition.device_weights_array));
}

void printKernelTypes(CSR* csr, VertexID_t* dUniqueTransits, VertexID_t* dUniqueTransitsCounts, EdgePos_t* dUniqueTransitsNumRuns)
{
  EdgePos_t* hUniqueTransitsNumRuns = GPUUtils::copyDeviceMemToHostMem(dUniqueTransitsNumRuns, 1);
  VertexID_t* hUniqueTransits = GPUUtils::copyDeviceMemToHostMem(dUniqueTransits, *hUniqueTransitsNumRuns);
  VertexID_t* hUniqueTransitsCounts = GPUUtils::copyDeviceMemToHostMem(dUniqueTransitsCounts, *hUniqueTransitsNumRuns);

  size_t identityKernelTransits = 0, identityKernelSamples = 0, maxEdgesOfIdentityTransits = 0;
  size_t subWarpLevelTransits = 0, subWarpLevelSamples = 0, maxEdgesOfSubWarpTransits = 0, subWarpTransitsWithEdgesLessThan384 = 0, subWarpTransitsWithEdgesMoreThan384 = 0, numSubWarps = 0;
  size_t threadBlockLevelTransits = 0, threadBlockLevelSamples = 0, tbVerticesWithEdgesLessThan3K = 0, tbVerticesWithEdgesMoreThan3K = 0;
  size_t gridLevelTransits = 0, gridLevelSamples = 0, gridVerticesWithEdgesLessThan10K = 0, gridVerticesWithEdgesMoreThan10K = 0;
  EdgePos_t maxEdgesOfGridTransits = 0;

  for (EdgePos_t tr = 0; tr < *hUniqueTransitsNumRuns; tr++) {
    // if (tr == 0) {printf("%s:%d hUniqueTransitsCounts[0] is %d\n", __FILE__, __LINE__, hUniqueTransitsCounts[tr]);}
    if (hUniqueTransitsCounts[tr] < 8) {
      identityKernelTransits++;
      identityKernelSamples += hUniqueTransitsCounts[tr];
      maxEdgesOfIdentityTransits = max(maxEdgesOfIdentityTransits, (size_t)csr->n_edges_for_vertex(tr));
    } else if (hUniqueTransitsCounts[tr] <= LoadBalancing::LoadBalancingThreshold::BlockLevel && 
               hUniqueTransitsCounts[tr] >= 8) {
      subWarpLevelTransits++;
      subWarpLevelSamples += hUniqueTransitsCounts[tr];
      maxEdgesOfSubWarpTransits = max(maxEdgesOfSubWarpTransits, (size_t)csr->n_edges_for_vertex(tr));
      numSubWarps += DIVUP(hUniqueTransitsCounts[tr], LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
      if (csr->n_edges_for_vertex(tr) <= 96) {
        subWarpTransitsWithEdgesLessThan384 += 1;
      } else {
        subWarpTransitsWithEdgesMoreThan384 += 1;
      }
    } else if (hUniqueTransitsCounts[tr] > LoadBalancing::LoadBalancingThreshold::BlockLevel && 
               hUniqueTransitsCounts[tr] <= LoadBalancing::LoadBalancingThreshold::GridLevel) {
      threadBlockLevelTransits++;
      threadBlockLevelSamples += hUniqueTransitsCounts[tr];
      if (csr->n_edges_for_vertex(tr) <= 3*1024) {
        tbVerticesWithEdgesLessThan3K += 1;
      } else {
        tbVerticesWithEdgesMoreThan3K += 1;
      }
    } else {
      gridLevelTransits++;
      gridLevelSamples += hUniqueTransitsCounts[tr];
      if (csr->n_edges_for_vertex(tr) <= 3*1024) {
        gridVerticesWithEdgesLessThan10K += 1;
      } else {
        gridVerticesWithEdgesMoreThan10K += 1;
      }
      maxEdgesOfGridTransits = max(maxEdgesOfGridTransits, csr->n_edges_for_vertex(tr));
    }
  }

  printf("IdentityKernelTransits: %ld, IdentityKernelSamples: %ld, MaxEdgesOfIdentityTransits: %ld\n" 
         "SubWarpLevelTransits: %ld, SubWarpLevelSamples: %ld, MaxEdgesOfSubWarpTranits: %ld, VerticesWithEdges > 384: %ld, VerticesWithEdges <= 384: %ld, NumSubWarps: %ld\n"
         "ThreadBlockLevelTransits: %ld, ThreadBlockLevelSamples: %ld, VerticesWithEdges > 3K: %ld, VerticesWithEdges < 3K: %ld\n"
         "GridLevelTransits: %ld, GridLevelSamples: %ld, VerticesWithEdges > 10K: %ld, VerticesWithEdges < 10K: %ld, MaxEdgesOfTransit: %d\n", 
         identityKernelTransits, identityKernelSamples, maxEdgesOfIdentityTransits, 
         subWarpLevelTransits, subWarpLevelSamples, maxEdgesOfSubWarpTransits, 
            subWarpTransitsWithEdgesMoreThan384, subWarpTransitsWithEdgesLessThan384, numSubWarps, 
         threadBlockLevelTransits, threadBlockLevelSamples, tbVerticesWithEdgesMoreThan3K, tbVerticesWithEdgesLessThan3K,
         gridLevelTransits, gridLevelSamples, gridVerticesWithEdgesMoreThan10K, gridVerticesWithEdgesLessThan10K, maxEdgesOfGridTransits);

  delete hUniqueTransits;
  delete hUniqueTransitsCounts;
  delete hUniqueTransitsNumRuns;
}

template<class SampleType, typename App>
bool doTransitParallelSampling(CSR* csr, GPUCSRPartition gpuCSRPartition, NextDoorData<SampleType>& nextDoorData, bool enableLoadBalancing)
{
  //Size of each sample output
  size_t maxNeighborsToSample = (App().samplingType() == CollectiveNeighborhood) ? 1 : App().initialSampleSize(csr);
  for (int step = 0; step < App().steps() - 1; step++) {
    if (App().samplingType() == CollectiveNeighborhood) {
      maxNeighborsToSample = max(maxNeighborsToSample, (size_t)App().stepSize(step));
    } else {
      maxNeighborsToSample *= App().stepSize(step);
    }
  }
  
  size_t finalSampleSize = getFinalSampleSize<App>();
  CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapKeys, &nextDoorData.initialContents[0], 
                    sizeof(VertexID_t)*nextDoorData.initialContents.size(), 
                    cudaMemcpyHostToDevice));
  CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapValues, &nextDoorData.initialTransitToSampleValues[0], 
                    sizeof(VertexID_t)*nextDoorData.initialTransitToSampleValues.size(), 
                    cudaMemcpyHostToDevice));
  VertexID_t* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  VertexID_t* dUniqueTransits = nullptr;
  VertexID_t* dUniqueTransitsCounts = nullptr;
  EdgePos_t* dUniqueTransitsNumRuns = nullptr;
  EdgePos_t* dTransitPositions = nullptr;
  EdgePos_t* uniqueTransitNumRuns = nullptr;
   
  /**Pointers for each kernel type**/
  EdgePos_t* gridKernelTransitsNum = nullptr;
  EdgePos_t* dGridKernelTransitsNum = nullptr;
  VertexID_t* dGridKernelTransits = nullptr;
  
  EdgePos_t* threadBlockKernelTransitsNum = nullptr;
  EdgePos_t* dThreadBlockKernelTransitsNum = nullptr;
  VertexID_t* dThreadBlockKernelTransits = nullptr;

  EdgePos_t* subWarpKernelTransitsNum = nullptr;
  EdgePos_t* dSubWarpKernelTransitsNum = nullptr;
  VertexID_t* dSubWarpKernelTransits = nullptr;
  /**********************************/

  /*Single Memory Location on both CPU and GPU for transferring
   *number of transits for all kernels */
  EdgePos_t* dKernelTransitNums;
  EdgePos_t* hKernelTransitNums;
  const int NUM_KERNEL_TYPES = TransitKernelTypes::NumKernelTypes;

  int* dKernelTypeForTransit = nullptr;

  CHK_CU(cudaMallocHost(&uniqueTransitNumRuns, sizeof(EdgePos_t)));
  CHK_CU(cudaMallocHost(&hKernelTransitNums, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
  
  gridKernelTransitsNum = hKernelTransitNums;
  threadBlockKernelTransitsNum = hKernelTransitNums + 1;
  subWarpKernelTransitsNum = hKernelTransitNums + 2;
  //threadBlockKernelTransitsNum = hKernelTransitNums[3];
  
  CHK_CU(cudaMalloc(&dKernelTypeForTransit, sizeof(VertexID_t)*csr->get_n_vertices()));
  CHK_CU(cudaMalloc(&dTransitPositions, 
                    sizeof(VertexID_t)*nextDoorData.samples.size()));
  CHK_CU(cudaMalloc(&dGridKernelTransits, 
                    sizeof(VertexID_t)*nextDoorData.samples.size()*maxNeighborsToSample));
  CHK_CU(cudaMalloc(&dThreadBlockKernelTransits, 
                    sizeof(VertexID_t)*nextDoorData.samples.size()*maxNeighborsToSample));
  CHK_CU(cudaMalloc(&dSubWarpKernelTransits,
                    sizeof(VertexID_t)*nextDoorData.samples.size()*maxNeighborsToSample));

  CHK_CU(cudaMalloc(&dKernelTransitNums, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
  CHK_CU(cudaMemset(dKernelTransitNums, 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
  dGridKernelTransitsNum = dKernelTransitNums;
  dThreadBlockKernelTransitsNum = dKernelTransitNums + 1;
  dSubWarpKernelTransitsNum = dKernelTransitNums + 2;

  int* atomicPtrTest = nullptr;
  CHK_CU(cudaMalloc(&atomicPtrTest, sizeof(int)));
  //Check if the space runs out.
  //TODO: Use DoubleBuffer version that requires O(P) space.
  //TODO: hFinalSamples.size() is wrong.
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
            nextDoorData.dSamplesToTransitMapValues, nextDoorData.dTransitToSampleMapKeys, 
            nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dTransitToSampleMapValues, 
            nextDoorData.samples.size()*maxNeighborsToSample);

  CHK_CU(cudaMalloc(&dUniqueTransits, (csr->get_n_vertices() + 1)*sizeof(VertexID_t)));
  CHK_CU(cudaMalloc(&dUniqueTransitsCounts, (csr->get_n_vertices() + 1)*sizeof(VertexID_t)));
  CHK_CU(cudaMalloc(&dUniqueTransitsNumRuns, sizeof(size_t)));
  
  if (temp_storage_bytes < nextDoorData.samples.size()*maxNeighborsToSample) {
    temp_storage_bytes = nextDoorData.samples.size()*maxNeighborsToSample;
  }

  // VertexID_t* gt1, *gt2;
  // CHK_CU(cudaMalloc(&gt1, nextDoorData.samples.size()*maxNeighborsToSample*sizeof(VertexID_t)));
  // CHK_CU(cudaMalloc(&gt2, nextDoorData.samples.size()*maxNeighborsToSample*sizeof(VertexID_t)));

  size_t free = 0, total = 0;
  CHK_CU(cudaMemGetInfo(&free, &total));
  // printf("free memory %ld temp_storage_bytes %ld nextDoorData.samples.size() %ld maxNeighborsToSample %ld\n", free, temp_storage_bytes, nextDoorData.samples.size(), maxNeighborsToSample);
  CHK_CU(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  std::vector<VertexID_t*> hAllSamplesToTransitMapKeys;
  std::vector<VertexID_t*> hAllTransitToSampleMapValues;

  double loadBalancingTime = 0;
  double inversionTime = 0;
  double gridKernelTime = 0;
  double subWarpKernelTime = 0;
  double identityKernelTime = 0;
  double threadBlockKernelTime = 0;
  size_t neighborsToSampleAtStep = (App().samplingType() == CollectiveNeighborhood) ? 1 : App().initialSampleSize(csr);

  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  for (int step = 0; step < App().steps(); step++) {
    const size_t numTransits = (App().samplingType() == CollectiveNeighborhood) ? 1 : neighborsToSampleAtStep;
    neighborsToSampleAtStep = (App().samplingType() == CollectiveNeighborhood) ? App().stepSize(step) : neighborsToSampleAtStep * App().stepSize(step);    
    const size_t totalThreads = App().numSamples(csr)*neighborsToSampleAtStep;

    if (step == 0 || !enableLoadBalancing) {
      //When not doing load balancing call baseline transit parallel
      for (int threadsExecuted = 0; threadsExecuted < totalThreads; threadsExecuted += nextDoorData.maxThreadsPerKernel) {
        size_t currExecutionThreads = min((size_t)nextDoorData.maxThreadsPerKernel, totalThreads - threadsExecuted);
        samplingKernel<SampleType, App><<<thread_block_size(currExecutionThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, 
                        threadsExecuted, currExecutionThreads, nextDoorData.INVALID_VERTEX,
                        (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys, (const VertexID_t*)nextDoorData.dTransitToSampleMapValues,
                        totalThreads, nextDoorData.dOutputSamples, nextDoorData.samples.size(),
                        nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dSamplesToTransitMapValues,
                        nextDoorData.dFinalSamples, finalSampleSize, nextDoorData.dSampleInsertionPositions,
                        nextDoorData.dCurandStates);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
      }
    } else {
      double loadBalancingT1 = convertTimeValToDouble(getTimeOfDay ());
      void* dRunLengthEncodeTmpStorage = nullptr;
      size_t dRunLengthEncodeTmpStorageSize = 0;

      cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize, 
                                        nextDoorData.dTransitToSampleMapKeys,
                                        dUniqueTransits, dUniqueTransitsCounts, dUniqueTransitsNumRuns, totalThreads);

      assert(dRunLengthEncodeTmpStorageSize < temp_storage_bytes);
      dRunLengthEncodeTmpStorage = d_temp_storage;
      cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize, 
                                        nextDoorData.dTransitToSampleMapKeys,
                                        dUniqueTransits, dUniqueTransitsCounts, dUniqueTransitsNumRuns, totalThreads);

      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
      
      CHK_CU(cudaMemcpy(uniqueTransitNumRuns, dUniqueTransitsNumRuns, sizeof(*uniqueTransitNumRuns), cudaMemcpyDeviceToHost));

      void* dExclusiveSumTmpStorage = nullptr;
      size_t dExclusiveSumTmpStorageSize = 0;
      
      cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts, dTransitPositions, *uniqueTransitNumRuns);

      assert(dExclusiveSumTmpStorageSize < temp_storage_bytes);
      dExclusiveSumTmpStorage = d_temp_storage;

      cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts, dTransitPositions, *uniqueTransitNumRuns);

      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
      // printKernelTypes(csr, dUniqueTransits, dUniqueTransitsCounts, dUniqueTransitsNumRuns);

      CHK_CU(cudaMemset(dKernelTransitNums, 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
      partitionTransitsInKernels<1024><<<thread_block_size((*uniqueTransitNumRuns), 1024), 1024>>>(step, dUniqueTransits, dUniqueTransitsCounts, 
          dTransitPositions, *uniqueTransitNumRuns, nextDoorData.INVALID_VERTEX, dGridKernelTransits, dGridKernelTransitsNum, 
          dThreadBlockKernelTransits, dThreadBlockKernelTransitsNum, dSubWarpKernelTransits, dSubWarpKernelTransitsNum, nullptr, nullptr, dKernelTypeForTransit,
          nextDoorData.dTransitToSampleMapKeys);

      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
      CHK_CU(cudaMemcpy(hKernelTransitNums, dKernelTransitNums, NUM_KERNEL_TYPES * sizeof(EdgePos_t), cudaMemcpyDeviceToHost));

      // GPUUtils::printDeviceArray(dGridKernelTransits, *gridKernelTransitsNum, ',');
      // getchar();
      double loadBalancingT2 = convertTimeValToDouble(getTimeOfDay ());
      loadBalancingTime += (loadBalancingT2 - loadBalancingT1);
      
      double identityKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
      identityKernel<SampleType, App, N_THREADS, true><<<DIVUP(totalThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
        (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys, (const VertexID_t*)nextDoorData.dTransitToSampleMapValues,
        totalThreads, nextDoorData.dOutputSamples, nextDoorData.samples.size(),
        nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dSamplesToTransitMapValues,
        nextDoorData.dFinalSamples, finalSampleSize, nextDoorData.dSampleInsertionPositions,
        nextDoorData.dCurandStates, dKernelTypeForTransit);
      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
      double identityKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
      identityKernelTime += (identityKernelTimeT2 - identityKernelTimeT1);
      
      const int perThreadSamplesForSubWarpKernel = 1;
      int threadBlocks = DIVUP(DIVUP(*subWarpKernelTransitsNum*LoadBalancing::LoadBalancingThreshold::SubWarpLevel, perThreadSamplesForSubWarpKernel), 256);
      //std::cout << "subWarpKernelTransitsNum " << *subWarpKernelTransitsNum << " threadBlocks " << threadBlocks << std::endl;
      double subWarpKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
      if (useSubWarpKernel) {
        subWarpKernel<SampleType, App, 256,3*1024,false,false,false,perThreadSamplesForSubWarpKernel,true><<<threadBlocks, 256>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
          (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys, (const VertexID_t*)nextDoorData.dTransitToSampleMapValues,
          totalThreads, nextDoorData.dOutputSamples, nextDoorData.samples.size(),
          nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dSamplesToTransitMapValues,
          nextDoorData.dFinalSamples, finalSampleSize, nextDoorData.dSampleInsertionPositions,
          nextDoorData.dCurandStates, dKernelTypeForTransit, dSubWarpKernelTransits, *subWarpKernelTransitsNum);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
      }
      double subWarpKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
      subWarpKernelTime += (subWarpKernelTimeT2 - subWarpKernelTimeT1);

      double threadBlockKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
      const int perThreadSamplesForThreadBlockKernel = 1;
      threadBlocks = DIVUP(*threadBlockKernelTransitsNum, perThreadSamplesForThreadBlockKernel);
      if (useThreadBlockKernel) {
        threadBlockKernel<SampleType, App, 256,3*1024-3,true,false,false,perThreadSamplesForThreadBlockKernel,true><<<threadBlocks, 32>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
          (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys, (const VertexID_t*)nextDoorData.dTransitToSampleMapValues,
          totalThreads, nextDoorData.dOutputSamples, nextDoorData.samples.size(),
          nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dSamplesToTransitMapValues,
          nextDoorData.dFinalSamples, finalSampleSize, nextDoorData.dSampleInsertionPositions,
          nextDoorData.dCurandStates, dKernelTypeForTransit, dThreadBlockKernelTransits, *threadBlockKernelTransitsNum);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
      }
      double threadBlockKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
      threadBlockKernelTime += (threadBlockKernelTimeT2 - threadBlockKernelTimeT1);

      const int perThreadSamplesForGridKernel = 4;
      double gridKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
      threadBlocks = DIVUP(*gridKernelTransitsNum, perThreadSamplesForGridKernel);
      if (useGridKernel) {
        gridKernel<SampleType,App,256,3*1024-3,false,false,false,perThreadSamplesForGridKernel,true><<<threadBlocks, 256>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
          (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys, (const VertexID_t*)nextDoorData.dTransitToSampleMapValues,
          totalThreads,  nextDoorData.dOutputSamples, nextDoorData.samples.size(),
          nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dSamplesToTransitMapValues,
          nextDoorData.dFinalSamples, finalSampleSize, nextDoorData.dSampleInsertionPositions,
          nextDoorData.dCurandStates, dKernelTypeForTransit, dGridKernelTransits, *gridKernelTransitsNum);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
      }
      double gridKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
      gridKernelTime += (gridKernelTimeT2 - gridKernelTimeT1);
    }

    if (step != App().steps() - 1) {
      double inversionT1 = convertTimeValToDouble(getTimeOfDay ());
      //Invert sample->transit map by sorting samples based on the transit vertices
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                      nextDoorData.dSamplesToTransitMapValues, nextDoorData.dTransitToSampleMapKeys, 
                                      nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dTransitToSampleMapValues, 
                                      totalThreads, 0, nextDoorData.maxBits);
      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
      double inversionT2 = convertTimeValToDouble(getTimeOfDay ());
      inversionTime += (inversionT2 - inversionT1);

      #if 0
      VertexID_t* hTransitToSampleMapKeys = new VertexID_t[totalThreads];
      VertexID_t* hTransitToSampleMapValues = new VertexID_t[totalThreads];
      VertexID_t* hSampleToTransitMapKeys = new VertexID_t[totalThreads];
      VertexID_t* hSampleToTransitMapValues = new VertexID_t[totalThreads];

      
      CHK_CU(cudaMemcpy(hSampleToTransitMapKeys, nextDoorData.dSamplesToTransitMapKeys, 
        totalThreads*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      CHK_CU(cudaMemcpy(hSampleToTransitMapValues, nextDoorData.dSamplesToTransitMapValues,
        totalThreads*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      CHK_CU(cudaMemcpy(hTransitToSampleMapKeys, nextDoorData.dTransitToSampleMapKeys, 
                        totalThreads*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      CHK_CU(cudaMemcpy(hTransitToSampleMapValues, nextDoorData.dTransitToSampleMapValues,
                        totalThreads*sizeof(VertexID_t), cudaMemcpyDeviceToHost));
      hAllTransitToSampleMapValues.push_back(hTransitToSampleMapValues);
      hAllSamplesToTransitMapKeys.push_back(hSampleToTransitMapKeys);
      #endif
    }
  }

  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());

  CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions, 0, sizeof(EdgePos_t)*nextDoorData.samples.size()));

  std::cout << "Transit Parallel: End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  std::cout << "InversionTime: " << inversionTime <<", " << "LoadBalancingTime: " << loadBalancingTime << ", " << "GridKernelTime: " << gridKernelTime << ", ThreadBlockKernelTime: " << threadBlockKernelTime << ", SubWarpKernelTime: " << subWarpKernelTime << ", IdentityKernelTime: "<< identityKernelTime << std::endl;
  CHK_CU(cudaFree(d_temp_storage));
  CHK_CU(cudaFree(dUniqueTransits));
  CHK_CU(cudaFree(dUniqueTransitsCounts));
  CHK_CU(cudaFree(dUniqueTransitsNumRuns));

  #if 0
  for (int s = 1; s < App().steps() - 2; s++) {
    std::unordered_set<VertexID_t> s1, s2, intersection;
    for (int i = 100000; i < 200000; i++) {
      VertexID_t v1 = hAllSamplesToTransitMapKeys[s+1][i];
      VertexID_t v2 = hAllTransitToSampleMapValues[s+2][i];
      //printf("v1 %d v2 %d\n", v1, v2);
      s1.insert(v1);
      s2.insert(v2);
    }
    
    for (auto e : s1) {
      if (s2.count(e) == 1) intersection.insert(e);
    }

    std::cout << "s: " << s << " intersection: " << intersection.size() << std::endl;
  }
  #endif
  return true;
}

template<class SampleType, typename App>
bool doSampleParallelSampling(CSR* csr, GPUCSRPartition gpuCSRPartition, NextDoorData<SampleType>& nextDoorData)
{
  //Size of each sample output
  int finalSampleSize = getFinalSampleSize<App>();
  int neighborsToSampleAtStep = App().initialSampleSize(csr);

  EdgePos_t* hSumNeighborhoodSizes;
  EdgePos_t* dSumNeighborhoodSizes;
  EdgePos_t* dSampleNeighborhoodPos;
  VertexID_t* dCollectiveNeighborhoodCSRCols;
  EdgePos_t* dCollectiveNeighborhoodCSRRows;

  if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
    CHK_CU(cudaMallocHost(&hSumNeighborhoodSizes, sizeof(EdgePos_t)));
    CHK_CU(cudaMalloc(&dSumNeighborhoodSizes, sizeof(EdgePos_t)));
    CHK_CU(cudaMalloc(&dSampleNeighborhoodPos, sizeof(EdgePos_t)*nextDoorData.samples.size()));
  }

  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  for (int step = 0; step < App().steps(); step++) {
    //Number of threads created are equal to number of new neighbors to be sampled at a step.
    //In collective neighborhood we sample stepSize(step) vertices at each step
    //Otherwise need to sample product.
    const size_t numTransits = (App().samplingType() == CollectiveNeighborhood) ? 1 : neighborsToSampleAtStep;
    neighborsToSampleAtStep = (App().samplingType() == CollectiveNeighborhood) ? App().stepSize(step) : neighborsToSampleAtStep * App().stepSize(step);
    const size_t totalThreads = App().numSamples(csr)*neighborsToSampleAtStep;
    // std::cout << "totalThreads " << totalThreads << std::endl;
    for (int threadsExecuted = 0; threadsExecuted < totalThreads; threadsExecuted += nextDoorData.maxThreadsPerKernel) {
      size_t currExecutionThreads = min(nextDoorData.maxThreadsPerKernel, totalThreads - threadsExecuted);
      if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
        //TODO: No need to do this right now.
        //Create collective neighborhood for all transits related to a sample
        collectiveNeighbrsSize<App><<<nextDoorData.samples.size(), N_THREADS>>>(step, gpuCSRPartition, 
                                                                           nextDoorData.INVALID_VERTEX,
                                                                           nextDoorData.dInitialSamples, 
                                                                           nextDoorData.dFinalSamples, 
                                                                           nextDoorData.samples.size(),
                                                                           dSampleNeighborhoodPos,
                                                                           dSumNeighborhoodSizes);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
        CHK_CU(cudaMemcpy(hSumNeighborhoodSizes, dSumNeighborhoodSizes, sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
        CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRCols, sizeof(VertexID_t)*(*hSumNeighborhoodSizes)));
        CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRRows, sizeof(EdgePos_t)*App().initialSampleSize(csr)*nextDoorData.samples.size()));

        collectiveNeighborhood<App><<<nextDoorData.samples.size(), N_THREADS>>>(step, gpuCSRPartition, 
                                                                           nextDoorData.INVALID_VERTEX,
                                                                           nextDoorData.dInitialSamples,
                                                                           nextDoorData.dFinalSamples, 
                                                                           nextDoorData.samples.size(),
                                                                           dCollectiveNeighborhoodCSRRows,
                                                                           dCollectiveNeighborhoodCSRCols,
                                                                           dSampleNeighborhoodPos,
                                                                           dSumNeighborhoodSizes);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
        
    #if 0
        //Check if the CSR is correct
        EdgePos_t* csrRows = new EdgePos_t[App().initialSampleSize(csr)*nextDoorData.samples.size()];
        EdgePos_t* csrCols = new VertexID_t[(*hSumNeighborhoodSizes)];
        EdgePos_t* samplePos = new EdgePos_t[nextDoorData.samples.size()];
        
        CHK_CU(cudaMemcpy(csrCols, dCollectiveNeighborhoodCSRCols, sizeof(VertexID_t)*(*hSumNeighborhoodSizes), 
                          cudaMemcpyDeviceToHost));
        CHK_CU(cudaMemcpy(csrRows, dCollectiveNeighborhoodCSRRows, sizeof(EdgePos_t)*App().initialSampleSize(csr)*nextDoorData.samples.size(), 
                          cudaMemcpyDeviceToHost));
        CHK_CU(cudaMemcpy(samplePos, dSampleNeighborhoodPos, sizeof(EdgePos_t)*nextDoorData.samples.size(), 
                          cudaMemcpyDeviceToHost));
        const int SZ = App().initialSampleSize(csr)*nextDoorData.samples.size();
        for (int sample = 0; sample < nextDoorData.samples.size(); sample++) {
          for (int v = 0; v < App().initialSampleSize(csr); v++) {
            EdgePos_t edgeStart = csrRows[sample * App().initialSampleSize(csr) + v];
            EdgePos_t edgeEnd = -1;
            EdgePos_t idxInRows = sample * App().initialSampleSize(csr) + v;
            
            //TODO: Add one more field to a vertex to each sample that is the length of all edges.
            if (v + 1 == App().initialSampleSize(csr)) {
              continue;
            }
            if (v + 1 < App().initialSampleSize(csr)) {
              edgeEnd = csrRows[idxInRows + 1];
            } else if (sample + 1 < nextDoorData.samples.size()) {
              edgeEnd = samplePos[sample + 1];
            } else {
              edgeEnd = (*hSumNeighborhoodSizes);
            }
             
            VertexID transit = nextDoorData.initialContents[sample * App().initialSampleSize(csr) + v];
            if (edgeEnd - edgeStart != csr->n_edges_for_vertex(transit)) {
              printf("transit %d edgeEnd %d edgeStart %d csr->n_edges_for_vertex(transit) %d\n", transit, edgeEnd, edgeStart, csr->n_edges_for_vertex(transit));
            }
            assert(edgeEnd - edgeStart == csr->n_edges_for_vertex(transit));
          }
        }
    #endif
        /*Sorting takes a ton of time (2-3x more). So, it probably be benificial to 
         * create a CSR matrix of the neighborhood of transit vertices.*/
        //Sort these edges of neighborhood
        /****************************
        void* dTempStorage = nullptr;
        size_t dTempStorageBytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(dTempStorage, dTempStorageBytes, (const VertexID_t*)dCollectiveNeighborhood, 
                                                dCollectiveNeighborhood + sizeof(VertexID_t)*(*hSumNeighborhoodSizes), 
                                                *hSumNeighborhoodSizes, (int)nextDoorData.samples.size(),
                                                dSampleNeighborhoodPos, dSampleNeighborhoodPos + 1, 0, nextDoorData.maxBits);
        
        CHK_CU(cudaMalloc(&dTempStorage, dTempStorageBytes));
        cub::DeviceSegmentedRadixSort::SortKeys(dTempStorage, dTempStorageBytes, (const VertexID_t*)dCollectiveNeighborhood, 
                                                dCollectiveNeighborhood + sizeof(VertexID_t)*(*hSumNeighborhoodSizes), 
                                                *hSumNeighborhoodSizes, (int)nextDoorData.samples.size(),
                                                dSampleNeighborhoodPos, dSampleNeighborhoodPos + 1, 0, nextDoorData.maxBits);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
        ****************************/
      }
      if (App().hasExplicitTransits() and step > 0) {
        const size_t totalThreads = App().numSamples(csr)*neighborsToSampleAtStep;
        for (int _thExecs = 0; _thExecs < totalThreads; _thExecs += nextDoorData.maxThreadsPerKernel) {
          const size_t currExecThreads = min(nextDoorData.maxThreadsPerKernel, totalThreads - _thExecs);

          explicitTransitsKernel<SampleType, App, false><<<DIVUP(currExecThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, 
                                                                                                      nextDoorData.INVALID_VERTEX,
                                                                                                      _thExecs, currExecThreads,
                                                                                                      totalThreads,
                                                                                                      nextDoorData.dOutputSamples,
                                                                                                      nextDoorData.samples.size(),
                                                                                                      nullptr,
                                                                                                      nextDoorData.dSamplesToTransitMapValues,
                                                                                                      nextDoorData.dCurandStates);
          
          CHK_CU(cudaGetLastError());
          CHK_CU(cudaDeviceSynchronize());
        }
      }
//       template<class SampleType>
// __global__ void sampleParallelKernel(const int step, GPUCSRPartition graph, 
//                                      const VertexID_t invalidVertex,
//                                      const size_t threadsExecuted, 
//                                      const size_t currExecutionThreads,
//                                      const size_t totalThreads,
//                                      VertexID_t* initialSamples,
//                                      SampleType* samples,
//                                      const size_t NumSamples,
//                                      VertexID_t* finalSamplesCSRCol, 
//                                      EdgePos_t* finalSamplesCSRRow,
//                                      float* finalSamplesCSRVal,
//                                      VertexID_t* finalSamples,
//                                      const size_t finalSampleSize, 
//                                      VertexID_t* explicitTransits,
//                                      EdgePos_t* sampleInsertionPositions,
//                                      curandState* randStates)
      //Perform SampleParallel Sampling
      sampleParallelKernel<SampleType, App><<<thread_block_size(currExecutionThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, 
                    nextDoorData.INVALID_VERTEX,
                    threadsExecuted, currExecutionThreads, totalThreads, 
                    nextDoorData.dInitialSamples, nextDoorData.dOutputSamples, nextDoorData.samples.size(),
                    nextDoorData.dFinalSamplesCSRCol, nextDoorData.dFinalSamplesCSRRow, nextDoorData.dFinalSamplesCSRVal, 
                    nextDoorData.dFinalSamples, finalSampleSize, 
                    nextDoorData.dSamplesToTransitMapValues,
                    nextDoorData.dSampleInsertionPositions, nextDoorData.dCurandStates);
      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
    }
  }

  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());
  

  CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions, 0, sizeof(EdgePos_t)*nextDoorData.samples.size()));

  std::cout << "SampleParallel: End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  return true;
}

template<class SampleType>
std::vector<VertexID_t>& getFinalSamples(NextDoorData<SampleType>& nextDoorData)
{
  CHK_CU(cudaMemcpy(&nextDoorData.hFinalSamples[0], nextDoorData.dFinalSamples, 
                    nextDoorData.hFinalSamples.size()*sizeof(nextDoorData.hFinalSamples[0]), cudaMemcpyDeviceToHost));
  CHK_CU(cudaMemcpy(&nextDoorData.samples[0], nextDoorData.dOutputSamples, 
                    nextDoorData.samples.size()*sizeof(SampleType), cudaMemcpyDeviceToHost));
  return nextDoorData.hFinalSamples;
}

template<class SampleType, typename App>
bool nextdoor(const char* graph_file, const char* graph_type, const char* graph_format, 
             const int nruns, const bool chk_results, const bool print_samples,
             const char* kernelType, const bool enableLoadBalancing,
             bool (*checkResultsFunc)(NextDoorData<SampleType>&))
{
  std::vector<Vertex> vertices;

  //Load Graph
  Graph graph;
  CSR* csr;
  if ((csr = loadGraph(graph, (char*)graph_file, (char*)graph_type, (char*)graph_format)) == nullptr) {
    return false;
  }

  std::cout << "Graph has " <<graph.get_n_edges () << " edges and " << 
      graph.get_vertices ().size () << " vertices " << std::endl; 

  //graph.print(std::cout);
  GPUCSRPartition gpuCSRPartition = transferCSRToGPU(csr);
  
  NextDoorData<SampleType> nextDoorData;
  nextDoorData.csr = csr;
  nextDoorData.gpuCSRPartition = gpuCSRPartition;
  allocNextDoorDataOnGPU<SampleType, App>(csr, nextDoorData);
  
  for (int i = 0; i < nruns; i++) {
    if (strcmp(kernelType, "TransitParallel") == 0)
      doTransitParallelSampling<SampleType, App>(csr, gpuCSRPartition, nextDoorData, enableLoadBalancing);
    else if (strcmp(kernelType, "SampleParallel") == 0)
      doSampleParallelSampling<SampleType, App>(csr, gpuCSRPartition, nextDoorData);
    else
      abort();
  }
    

  getFinalSamples(nextDoorData);

  size_t maxNeighborsToSample = 1;
  for (int step = 0; step < App().steps(); step++) {
    maxNeighborsToSample *= App().stepSize(step);
  }

  size_t finalSampleSize = getFinalSampleSize<App>();
  
  size_t totalSampledVertices = 0;

  for (auto s : nextDoorData.hFinalSamples) {
    totalSampledVertices += (int)(s != nextDoorData.INVALID_VERTEX);
  }

  if (print_samples) {
    for (size_t s = 0; s < nextDoorData.hFinalSamples.size(); s += finalSampleSize) {
      std::cout << "Contents of sample " << s/finalSampleSize << " [";
      for(size_t v = s; v < s + finalSampleSize; v++)
        std::cout << nextDoorData.hFinalSamples[v] << ", ";
      std::cout << "]" << std::endl;
    }
  }

  std::cout << "totalSampledVertices " << totalSampledVertices << std::endl;
  freeDeviceData(nextDoorData);
  if (chk_results) {
      return checkResultsFunc(nextDoorData);
  }
  
  return true;
}

#endif