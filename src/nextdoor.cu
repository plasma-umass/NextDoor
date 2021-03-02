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

#include "csr.hpp"
#include "utils.hpp"
#include "sampler.cuh"
#include "rand_num_gen.cuh"
#include "libNextDoor.hpp"

using namespace utils;
using namespace GPUUtils;

#define CHECK_RESULT

const size_t N_THREADS = 256;


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

template<typename T, size_t CACHE_SIZE, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE>
struct CachedArray {
  const T* glArray;
  T* shArray;
  
  __device__
  T operator[](int id)
  {
    return at(id);
  }

  __device__
  T at(int id)
  {
    if (id >= CACHE_SIZE) {
      return glArray[id];
    }
    
    VertexID_t e;

    // if (false && COALESCE_GL_LOADS) {
    //   e = cachedEdges[id];
    //   if (e == -1)
    //     cachedEdges[id] = -2;

    //   int subWarpThreadIdx = threadIdx.x % LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
    //   //int subWarp = threadIdx.x / LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
    //   for (int i = subWarpThreadIdx; i < CACHE_SIZE; i += LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
    //     if (cachedEdges[i] == -2) {
    //       cachedEdges[i] = transitEdges[i];
    //     }
    //   }
      
    //   e = cachedEdges[id];
    // } else 
    {
      e = shArray[id];
      if (id < STATIC_CACHE_SIZE)
        return e;

      if (ONDEMAND_CACHING and e == -1) {
        e = glArray[id];
        shArray[id] = e;
      }
    }

    return e;
  }
};

template<typename App>
__host__ __device__
EdgePos_t newNeighborsSize(int hop, EdgePos_t num_edges)
{
  return (App().stepSize(hop) == ALL_NEIGHBORS) ? num_edges : (EdgePos_t)App().stepSize(hop);
}

template<typename App>
__host__ __device__
EdgePos_t subWarpSizeAtStep(int step)
{
  if (step == -1)
    return 0;
  
  //SubWarpSize is set to next power of 2
  
  EdgePos_t x = App().stepSize(step);

  if (x && (!(x&(x-1)))) {
    return x;
  } 

  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  if (sizeof(EdgePos_t) == sizeof(uint64_t)) {
    //x |= x >> 32;
  }
  x++;

  return x;
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

__host__ __device__ bool isValidSampledVertex(VertexID_t neighbor, VertexID_t InvalidVertex) 
{
  return neighbor != InvalidVertex && neighbor != -1;
}

enum TransitParallelMode {
  //Describes the execution mode of Transit Parallel.
  NextFuncExecution, //Execute the next function
  CollectiveNeighborhoodSize, //Compute size of collective neighborhood
  CollectiveNeighborhoodComputation, //Compute the collective neighborhood 
};

#define STORE_TRANSIT_INDEX false
template<class SamplingType, typename App, TransitParallelMode tpMode, int CollNeighStepSize>
__global__ void samplingKernel(const int step, GPUCSRPartition graph, const size_t threadsExecuted, const size_t currExecutionThreads,
                               const VertexID_t deviceFirstSample, const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, SamplingType* samples, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               EdgePos_t* sampleNeighborhoodSizes, EdgePos_t* sampleNeighborhoodPos, 
                               VertexID_t* collectiveNeighborhoodCSRRows, 
                               EdgePos_t* collectiveNeighborhoodCSRCols, curandState* randStates)
{
  EdgePos_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
  //__shared__ VertexID newNeigbhors[N_THREADS];

  if (threadId >= currExecutionThreads)
    return;
  
  curandState* randState = &randStates[threadId];

  threadId += threadsExecuted;
  int stepSize;
  if (tpMode == NextFuncExecution) {
    stepSize = App().stepSize(step);
  } else if (tpMode == CollectiveNeighborhoodComputation) {
    stepSize = CollNeighStepSize;
  } else if (tpMode == CollectiveNeighborhoodSize) {
    stepSize = 1;
  }
  EdgePos_t transitIdx = threadId/App().stepSize(step);
  EdgePos_t transitNeighborIdx = threadId % App().stepSize(step);
  EdgePos_t numTransits = numberOfTransits<App>(step);

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
    
    if (numTransitEdges != 0 && (tpMode == NextFuncExecution || tpMode == CollectiveNeighborhoodComputation)) {
      //Execute next in this mode only
      const CSR::Edge* transitEdges = graph.device_csr->get_edges(transit);
      const float* transitEdgeWeights = graph.device_csr->get_weights(transit);
      const float maxWeight = graph.device_csr->get_max_weight(transit);
      if (tpMode == NextFuncExecution) {
        neighbor = App().next(step, graph.device_csr, &transit, sampleIdx, &samples[(sampleIdx - deviceFirstSample)], maxWeight, transitEdges, transitEdgeWeights, 
                              numTransitEdges, transitNeighborIdx, randState);
      } else {
        int insertionPos = utils::atomicAdd(&sampleInsertionPositions[sampleIdx- deviceFirstSample], numTransitEdges);
        collectiveNeighborhoodCSRRows[(sampleIdx - deviceFirstSample)*App().initialSampleSize(nullptr) + 0] = insertionPos;

        for (int e = transitNeighborIdx; e < numTransitEdges; e += stepSize) {
          EdgePos_t pos = sampleNeighborhoodPos[(sampleIdx - deviceFirstSample)] + e + insertionPos%2;
          collectiveNeighborhoodCSRCols[pos] = transitEdges[e];
        }
      }
    } else if (tpMode == CollectiveNeighborhoodSize) {
      //Compute size of collective neighborhood for each sample.
      ::atomicAdd(&sampleNeighborhoodSizes[(sampleIdx - deviceFirstSample)], numTransitEdges);
    }
  }

  __syncwarp();
  if (tpMode == NextFuncExecution) {
    EdgePos_t insertionPos = 0;

    //TODO: templatize over hasExplicitTransits()
    if (step != App().steps() - 1) {
      //No need to store at last step
      if (App().hasExplicitTransits()) {
        VertexID_t newTransit = App().stepTransits(step+1, sampleIdx, samples[(sampleIdx - deviceFirstSample)], threadId%numTransits, randState);
        samplesToTransitValues[threadId] = newTransit != -1 ? newTransit : invalidVertex;
      } else {
        samplesToTransitValues[threadId] = neighbor != -1 ? neighbor : invalidVertex;
      }
      samplesToTransitKeys[threadId] = sampleIdx;
    }

    if (numberOfTransits<App>(step) > 1 && isValidSampledVertex(neighbor, invalidVertex)) {   
      //insertionPos = finalSampleSizeTillPreviousStep + transitNeighborIdx; //
      if (step == 0) {
        insertionPos = transitNeighborIdx;
      } else {
        size_t finalSampleSizeTillPreviousStep = 0;
        size_t neighborsToSampleAtStep = 1;
        for (int _s = 0; _s < step; _s++) {
          neighborsToSampleAtStep *= App().stepSize(_s);
          finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
        }
        insertionPos = finalSampleSizeTillPreviousStep + utils::atomicAdd(&sampleInsertionPositions[(sampleIdx - deviceFirstSample)], 1);
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
        finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos - 1] = transit;
        if (isValidSampledVertex(neighbor, invalidVertex)) finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
      } else if (step == App().steps() - 1 && isValidSampledVertex(neighbor, invalidVertex)) {
        finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
      }
    }
    else {
      // if (STORE_TRANSIT_INDEX) {
      //   //Store Index of transit in each sample's output
      //   if (step == 0) {
      //     transitIndexInSample[threadId] = insertionPos;
      //   } else if (step != App().steps() - 1) {
      //     transitIndexInSample[threadId] = prevTransitIndexInSample[];
      //   }
      // }
      if (isValidSampledVertex(neighbor, invalidVertex))
        finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
    }
  }
}

template<class SampleType, typename App, int THREADS, bool COALESCE_CURAND_LOAD>
__global__ void identityKernel(const int step, GPUCSRPartition graph, const VertexID_t deviceFirstSample, const VertexID_t invalidVertex,
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
  
  for (; threadId < transitToSamplesSize; threadId += gridDim.x * blockDim.x) {
    //__shared__ VertexID newNeigbhors[N_THREADS];
    EdgePos_t transitIdx;
    EdgePos_t transitNeighborIdx;
    VertexID_t transit;
    int kernelTy;

    if (threadId >= transitToSamplesSize)
      continue;
    
    int subWarpSize = subWarpSizeAtStep<App>(step);
    transitIdx = threadId/subWarpSize;
    transitNeighborIdx = threadId % subWarpSize;
    if (transitNeighborIdx == 0) {
      transit = transitToSamplesKeys[transitIdx];
      kernelTy = kernelTypeForTransit[transit];
    }

    transit = __shfl_sync(FULL_WARP_MASK, transit, 0, subWarpSize);
    kernelTy = __shfl_sync(FULL_WARP_MASK, kernelTy, 0, subWarpSize);

    if (transitNeighborIdx >= App().stepSize(step)) {
      continue;
    }

    if ((useGridKernel && kernelTy == TransitKernelTypes::GridKernel && numberOfTransits<App>(step) > 1) || 
        (useSubWarpKernel && kernelTy == TransitKernelTypes::SubWarpKernel) || 
        (useThreadBlockKernel && kernelTy == TransitKernelTypes::ThreadBlockKernel)) {
      continue;
    }

    CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];
    VertexID_t sampleIdx = -1;
    
    if (transitNeighborIdx == 0) {
      sampleIdx = transitToSamplesValues[transitIdx];
    }

    sampleIdx = __shfl_sync(FULL_WARP_MASK, sampleIdx, 0, subWarpSize);
    assert(sampleIdx < NumSamples);
    VertexID_t neighbor = invalidVertex;

    if (transit != invalidVertex) {
      // if (graph.device_csr->has_vertex(transit) == false)
      //   printf("transit %d\n", transit);
      EdgePos_t numTransitEdges = csr->get_n_edges_for_vertex(transit);
      
      if (numTransitEdges != 0) {
        const CSR::Edge* transitEdges = csr->get_edges(transit);
        const float* transitEdgeWeights = csr->get_weights(transit);
        const float maxWeight = csr->get_max_weight(transit);

        neighbor = App().next(step, csr, &transit, sampleIdx, &samples[sampleIdx - deviceFirstSample], maxWeight, transitEdges, transitEdgeWeights, 
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
  
  if (step != App().steps() - 1) {
    //No need to store at last step
    if (App().hasExplicitTransits()) {
      VertexID_t newTransit = App().stepTransits(step + 1, sampleIdx, samples[sampleIdx - deviceFirstSample], transitIdx, &localRandState);
      samplesToTransitValues[threadId] = newTransit != -1 ? newTransit : invalidVertex;
    } else {
      samplesToTransitValues[threadId] = neighbor != -1 ? neighbor : invalidVertex;;
    }
    samplesToTransitKeys[threadId] = sampleIdx;
  }

  //FIXME: in deepwalk if there is an invalid vertex at step k, it will not store the
  //transits of step k -1 due to coalescing the stores. 
  EdgePos_t finalSampleSizeTillPreviousStep = 0;
  EdgePos_t neighborsToSampleAtStep = 1;
  EdgePos_t insertionPos = 0; 
  if (numberOfTransits<App>(step) > 1 && isValidSampledVertex(neighbor, invalidVertex)) {    
    if (step == 0) {
      insertionPos = transitNeighborIdx;
    } else {
      EdgePos_t numTransits = numberOfTransits<App>(step);
      for (int _s = 0; _s < step; _s++) {
        neighborsToSampleAtStep *= App().stepSize(_s);
        finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
      }
      EdgePos_t insertionStartPosForTransit = 0;
      if (threadIdx.x % subWarpSize == 0) {
          insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx - deviceFirstSample], App().stepSize(step));
      }
      insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
      insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
    }
  } else {
    insertionPos = step;
  }

  // if (insertionPos < finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
  // }
  // if (sampleIdx == 1747842) {
  //   printf("490: insertionPos %d finalSampleSize %ld sampleIdx %d transit %d\n", insertionPos, finalSampleSize, sampleIdx, transit);
  // }
  assert(insertionPos < finalSampleSize);

  if (combineTwoSampleStores && numberOfTransits<App>(step) == 1) {
    //TODO: We can combine stores even when numberOfTransits<App>(step) > 1
    if (step % 2 == 1) {
      int2 *ptr = (int2*)&finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos - 1];
      int2 res;
      res.x = transit;
      res.y = neighbor;
      *ptr = res;
      //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
    } else if (step == App().steps() - 1) {
      finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
    }
  } else {
    if (isValidSampledVertex(neighbor, invalidVertex))
      finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
  }
  // if (sample == 100) {
  //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
  // }
  //TODO: We do not need atomic instead store indices of transit in another array,
  //wich can be accessed based on sample and transitIdx.
  }
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
    // CachedArray<CSR::Edge, CACHE_SIZE, ONDEMAND_CACHING, STATIC_CACHE_SIZE> cachedEdges = {shMem.glTransitEdges, edgesInShMem};
    if (numTransitEdges != 0) {
      assert(false);//TODO: Disable for now.
      // neighbor = App().template nextCached<SampleType, CACHE_SIZE_PER_SUBWARP, CACHE_EDGES, CACHE_WEIGHTS, COALESCE_GL_LOADS, false, 0>(step, transit, sampleIdx, &samples[sampleIdx], maxWeight, 
      //                                                                               cachedEdges, glTransitEdgeWeights, 
      //                                                                               numTransitEdges, transitNeighborIdx, &localRandState,
      //                                                                               edgeWeightsInShMem,
      //                                                                               globalLoadBV);
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

template<int CACHE_SIZE, bool COALESCE_GL_LOADS, typename T, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE>
__device__ inline VertexID_t cacheAndGet(EdgePos_t id, const T* transitEdges, T* cachedEdges, bool* globalLoadBV)
{
  if (id >= CACHE_SIZE) {
    return transitEdges[id];
  }
  
  VertexID_t e;

  // if (false && COALESCE_GL_LOADS) {
  //   e = cachedEdges[id];
  //   if (e == -1)
  //     cachedEdges[id] = -2;

  //   int subWarpThreadIdx = threadIdx.x % LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
  //   //int subWarp = threadIdx.x / LoadBalancing::LoadBalancingThreshold::SubWarpLevel;
  //   for (int i = subWarpThreadIdx; i < CACHE_SIZE; i += LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
  //     if (cachedEdges[i] == -2) {
  //       cachedEdges[i] = transitEdges[i];
  //     }
  //   }
    
  //   e = cachedEdges[id];
  // } else 
  {
    e = cachedEdges[id];
    if (id < STATIC_CACHE_SIZE)
      return e;

    if (ONDEMAND_CACHING and e == -1) {
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
      if (numEdgesInShMem > 0)
      assert(false);//TODO: Disabled for now.
        // neighbor = App().template nextCached<SampleType, CACHE_SIZE, CACHE_EDGES, CACHE_WEIGHTS, 0, false, 0>(step, transit, sampleIdx, &samples[sampleIdx], maxWeight, 
        //                                                       glTransitEdges, glTransitEdgeWeights, 
        //                                                       numEdgesInShMem, transitNeighborIdx, &localRandState,
        //                                                       edgesInShMem, edgeWeightsInShMem,
        //                                                       &globalLoadBV[0]);
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

template<class SampleType, typename App, int THREADS, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool COALESCE_GL_LOADS, int TRANSITS_PER_THREAD, 
bool COALESCE_CURAND_LOAD, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE, int SUB_WARP_SIZE>
__global__ void gridKernel(const int step, GPUCSRPartition graph, const VertexID_t deviceFirstSample, 
                           const VertexID_t invalidVertex,
                           const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                           const size_t transitToSamplesSize, SampleType* samples, const size_t NumSamples,
                           VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                           VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                           curandState* randStates, const int* kernelTypeForTransit, const VertexID_t* gridKernelTBPositions, 
                           const EdgePos_t gridKernelTBPositionsNum, int totalThreadBlocks)
{
  #define EDGE_CACHE_SIZE (CACHE_EDGES ? CACHE_SIZE*sizeof(CSR::Edge) : 0)
  #define WEIGHT_CACHE_SIZE (CACHE_WEIGHTS ? CACHE_SIZE*sizeof(float) : 0)
  #define CURAND_SHMEM_SIZE (sizeof(curandState)*THREADS)
  // #define COALESCE_GL_LOADS_SHMEM_SIZE ()

  union unionShMem {
    struct {
      unsigned char edgeAndWeightCache[EDGE_CACHE_SIZE+WEIGHT_CACHE_SIZE];
      bool invalidateCache;
      VertexID_t transitForTB;
      // VertexID_t numEdgesInShMem[TRANSITS_PER_THREAD];
      // CSR::Edge* glTransitEdges[TRANSITS_PER_THREAD];
      // float* glTransitEdgeWeights[TRANSITS_PER_THREAD];
      // float maxWeight[TRANSITS_PER_THREAD];
      EdgePos_t mapStartPos[TRANSITS_PER_THREAD];
      EdgePos_t subWarpTransits[TRANSITS_PER_THREAD][THREADS/SUB_WARP_SIZE];
      EdgePos_t subWarpSampleIdx[TRANSITS_PER_THREAD][THREADS/SUB_WARP_SIZE];
      unsigned char transitVertices[TRANSITS_PER_THREAD*sizeof(CSR::Vertex)];
    };
    unsigned char shMemAlloc[sizeof(curandState)*THREADS];
  };
  __shared__ unionShMem shMem;
  
  //__shared__ bool globalLoadBV[COALESCE_GL_LOADS ? CACHE_SIZE : 1];
  
  CSR::Edge* edgesInShMem = CACHE_EDGES ? (CSR::Edge*)&shMem.edgeAndWeightCache[0] : nullptr;
  float* edgeWeightsInShMem = CACHE_WEIGHTS ? (float*)&shMem.edgeAndWeightCache[EDGE_CACHE_SIZE] : nullptr;
  
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  
  curandState* curandSrcPtr;

  const int subWarpSize = SUB_WARP_SIZE;

  if (COALESCE_CURAND_LOAD) {
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shMem.shMemAlloc[0];

    int* randStatesAsInts = (int*)randStates;
    
    //Load curand only for the number of threads that are going to do sampling in this warp
    for (int i = threadIdx.x; i < intsInRandState*(blockDim.x/subWarpSize)*App().stepSize(step); i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    if (threadIdx.x % subWarpSize < App().stepSize(step)) {
      //Load curand only for the threads that are going to do sampling.
      int ld = threadIdx.x - (threadIdx.x/subWarpSize)*(subWarpSize-App().stepSize(step));
      curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
    }
  } else {
    curandSrcPtr = &randStates[threadId];
  }

  curandState localRandState = (threadIdx.x % subWarpSize < App().stepSize(step))? *curandSrcPtr: curandState();
  //curand_init(threadId, 0,0, &localRandState);

  //__shared__ VertexID newNeigbhors[N_THREADS];
  //if (threadIdx.x == 0) printf("blockIdx.x %d\n", blockIdx.x);
  //shRandStates[threadIdx.x] = randStates[threadId];  
  //__syncthreads();
  
  CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];
  for (int fullBlockIdx = blockIdx.x; fullBlockIdx < totalThreadBlocks; fullBlockIdx += gridDim.x) {
    EdgePos_t transitIdx = 0;
    if (threadIdx.x < TRANSITS_PER_THREAD) {
      if (TRANSITS_PER_THREAD * (fullBlockIdx) + threadIdx.x < gridKernelTBPositionsNum) {
        shMem.mapStartPos[threadIdx.x] = gridKernelTBPositions[TRANSITS_PER_THREAD * fullBlockIdx + threadIdx.x];
      }
    }
    __syncthreads();
    if (threadIdx.x < THREADS/SUB_WARP_SIZE * TRANSITS_PER_THREAD) {
      //Coalesce loads of transits per sub-warp by loading transits for all sub-warps in one warp.
      // Assign THREADS/SUB_WARP_SIZE threads to each Transit in TRANSITS_PER_THREAD
      assert ((THREADS/SUB_WARP_SIZE * TRANSITS_PER_THREAD) < blockDim.x);
      int transitI = threadIdx.x / (THREADS/SUB_WARP_SIZE);// * TRANSITS_PER_THREAD);
      transitIdx = shMem.mapStartPos[transitI] + threadIdx.x % (THREADS/SUB_WARP_SIZE);
      //TODO: Specialize this for subWarpSizez = 1.
      VertexID_t transit = invalidVertex;
      if (subWarpSize == 1) {
        transit = transitToSamplesKeys[transitIdx];
      } else {
        transit = transitToSamplesKeys[transitIdx];
      }

      shMem.subWarpTransits[transitI][threadIdx.x% (THREADS/SUB_WARP_SIZE)] = transit;
      shMem.subWarpSampleIdx[transitI][threadIdx.x% (THREADS/SUB_WARP_SIZE)] = transitToSamplesValues[transitIdx];
    }
    __syncthreads();
    const int threadsToLoadTransit = sizeof(CSR::Vertex)/sizeof(int);
    if (threadIdx.x < threadsToLoadTransit * TRANSITS_PER_THREAD) {
      //Load Transit Vertex Information in a Coalesced manner
      int transitI = threadIdx.x / threadsToLoadTransit;
      VertexID transit = shMem.subWarpTransits[transitI][0];
      const CSR::Vertex* transitVertex = csr->get_vertices() + transit;
      int tid = threadIdx.x % threadsToLoadTransit;
      int data = ((const int*)transitVertex)[tid];
      *(((int*)&shMem.transitVertices[transitI * sizeof(CSR::Vertex)]) + tid) = data;
    }

    for (int transitI = 0; transitI < TRANSITS_PER_THREAD; transitI++) {
      if (TRANSITS_PER_THREAD * (fullBlockIdx) + transitI >= gridKernelTBPositionsNum) {
        continue;
      }
      __syncthreads();
      VertexID_t transit = shMem.subWarpTransits[transitI][threadIdx.x/subWarpSize];
      CSR::Vertex* shMemTransitVertex = ((CSR::Vertex*)(&shMem.transitVertices[transitI * sizeof(CSR::Vertex)]));
      EdgePos_t numEdgesInShMem = shMemTransitVertex->num_edges();
      const CSR::Edge* glTransitEdges = (CSR::Edge*)csr->get_edges() + shMemTransitVertex->get_start_edge_idx();
      const float* glTransitEdgeWeights = (float*)(CSR::Edge*)csr->get_weights() + shMemTransitVertex->get_start_edge_idx();
      float maxWeight = shMemTransitVertex->get_max_weight();

      if (threadIdx.x == 0) {
        shMem.invalidateCache = shMem.transitForTB != transit || transitI == 0;
        shMem.transitForTB = transit;
      }

      __syncthreads();
      if (CACHE_EDGES && shMem.invalidateCache) {
        for (int i = threadIdx.x; i < min(CACHE_SIZE, numEdgesInShMem); i += blockDim.x) {
          if (ONDEMAND_CACHING) {
            if (i < STATIC_CACHE_SIZE)
              edgesInShMem[i] = glTransitEdges[i];
            else 
              edgesInShMem[i] = -1;
          } else {
            edgesInShMem[i] = glTransitEdges[i];
          }
        }
      }
  
      if (CACHE_WEIGHTS && shMem.invalidateCache) {
        for (int i = threadIdx.x; i < min(CACHE_SIZE, numEdgesInShMem); i += blockDim.x) {
          edgeWeightsInShMem[i] = (ONDEMAND_CACHING) ? -1 : glTransitEdgeWeights[i];
        }
      }

      __syncthreads();

      if (transit == shMem.transitForTB) {
        //A thread will run next only when it's transit is same as transit of the threadblock.
        transitIdx = shMem.mapStartPos[transitI] + threadIdx.x/subWarpSize; //threadId/stepSize(step);
        VertexID_t transitNeighborIdx = threadIdx.x % subWarpSize;
        VertexID_t sampleIdx = shMem.subWarpSampleIdx[transitI][threadIdx.x/subWarpSize];;

        // if (threadIdx.x % subWarpSize == 0) {
        //   sampleIdx = transitToSamplesValues[transitIdx];
        // }
        
        // sampleIdx = __shfl_sync(FULL_WARP_MASK, sampleIdx, 0, subWarpSize);

        if (transitNeighborIdx >= App().stepSize(step)) 
          continue;
        // if (threadIdx.x == 0 && kernelTypeForTransit[transit] != TransitKernelTypes::GridKernel) {
        //   printf("transit %d transitIdx %d gridDim.x %d\n", transit, transitIdx, gridDim.x);
        // }
        // assert (kernelTypeForTransit[transit] == TransitKernelTypes::GridKernel);
        
        //TODO: Set this based on the input template parameters.
        typedef CachedArray<CSR::Edge, CACHE_SIZE, ONDEMAND_CACHING, STATIC_CACHE_SIZE> CachedEdges;
        typedef CachedArray<float, CACHE_SIZE, ONDEMAND_CACHING, STATIC_CACHE_SIZE> CachedWeights;

        CachedEdges cachedEdges = {glTransitEdges, edgesInShMem};
        CachedWeights cachedWeights = {glTransitEdgeWeights, edgeWeightsInShMem};

        VertexID_t neighbor = invalidVertex;
        // if (graph.device_csr->has_vertex(transit) == false)
        //   printf("transit %d\n", transit);
        if (numEdgesInShMem > 0)
          neighbor = App().template next<SampleType, CachedEdges, CachedWeights>(step, csr, &transit, sampleIdx, &samples[sampleIdx-deviceFirstSample], maxWeight, 
                                                                cachedEdges, cachedWeights,
                                                                numEdgesInShMem, transitNeighborIdx, &localRandState);
        // //EdgePos_t totalSizeOfSample = stepSizeAtStep<App>(step - 1);
        // if ((transit == 612657 || transit == 348930) && sampleIdx == 17175) {
        //   printf("transit %d fullBlockIdx  %d sampleIdx %d neighbor %d\n", transit, fullBlockIdx, sampleIdx, neighbor);
        // }

        
        if (step != App().steps() - 1) {
          //No need to store at last step
          samplesToTransitKeys[transitIdx] = sampleIdx; //TODO: Update this for khop to transitIdx + transitNeighborIdx
          if (App().hasExplicitTransits()) {
            VertexID_t newTransit = App().stepTransits(step, sampleIdx, samples[sampleIdx-deviceFirstSample], transitIdx, &localRandState);
            samplesToTransitValues[transitIdx] = newTransit != -1 ? newTransit : invalidVertex;
          } else {
            samplesToTransitValues[transitIdx] = neighbor != -1 ? neighbor : invalidVertex;
          }
        }

        if (isValidSampledVertex(neighbor, invalidVertex)) {
          EdgePos_t insertionPos = transitNeighborIdx; 
          if (numberOfTransits<App>(step) > 1) {
            if (step == 0) {
              insertionPos = transitNeighborIdx;
            } else {
              EdgePos_t numTransits = numberOfTransits<App>(step);
              EdgePos_t finalSampleSizeTillPreviousStep = 0;
              EdgePos_t neighborsToSampleAtStep = 1;
              for (int _s = 0; _s < step; _s++) {
                neighborsToSampleAtStep *= App().stepSize(_s);
                finalSampleSizeTillPreviousStep += neighborsToSampleAtStep;
              }
              EdgePos_t insertionStartPosForTransit = 0;
              if (threadIdx.x % subWarpSize == 0) {
                insertionStartPosForTransit = utils::atomicAdd(&sampleInsertionPositions[sampleIdx-deviceFirstSample], App().stepSize(step));
              }
              insertionStartPosForTransit = __shfl_sync(FULL_WARP_MASK, insertionStartPosForTransit, 0, subWarpSize);
              insertionPos = finalSampleSizeTillPreviousStep + insertionStartPosForTransit + transitNeighborIdx;
            }
          } else {
            insertionPos = step;
          }

          // if (insertionPos < finalSampleSize) {
          //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
          // }
          // if (sampleIdx == 1747842)
          //     printf("insertionPos %d finalSampleSize %ld sample %d threadIdx.x %d mapStartPos %d blockIdx.x %d transit %d\n", 
          //            insertionPos, finalSampleSize, sampleIdx, threadIdx.x, mapStartPos, blockIdx.x, transit);

          // if (insertionPos >= finalSampleSize) {
            
          //   return;
          // }
          assert(insertionPos < finalSampleSize);

          if (combineTwoSampleStores && numberOfTransits<App>(step) == 1) {
            if (step % 2 == 1) {
              int2 *ptr = (int2*)&finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos - 1];
              int2 res;
              res.x = transit;
              res.y = neighbor;
              *ptr = res;
              //finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
            } else if (step == App().steps() - 1) {
              finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
            }
          } else {
            finalSamples[(sampleIdx-deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
          }
          // if (sample == 100) {
          //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
          // }
          //TODO: We do not need atomic instead store indices of transit in another array,
          //wich can be accessed based on sample and transitIdx.
        }
      }
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
    VertexID_t transit;
    if (step == 0) 
      transit = initialSamples[sampleIdx*App().initialSampleSize(nullptr) + transitIdx];
    else 
      transit = finalSamples[sampleIdx*App().initialSampleSize(nullptr) + transitIdx];
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
  if (threadId >= totalThreads)
    return;
  EdgePos_t numTransits = numberOfTransits<App>(step - 1);
  EdgePos_t sampleIdx = threadId/numTransits;
  if (sampleIdx > NumSamples)
    return;

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

/**
  sampleParallelKernel()    - Sample Parallel Kernel
  Arguments:
  @SampleType               : class of Sample
  @App                      : App class
  @THREADS                  : Number of threads in a thread block
  @step                     : Current executing step
  @graph                    : CSR Partition stored in GPU 
  @invalidVertex            : Value of invalid vertex
  @threadsExecuted          : Number of threads already executed
  @currExecutionThreads     : Number of grid threads in current execution
  @totalThreads             : Total number of threads this kernel will be invoked with
  @initialSamples           : Array of initial contents of all samples
  @samples                  : Array of all samples
  @NumSamples               : Number of samples
  @finalSamples             : Sampled vertices for all samples
  @finalSampleSize          : Final number of vertices in sample 
  @explicitTransits         : Array of explicit transits
  @sampleInsertionPositions : Insertion Position for sampled vertex in a Sample
  @randStates               : curand states

  Sample Parallel Kernel doing sampling on GPU using a sample parallel paradigm. 
  */
  template<class SampleType, typename App, int THREADS, bool WriteSampleToTransitMap>
  __global__ void sampleParallelKernel(const int step, GPUCSRPartition graph, 
                                       const size_t deviceFirstSample,
                                       const VertexID_t invalidVertex,
                                       const size_t totalThreads,
                                       VertexID_t* initialSamples,
                                       SampleType* samples,
                                       const size_t NumSamples,
                                       VertexID_t* finalSamples,
                                       const size_t finalSampleSize, 
                                       VertexID_t* samplesToTransitMapKeys, 
                                       VertexID_t* samplesToTransitMapValues,
                                       EdgePos_t* sampleInsertionPositions,
                                       curandState* randStates)
  {
    __shared__ unsigned char shMemCuRand[sizeof(curandState)*THREADS];
  
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  
    graph.device_csr = (CSRPartition*)&csrPartitionBuff[0];
    
    curandState* curandSrcPtr;
    bool COALESCE_CURAND_LOAD = true;
    if (COALESCE_CURAND_LOAD) {
      //Load curand states efficiently in registers
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
  
    for (; threadId < totalThreads; threadId += gridDim.x * blockDim.x) {
      if (threadId >= totalThreads)
        return;
  
      EdgePos_t numTransits = numberOfTransits<App>(step);
      EdgePos_t numTransitsInPrevStep = numberOfTransits<App>(step - 1);
      VertexID_t sampleIdx = threadId / numTransits + deviceFirstSample; //(threadId / numTransits) ranges from [0, # of samples for this device]
  
      VertexID_t* transits = nullptr;
      VertexID_t singleTransit = 0;
      EdgePos_t numTransitsInNeghbrhood = 0;
      //TODO: Template this kernel based on the sampling type
      if (App().samplingType() == CollectiveNeighborhood) {
        assert(!App().hasExplicitTransits());
        numTransitsInNeghbrhood = numberOfTransits<App>(step);
        if (step == 0) {
          transits = &initialSamples[(sampleIdx - deviceFirstSample)*App().initialSampleSize(nullptr)];
        } else {
          size_t verticesAddTillPreviousStep = stepSizeAtStep<App>(step - 2);
          //printf("verticesAddTillPreviousStep %ld\n", verticesAddTillPreviousStep); 
  
          transits = &finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + verticesAddTillPreviousStep];
        }
      } else {
        if (step == 0) {
          EdgePos_t transitIdx = threadId % App().initialSampleSize(nullptr);
          singleTransit = initialSamples[(sampleIdx - deviceFirstSample)*App().initialSampleSize(nullptr) + transitIdx];
        } else if (App().hasExplicitTransits()) {
          singleTransit = samplesToTransitMapValues[(sampleIdx - deviceFirstSample)*numTransitsInPrevStep + (threadId % numTransits) / numTransitsInPrevStep];
        } else {
          singleTransit = finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + (step - 1) * numTransits + (threadId % numTransits) % numTransitsInPrevStep];
        }
  
        numTransitsInNeghbrhood = 1;
        transits = &singleTransit;
      }
      
      VertexID_t neighbor = invalidVertex;
      VertexID_t neighbrID = threadId % App().stepSize(step) ;//(threadId % numTransits) % numTransitsInPrevStep;
      VertexID_t transitID = (threadId % numTransits) / App().stepSize(step);

      if (*transits != invalidVertex and *transits != -1) {
        EdgePos_t numTransitEdges = 0;
  
        for (int i = 0; i < numTransitsInNeghbrhood; i++) {
          if (!graph.device_csr->has_vertex(transits[i])) {
            printf("transits[%d] %d step %d \n", i, transits[i]);
            return;
          }
          numTransitEdges += graph.device_csr->get_n_edges_for_vertex(transits[i]);
        }
        
        if (numTransitEdges != 0) {
          const CSR::Edge* transitEdges = (App().samplingType() == CollectiveNeighborhood) ? nullptr : graph.device_csr->get_edges(*transits);
          const float* transitEdgeWeights = (App().samplingType() == CollectiveNeighborhood) ? nullptr : graph.device_csr->get_weights(*transits);
          const float maxWeight = (App().samplingType() == CollectiveNeighborhood) ? 0.0 : graph.device_csr->get_max_weight(*transits);
  
          neighbor = App().template next<SampleType, const CSR::Edge*, const float*> (step, graph.device_csr, transits, sampleIdx, &samples[(sampleIdx - deviceFirstSample)], maxWeight, 
            transitEdges, transitEdgeWeights, numTransitEdges, neighbrID, &localRandState);
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
  
      if (WriteSampleToTransitMap) {
        samplesToTransitMapKeys[threadId] = sampleIdx;
        samplesToTransitMapValues[threadId] = neighbor;
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
      
      // if (*transits == invalidVertex) {
      //   assert (neighbor == invalidVertex);
      // }
      if (App().outputFormat() == AdjacencyMatrix && App().samplingType() == CollectiveNeighborhood) {
        finalSamples[sampleIdx*finalSampleSize + stepSizeAtStep<App>(step - 1) + neighbrID] = neighbor;
      } else if (App().outputFormat() == SampledVertices && App().samplingType() == IndividualNeighborhood) {
        if (numberOfTransits<App>(step) > 1) {    
          insertionPos = finalSampleSizeTillPreviousStep + (threadId % numTransits);//utils::atomicAdd(&sampleInsertionPositions[sampleIdx], 1);//
        } else {
          insertionPos = step;
        }
  
        finalSamples[(sampleIdx - deviceFirstSample)*finalSampleSize + insertionPos] = neighbor;
      }
      // if (sample == 100) {
      //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
      // }
      //TODO: We do not need atomic instead store indices of transit in another array,
      //wich can be accessed based on sample and transitIdx.
    }
  
    //Write back the updated curand states
    if (COALESCE_CURAND_LOAD) {
      const int intsInRandState = sizeof(curandState)/sizeof(int);
      curandState* shStateBuff = (curandState*)&shMemCuRand[0];
      shStateBuff[threadIdx.x] = localRandState;
      __syncthreads();
  
      int* shStateBuffAsInts = (int*)&shStateBuff[0];
      int* randStatesAsInts = (int*)randStates;
    
      for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
        randStatesAsInts[i + blockDim.x*blockIdx.x] = shStateBuffAsInts[i];
      }
    } else {
      *curandSrcPtr = localRandState;
    }
  }

template<typename App, int TB_THREADS>
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
  int subWarpSize = subWarpSizeAtStep<App>(step);

  int kernelType = -1;
  EdgePos_t numThreadGroups = 0;
  if (useGridKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::GridLevel) {    
    kernelType = TransitKernelTypes::GridKernel;
  } else if (useThreadBlockKernel && trCount * subWarpSize > LoadBalancing::LoadBalancingThreshold::BlockLevel) {
    kernelType = TransitKernelTypes::ThreadBlockKernel;
    // numThreadGroups = 0;
    // threadToTransitPos[threadIdx.x] = 0;
  } else if (useSubWarpKernel && trCount * subWarpSize >= LoadBalancing::LoadBalancingThreshold::SubWarpLevel) {
    kernelType = TransitKernelTypes::SubWarpKernel;
    
    // numThreadGroups = 0;
    // threadToTransitPos[threadIdx.x] = 0;
  } else {
    kernelType = TransitKernelTypes::IdentityKernel;
    // numThreadGroups = 0;
    // threadToTransitPos[threadIdx.x] = 0;
  }
  
  if (threadId < uniqueTransitCountsNum && kernelType != IdentityKernel && transit != invalidVertex) {
    kernelTypeForTransit[transit] = kernelType;
  } 

  if (kernelType == IdentityKernel && transit != invalidVertex && trCount !=-1) {
    *identityKernelTransitsNum = 1;
  }

  __syncthreads();

  for (int kTy = 1; kTy < TransitKernelTypes::SubWarpKernel + 1; kTy++) {
    EdgePos_t* glKernelTransitsNum, *glKernelTransits;
    const int threadGroupSize = (kTy == TransitKernelTypes::GridKernel) ? LoadBalancing::LoadBalancingThreshold::GridLevel/subWarpSize : 
                                (kTy == TransitKernelTypes::ThreadBlockKernel ? LoadBalancing::LoadBalancingThreshold::BlockLevel : 
                                (kTy == TransitKernelTypes::SubWarpKernel ? LoadBalancing::LoadBalancingThreshold::SubWarpLevel : -1));

    if (kTy == TransitKernelTypes::GridKernel && useGridKernel) {
      if (kernelType == TransitKernelTypes::GridKernel) {
        numThreadGroups = DIVUP(trCount, LoadBalancing::LoadBalancingThreshold::GridLevel/subWarpSize);
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
        numThreadGroups = DIVUP(trCount * subWarpSize, LoadBalancing::LoadBalancingThreshold::BlockLevel);
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
        numThreadGroups = DIVUP(trCount * subWarpSize, LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
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
          //   printf("blockIdx.x %d shGridKernelTransits[tbs] %d tbs %d tgIter %d startPos %d pos %d expectedTr %d threadTr %d kernelTy %d\n", blockIdx.x, shGridKernelTransits[tbs], tbs, tgIter, startPos, pos, transitToSamplesKeys[pos], transit, kTy);
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

__global__ void invalidVertexStartPos(int step, VertexID_t* transitToSamplesKeys, size_t totalTransits, 
                                      const VertexID_t invalidVertex, EdgePos_t* outputStartPos)
{
  int threadId = threadIdx.x + blockIdx.x*blockDim.x;

  if (threadId >= totalTransits) {
    return;
  }

  //If first transit is invalid.
  if (threadId == 0) {
    if (transitToSamplesKeys[0] == invalidVertex) {
      *outputStartPos = 0;
    }
    // printf("outputStartPos %d\n", *outputStartPos);
    return;
  }

  //TODO: Optimize this using overlaped tilling
  if (transitToSamplesKeys[threadId - 1] != invalidVertex && 
      transitToSamplesKeys[threadId] == invalidVertex)
  {
    *outputStartPos = threadId;
    return;
      // printf("outputStartPos %d\n", *outputStartPos);
  }

  //If no transit is invalid 
  // if (threadId == totalTransits - 1) {
  //   printf("1666: threadIdx.x %d v %d invalidVertex %d\n", threadId, transitToSamplesKeys[threadId], invalidVertex);
  //   *outputStartPos = totalTransits - 1;
  // }
}

__global__ void init_curand_states(curandState* states, size_t num_states)
{
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if (thread_id < num_states)
    curand_init(thread_id, threadIdx.x, 0, &states[thread_id]);
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
bool allocNextDoorDataOnGPU(CSR* csr, NextDoorData<SampleType, App>& data)
{
  char* deviceList;
  if ((deviceList = getenv("CUDA_DEVICES")) != nullptr) {
    std::string deviceListStr = deviceList;

    std::stringstream ss(deviceListStr);
    if (ss.peek() == '[')
        ss.ignore();
    for (int i; ss >> i;) {
      data.devices.push_back(i);    
      if (ss.peek() == ',')
        ss.ignore();
    }
    if (ss.peek() == ']')
      ss.ignore();
  } else {
    data.devices = {0};
  }

  std::cout << "Using GPUs: [";
  for (auto d : data.devices) {
    std::cout << d << ",";
  }
  std::cout << "]" << std::endl;

  int maxV = 0;
  // printf("App().numSamples(csr) %d\n", App().numSamples(csr));
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
  //Allocate storage for final samples on GPU
  data.hFinalSamples = std::vector<VertexID_t>(finalSampleSize*numSamples);
  data.dSamplesToTransitMapKeys = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dSamplesToTransitMapValues = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dTransitToSampleMapKeys = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dTransitToSampleMapValues = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dSampleInsertionPositions = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dNeighborhoodSizes = std::vector<EdgePos_t*>(data.devices.size(), nullptr);
  data.dCurandStates = std::vector<curandState*>(data.devices.size(), nullptr);
  data.maxThreadsPerKernel = std::vector<size_t>(data.devices.size(), 0);
  data.dFinalSamples = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dInitialSamples = std::vector<VertexID_t*>(data.devices.size(), nullptr);
  data.dOutputSamples = std::vector<SampleType*>(data.devices.size(), nullptr);
  const size_t numDevices = data.devices.size();
  for(auto deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++) {
    auto device = data.devices[deviceIdx];
    //Per Device Allocation
    CHK_CU(cudaSetDevice(device));
    
    const size_t perDeviceNumSamples = PartDivisionSize(numSamples, deviceIdx, numDevices);
    const size_t deviceSampleStartPtr = PartStartPointer(numSamples, deviceIdx, numDevices);

    //Allocate storage and copy initial samples on GPU
    size_t partDivisionSize = App().initialSampleSize(csr)*perDeviceNumSamples;
    size_t partStartPtr = App().initialSampleSize(csr)*deviceSampleStartPtr;
    CHK_CU(cudaMalloc(&data.dInitialSamples[deviceIdx], sizeof(VertexID_t)*partDivisionSize));
    CHK_CU(cudaMemcpy(data.dInitialSamples[deviceIdx], &data.initialContents[0] + partStartPtr, 
                      sizeof(VertexID_t)*partDivisionSize, cudaMemcpyHostToDevice));

    //Allocate storage for samples on GPU
    if (sizeof(SampleType) > 0) {
      CHK_CU(cudaMalloc(&data.dOutputSamples[deviceIdx], sizeof(SampleType)*perDeviceNumSamples));
      CHK_CU(cudaMemcpy(data.dOutputSamples[deviceIdx], &data.samples[0] + deviceSampleStartPtr, sizeof(SampleType)*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
    }

    CHK_CU(cudaMalloc(&data.dFinalSamples[deviceIdx], sizeof(VertexID_t)*finalSampleSize*perDeviceNumSamples));
    gpu_memset(data.dFinalSamples[deviceIdx], data.INVALID_VERTEX, finalSampleSize*perDeviceNumSamples);
    
    //Samples to Transit Map
    CHK_CU(cudaMalloc(&data.dSamplesToTransitMapKeys[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    CHK_CU(cudaMalloc(&data.dSamplesToTransitMapValues[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));

    //Transit to Samples Map
    CHK_CU(cudaMalloc(&data.dTransitToSampleMapKeys[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    CHK_CU(cudaMalloc(&data.dTransitToSampleMapValues[deviceIdx], sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));

    //Same as initial values of samples for first iteration
    CHK_CU(cudaMemcpy(data.dTransitToSampleMapKeys[deviceIdx], &data.initialContents[0] + partStartPtr, sizeof(VertexID_t)*partDivisionSize, 
                      cudaMemcpyHostToDevice));
    CHK_CU(cudaMemcpy(data.dTransitToSampleMapValues[deviceIdx], &data.initialTransitToSampleValues[0] + partStartPtr, 
                      sizeof(VertexID_t)*partDivisionSize, cudaMemcpyHostToDevice));
    //Insertion positions per transit vertex for each sample
    CHK_CU(cudaMalloc(&data.dSampleInsertionPositions[deviceIdx], sizeof(EdgePos_t)*perDeviceNumSamples));

    size_t curandDataSize = maxNeighborsToSample*perDeviceNumSamples*sizeof(curandState);
    const size_t curandSizeLimit = 5L*1024L*1024L*sizeof(curandState);
    if (curandDataSize < curandSizeLimit) {
      int maxSubWarpSize = 0;
      for (int s = 0; s < App().steps(); s++) {
        maxSubWarpSize = max(maxSubWarpSize, subWarpSizeAtStep<App>(s));
      }
      //Maximum threads for a kernel should ensure that for a transit for a sample all needed
      //neighbors are sampled.
      assert(maxSubWarpSize != 0);
      data.maxThreadsPerKernel[deviceIdx] = ROUNDUP(maxNeighborsToSample*perDeviceNumSamples, maxSubWarpSize*N_THREADS);
      curandDataSize = data.maxThreadsPerKernel[deviceIdx] * sizeof(curandState);
    } else {
      data.maxThreadsPerKernel[deviceIdx] = curandSizeLimit/sizeof(curandState);
      curandDataSize = curandSizeLimit;
    }
    printf("Maximum Threads Per Kernel: %ld\n", data.maxThreadsPerKernel[deviceIdx]);
    CHK_CU(cudaMalloc(&data.dCurandStates[deviceIdx], curandDataSize));
    init_curand_states<<<thread_block_size(data.maxThreadsPerKernel[deviceIdx], 256UL), 256UL>>> (data.dCurandStates[deviceIdx], data.maxThreadsPerKernel[deviceIdx]);
    CHK_CU(cudaDeviceSynchronize());
    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      CHK_CU(cudaMalloc(&data.dNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t)*perDeviceNumSamples));
    }
  }

  return true;
}

template<class SampleType, typename App>
void freeDeviceData(NextDoorData<SampleType, App>& data) 
{
  for(auto deviceIdx = 0; deviceIdx < data.devices.size(); deviceIdx++) {
    auto device = data.devices[deviceIdx];
    CHK_CU(cudaSetDevice(device));
    CHK_CU(cudaFree(data.dSamplesToTransitMapKeys[deviceIdx]));
    CHK_CU(cudaFree(data.dSamplesToTransitMapValues[deviceIdx]));
    CHK_CU(cudaFree(data.dTransitToSampleMapKeys[deviceIdx]));
    CHK_CU(cudaFree(data.dTransitToSampleMapValues[deviceIdx]));
    CHK_CU(cudaFree(data.dSampleInsertionPositions[deviceIdx]));
    CHK_CU(cudaFree(data.dCurandStates[deviceIdx]));
    CHK_CU(cudaFree(data.dFinalSamples[deviceIdx]));
    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      CHK_CU(cudaFree(data.dNeighborhoodSizes[deviceIdx]));
    }
    CHK_CU(cudaFree(data.dInitialSamples[deviceIdx]));
    if (sizeof(SampleType) > 0) CHK_CU(cudaFree(data.dOutputSamples[deviceIdx]));
  }

  //TODO:
  CHK_CU(cudaFree(data.gpuCSRPartition.device_vertex_array));
  CHK_CU(cudaFree(data.gpuCSRPartition.device_edge_array));
  CHK_CU(cudaFree(data.gpuCSRPartition.device_weights_array));
}

template<typename App>
void printKernelTypes(int step, CSR* csr, VertexID_t* dUniqueTransits, VertexID_t* dUniqueTransitsCounts, EdgePos_t* dUniqueTransitsNumRuns)
{
  EdgePos_t* hUniqueTransitsNumRuns = GPUUtils::copyDeviceMemToHostMem(dUniqueTransitsNumRuns, 1);
  VertexID_t* hUniqueTransits = GPUUtils::copyDeviceMemToHostMem(dUniqueTransits, *hUniqueTransitsNumRuns);
  VertexID_t* hUniqueTransitsCounts = GPUUtils::copyDeviceMemToHostMem(dUniqueTransitsCounts, *hUniqueTransitsNumRuns);

  size_t identityKernelTransits = 0, identityKernelSamples = 0, maxEdgesOfIdentityTransits = 0;
  size_t subWarpLevelTransits = 0, subWarpLevelSamples = 0, maxEdgesOfSubWarpTransits = 0, subWarpTransitsWithEdgesLessThan384 = 0, subWarpTransitsWithEdgesMoreThan384 = 0, numSubWarps = 0;
  size_t threadBlockLevelTransits = 0, threadBlockLevelSamples = 0, tbVerticesWithEdgesLessThan3K = 0, tbVerticesWithEdgesMoreThan3K = 0;
  size_t gridLevelTransits = 0, gridLevelSamples = 0, gridVerticesWithEdgesLessThan3K = 0, gridVerticesWithEdgesMoreThan3K = 0,
  gridVerticesWithEdgesLessThan1K = 0, gridVerticesWithEdgesLessThan2K = 0;
  EdgePos_t maxEdgesOfGridTransits = 0;
  int subWarpSize =  subWarpSizeAtStep<App>(step);

  for (EdgePos_t tr = 0; tr < *hUniqueTransitsNumRuns; tr++) {
    // if (tr == 0) {printf("%s:%d hUniqueTransitsCounts[0] is %d\n", __FILE__, __LINE__, hUniqueTransitsCounts[tr]);}
    if (hUniqueTransitsCounts[tr] * subWarpSize < 8) {
      identityKernelTransits++;
      identityKernelSamples += hUniqueTransitsCounts[tr];
      maxEdgesOfIdentityTransits = max(maxEdgesOfIdentityTransits, (size_t)csr->n_edges_for_vertex(tr));
    } else if (hUniqueTransitsCounts[tr] * subWarpSize <= LoadBalancing::LoadBalancingThreshold::BlockLevel && 
               hUniqueTransitsCounts[tr] * subWarpSize >= 8) {
      subWarpLevelTransits++;
      subWarpLevelSamples += hUniqueTransitsCounts[tr];
      maxEdgesOfSubWarpTransits = max(maxEdgesOfSubWarpTransits, (size_t)csr->n_edges_for_vertex(tr));
      numSubWarps += DIVUP(hUniqueTransitsCounts[tr], LoadBalancing::LoadBalancingThreshold::SubWarpLevel);
      if (csr->n_edges_for_vertex(tr) <= 96) {
        subWarpTransitsWithEdgesLessThan384 += 1;
      } else {
        subWarpTransitsWithEdgesMoreThan384 += 1;
      }
    } else if (hUniqueTransitsCounts[tr] * subWarpSize > LoadBalancing::LoadBalancingThreshold::BlockLevel && 
               hUniqueTransitsCounts[tr] * subWarpSize <= LoadBalancing::LoadBalancingThreshold::GridLevel) {
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
        if (csr->n_edges_for_vertex(tr) <= 1*1024) {
          gridVerticesWithEdgesLessThan1K += 1;
        } else if (csr->n_edges_for_vertex(tr) <= 2*1024) {
          gridVerticesWithEdgesLessThan2K += 1;
        } else 
          gridVerticesWithEdgesLessThan3K += 1;
      } else {
        gridVerticesWithEdgesMoreThan3K += 1;
      }
      maxEdgesOfGridTransits = max(maxEdgesOfGridTransits, csr->n_edges_for_vertex(tr));
    }
  }

  printf("IdentityKernelTransits: %ld, IdentityKernelSamples: %ld, MaxEdgesOfIdentityTransits: %ld\n" 
         "SubWarpLevelTransits: %ld, SubWarpLevelSamples: %ld, MaxEdgesOfSubWarpTranits: %ld, VerticesWithEdges > 384: %ld, VerticesWithEdges <= 384: %ld, NumSubWarps: %ld\n"
         "ThreadBlockLevelTransits: %ld, ThreadBlockLevelSamples: %ld, VerticesWithEdges > 3K: %ld, VerticesWithEdges < 3K: %ld\n"
         "GridLevelTransits: %ld, GridLevelSamples: %ld, VerticesWithEdges > 3K: %ld, VerticesWithEdges < 3K: %ld,"
         "VerticesWithEdges < 2K: %ld, VerticesWithEdges < 1K: %ld, MaxEdgesOfTransit: %d\n", 
         identityKernelTransits, identityKernelSamples, maxEdgesOfIdentityTransits, 
         subWarpLevelTransits, subWarpLevelSamples, maxEdgesOfSubWarpTransits, 
            subWarpTransitsWithEdgesMoreThan384, subWarpTransitsWithEdgesLessThan384, numSubWarps, 
         threadBlockLevelTransits, threadBlockLevelSamples, tbVerticesWithEdgesMoreThan3K, tbVerticesWithEdgesLessThan3K,
         gridLevelTransits, gridLevelSamples, gridVerticesWithEdgesMoreThan3K, gridVerticesWithEdgesLessThan3K, gridVerticesWithEdgesLessThan2K, gridVerticesWithEdgesLessThan1K, maxEdgesOfGridTransits);

  delete hUniqueTransits;
  delete hUniqueTransitsCounts;
  delete hUniqueTransitsNumRuns;
}

template<class SampleType, typename App>
bool doTransitParallelSampling(CSR* csr, GPUCSRPartition gpuCSRPartition, NextDoorData<SampleType, App>& nextDoorData, bool enableLoadBalancing)
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

  const size_t numDevices = nextDoorData.devices.size();
  size_t finalSampleSize = getFinalSampleSize<App>();
  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    CHK_CU(cudaSetDevice(device));
    const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
    const size_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
    if (App().steps() == 1) {
      CHK_CU(cudaMemcpy(nextDoorData.dSamplesToTransitMapValues[deviceIdx], 
                        &nextDoorData.initialContents[0] + App().initialSampleSize(csr)*deviceSampleStartPtr, 
                        sizeof(VertexID_t)*App().initialSampleSize(csr)*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
      CHK_CU(cudaMemcpy(nextDoorData.dSamplesToTransitMapKeys[deviceIdx], 
                        &nextDoorData.initialTransitToSampleValues[0] + App().initialSampleSize(csr)*deviceSampleStartPtr, 
                        sizeof(VertexID_t)*App().initialSampleSize(csr)*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
    } else {
      CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                        &nextDoorData.initialContents[0] + App().initialSampleSize(csr)*deviceSampleStartPtr, 
                        sizeof(VertexID_t)*App().initialSampleSize(csr)*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
      CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapValues[deviceIdx], 
                        &nextDoorData.initialTransitToSampleValues[0] + App().initialSampleSize(csr)*deviceSampleStartPtr,  
                        sizeof(VertexID_t)*App().initialSampleSize(csr)*perDeviceNumSamples, 
                        cudaMemcpyHostToDevice));
    }
  }

  // for (auto v : nextDoorData.initialTransitToSampleValues) {
  //   if (v != 0) {
  //     printf("v %d\n", v);
  //   }
  // }
  std::vector<VertexID_t*> d_temp_storage = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<size_t> temp_storage_bytes = std::vector<size_t>(nextDoorData.devices.size());;

  std::vector<VertexID_t*> dUniqueTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dUniqueTransitsCounts = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dUniqueTransitsNumRuns = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dTransitPositions = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> uniqueTransitNumRuns = std::vector<EdgePos_t*>(nextDoorData.devices.size());
   
  /**Pointers for each kernel type**/
  std::vector<EdgePos_t*> gridKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dGridKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dGridKernelTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());
  
  std::vector<EdgePos_t*> threadBlockKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dThreadBlockKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dThreadBlockKernelTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());

  std::vector<EdgePos_t*> subWarpKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dSubWarpKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dSubWarpKernelTransits = std::vector<VertexID_t*>(nextDoorData.devices.size());

  std::vector<EdgePos_t*> identityKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dIdentityKernelTransitsNum = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  /**********************************/
  
  /****Variables for Collective Transit Sampling***/
  std::vector<EdgePos_t*> hSumNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dSumNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dSampleNeighborhoodPos = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dSampleNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<VertexID_t*> dCollectiveNeighborhoodCSRCols = std::vector<VertexID_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> dCollectiveNeighborhoodCSRRows = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);

  if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
    for(auto idx = 0; idx < nextDoorData.devices.size(); idx++) {
      auto device = nextDoorData.devices[idx];
      CHK_CU(cudaSetDevice(device));
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), idx, numDevices);
      CHK_CU(cudaMallocHost(&hSumNeighborhoodSizes[idx], sizeof(EdgePos_t)));
      CHK_CU(cudaMalloc(&dSumNeighborhoodSizes[idx], sizeof(EdgePos_t)));
      CHK_CU(cudaMalloc(&dSampleNeighborhoodPos[idx], sizeof(EdgePos_t)*perDeviceNumSamples));
      CHK_CU(cudaMalloc(&dSampleNeighborhoodSizes[idx], sizeof(EdgePos_t)*perDeviceNumSamples));
      CHK_CU(cudaMemset(dSampleNeighborhoodSizes[idx], 0, sizeof(EdgePos_t)*perDeviceNumSamples));
    }
  }

  std::vector<EdgePos_t*> dInvalidVertexStartPosInMap = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> invalidVertexStartPosInMap = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  
  /*Single Memory Location on both CPU and GPU for transferring
   *number of transits for all kernels */
  std::vector<EdgePos_t*> dKernelTransitNums = std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  std::vector<EdgePos_t*> hKernelTransitNums= std::vector<EdgePos_t*>(nextDoorData.devices.size(), nullptr);
  const int NUM_KERNEL_TYPES = TransitKernelTypes::NumKernelTypes + 1;

  std::vector<int*> dKernelTypeForTransit = std::vector<int*>(nextDoorData.devices.size(), nullptr);;

  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
    CHK_CU(cudaSetDevice(device));
    CHK_CU(cudaMallocHost(&uniqueTransitNumRuns[deviceIdx], sizeof(EdgePos_t)));
    CHK_CU(cudaMallocHost(&hKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    
    gridKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx];
    threadBlockKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 1;
    subWarpKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 2;
    identityKernelTransitsNum[deviceIdx] = hKernelTransitNums[deviceIdx] + 3;
    invalidVertexStartPosInMap[deviceIdx] = hKernelTransitNums[deviceIdx] + 4;
    //threadBlockKernelTransitsNum = hKernelTransitNums[3];
    CHK_CU(cudaMalloc(&dKernelTypeForTransit[deviceIdx], sizeof(VertexID_t)*csr->get_n_vertices()));
    CHK_CU(cudaMalloc(&dTransitPositions[deviceIdx], 
                      sizeof(VertexID_t)*csr->get_n_vertices()));
    CHK_CU(cudaMalloc(&dGridKernelTransits[deviceIdx], 
                      sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    if (useThreadBlockKernel) {
      CHK_CU(cudaMalloc(&dThreadBlockKernelTransits[deviceIdx], 
                      sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    }

    if (useSubWarpKernel) {
      CHK_CU(cudaMalloc(&dSubWarpKernelTransits[deviceIdx],
                      sizeof(VertexID_t)*perDeviceNumSamples*maxNeighborsToSample));
    }

    CHK_CU(cudaMalloc(&dKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    CHK_CU(cudaMemset(dKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
    dGridKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx];
    dThreadBlockKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 1;
    dSubWarpKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 2;
    dIdentityKernelTransitsNum[deviceIdx] = dKernelTransitNums[deviceIdx] + 3;
    dInvalidVertexStartPosInMap[deviceIdx] = dKernelTransitNums[deviceIdx] + 4;

    //Check if the space runs out.
    //TODO: Use DoubleBuffer version that requires O(P) space.
    cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx], 
              nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx], 
              perDeviceNumSamples*maxNeighborsToSample);

    CHK_CU(cudaMalloc(&d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx]));
    CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, sizeof(EdgePos_t)*perDeviceNumSamples));

    CHK_CU(cudaMalloc(&dUniqueTransits[deviceIdx], (csr->get_n_vertices() + 1)*sizeof(VertexID_t)));
    CHK_CU(cudaMalloc(&dUniqueTransitsCounts[deviceIdx], (csr->get_n_vertices() + 1)*sizeof(VertexID_t)));
    CHK_CU(cudaMalloc(&dUniqueTransitsNumRuns[deviceIdx], sizeof(size_t)));
  }

  std::vector<VertexID_t*> hAllSamplesToTransitMapKeys;
  std::vector<VertexID_t*> hAllTransitToSampleMapValues;
  std::vector<size_t> totalTransits = std::vector<size_t>(nextDoorData.devices.size());

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
    std::vector<size_t> totalThreads = std::vector<size_t>(nextDoorData.devices.size());
    for(int i = 0; i < nextDoorData.devices.size(); i++) {
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), i, numDevices);
      totalThreads[i] = perDeviceNumSamples*neighborsToSampleAtStep;
    }
    //std::cout << "step " << step << std::endl;
    if (App().steps() == 1) {
      //FIXME: Currently a non-sorted Transit to Sample Map is passed to both TP and TP+LB.
      //Here, if there is only one step, a sorted map is passed.
      //Fix this to make sure a sorted map is always passed.
      double inversionT1 = convertTimeValToDouble(getTimeOfDay ());
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        //Invert sample->transit map by sorting samples based on the transit vertices
        cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx], 
                                        totalThreads[deviceIdx], 0, nextDoorData.maxBits);
        CHK_CU(cudaGetLastError());
      }
      
      CUDA_SYNC_DEVICE_ALL(nextDoorData);
      double inversionT2 = convertTimeValToDouble(getTimeOfDay ());
      //std::cout << "inversionTime at step " << step << " : " << (inversionT2 - inversionT1) << std::endl; 
      inversionTime += (inversionT2 - inversionT1);
    }

    neighborsToSampleAtStep = (App().samplingType() == CollectiveNeighborhood) ? App().stepSize(step) : neighborsToSampleAtStep * App().stepSize(step);    
    for(int i = 0; i < nextDoorData.devices.size(); i++) {
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), i, numDevices);
      totalThreads[i] = perDeviceNumSamples*neighborsToSampleAtStep;
    }

    if ((step == 0 && App().steps() > 1) || !enableLoadBalancing) {
      //When not doing load balancing call baseline transit parallel
      if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));

          CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, 
                            sizeof(VertexID_t) * nextDoorData.samples.size()));
          CHK_CU(cudaMemset(dSumNeighborhoodSizes[deviceIdx], 0, sizeof(EdgePos_t)));
          //Create collective neighborhood for all transits related to a sample
          collectiveNeighbrsSize<App><<<nextDoorData.samples.size(), N_THREADS>>>(step, gpuCSRPartition, 
                                                                              nextDoorData.INVALID_VERTEX,
                                                                              nextDoorData.dInitialSamples[deviceIdx], 
                                                                              nextDoorData.dFinalSamples[deviceIdx], 
                                                                              nextDoorData.samples.size(),
                                                                              dSampleNeighborhoodPos[deviceIdx],
                                                                              dSumNeighborhoodSizes[deviceIdx]);
          CHK_CU(cudaGetLastError());
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          //TODO: Neighborhood is edges of all transit vertices. Hence, neighborhood size is (# of Transit Vertices)/(|G.V|) * |G.E|
          CHK_CU(cudaMemcpy(hSumNeighborhoodSizes[deviceIdx], dSumNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t), cudaMemcpyDeviceToHost));        
          CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRCols[deviceIdx], sizeof(VertexID_t)*(*hSumNeighborhoodSizes[deviceIdx])));
          CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRRows[deviceIdx], sizeof(EdgePos_t)*App().initialSampleSize(csr)*nextDoorData.samples.size()));
        }

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
          CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, sizeof(VertexID_t) * nextDoorData.samples.size()));
          //Compute collective neighborhood using transit parallel kernel
          for (int threadsExecuted = 0; threadsExecuted < totalThreads[deviceIdx]; threadsExecuted += nextDoorData.maxThreadsPerKernel[deviceIdx]) {
            size_t currExecutionThreads = min((size_t)nextDoorData.maxThreadsPerKernel[deviceIdx], totalThreads[deviceIdx] - threadsExecuted);
            samplingKernel<SampleType, App, TransitParallelMode::CollectiveNeighborhoodComputation, 32><<<thread_block_size(currExecutionThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, 
                            threadsExecuted, currExecutionThreads, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                            (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                            totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                            nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                            nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                            dSampleNeighborhoodSizes[deviceIdx], dSampleNeighborhoodPos[deviceIdx], dCollectiveNeighborhoodCSRRows[deviceIdx], dCollectiveNeighborhoodCSRCols[deviceIdx], 
                            nextDoorData.dCurandStates[deviceIdx]);
            CHK_CU(cudaGetLastError());
            // CHK_CU(cudaDeviceSynchronize());
          }
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);
      } else {
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
          
          for (int threadsExecuted = 0; threadsExecuted < totalThreads[deviceIdx]; threadsExecuted += nextDoorData.maxThreadsPerKernel[deviceIdx]) {
            size_t currExecutionThreads = min((size_t)nextDoorData.maxThreadsPerKernel[deviceIdx], totalThreads[deviceIdx] - threadsExecuted);
            samplingKernel<SampleType, App, TransitParallelMode::NextFuncExecution, 0><<<thread_block_size(currExecutionThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, 
                            threadsExecuted, currExecutionThreads, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                            (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                            totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                            nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                            nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                            nullptr,  nullptr,  nullptr,  nullptr, nextDoorData.dCurandStates[deviceIdx]);
            CHK_CU(cudaGetLastError());
          }
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);
      }
    } else {
      if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions[deviceIdx], 0, sizeof(VertexID_t) * nextDoorData.samples.size()));
          CHK_CU(cudaMemset(dSumNeighborhoodSizes[deviceIdx], 0, sizeof(EdgePos_t)));
          //Create collective neighborhood for all transits related to a sample
          collectiveNeighbrsSize<App><<<nextDoorData.samples.size(), N_THREADS>>>(step, gpuCSRPartition, 
                                                                              nextDoorData.INVALID_VERTEX,
                                                                              nextDoorData.dInitialSamples[deviceIdx], 
                                                                              nextDoorData.dFinalSamples[deviceIdx], 
                                                                              nextDoorData.samples.size(),
                                                                              dSampleNeighborhoodPos[deviceIdx],
                                                                              dSumNeighborhoodSizes[deviceIdx]);
          CHK_CU(cudaGetLastError());
          CHK_CU(cudaDeviceSynchronize());
          //TODO: Neighborhood is edges of all transit vertices. Hence, neighborhood size is (# of Transit Vertices)/(|G.V|) * |G.E|
          CHK_CU(cudaMemcpy(hSumNeighborhoodSizes[deviceIdx], dSumNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
          //std::cout <<" hSumNeighborhoodSizes " << *hSumNeighborhoodSizes << std::endl;
          CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRCols[deviceIdx], sizeof(VertexID_t)*(*hSumNeighborhoodSizes[deviceIdx])));
          CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRRows[deviceIdx], sizeof(EdgePos_t)*App().initialSampleSize(csr)*nextDoorData.samples.size()));
        }
      } else {
        double loadBalancingT1 = convertTimeValToDouble(getTimeOfDay ());
        
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          CHK_CU(cudaMemset(dKernelTransitNums[deviceIdx], 0, NUM_KERNEL_TYPES * sizeof(EdgePos_t)));
          CHK_CU(cudaMemset(dInvalidVertexStartPosInMap[deviceIdx], 0xFF, sizeof(EdgePos_t)));
          const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
          totalTransits[deviceIdx] = perDeviceNumSamples*numTransits;

          //Find the index of first invalid transit vertex. 
          invalidVertexStartPos<<<DIVUP(totalTransits[deviceIdx], 256), 256>>>(step, nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                                                                               totalTransits[deviceIdx], nextDoorData.INVALID_VERTEX, 
                                                                               dInvalidVertexStartPosInMap[deviceIdx]);
          CHK_CU(cudaGetLastError());
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          CHK_CU(cudaMemcpy(invalidVertexStartPosInMap[deviceIdx], dInvalidVertexStartPosInMap[deviceIdx], 
                            1 * sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
          //Now the number of threads launched are equal to number of valid transit vertices
          if (*invalidVertexStartPosInMap[deviceIdx] == 0xFFFFFFFF) {
            *invalidVertexStartPosInMap[deviceIdx] = totalTransits[deviceIdx];
          }
          totalThreads[deviceIdx] = *invalidVertexStartPosInMap[deviceIdx];
        }


        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          void* dRunLengthEncodeTmpStorage = nullptr;
          size_t dRunLengthEncodeTmpStorageSize = 0;
          //Find the number of transit vertices
          cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize, 
                                            nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                            dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], dUniqueTransitsNumRuns[deviceIdx], totalThreads[deviceIdx]);

          assert(dRunLengthEncodeTmpStorageSize < temp_storage_bytes[deviceIdx]);
          dRunLengthEncodeTmpStorage = d_temp_storage[deviceIdx];
          cub::DeviceRunLengthEncode::Encode(dRunLengthEncodeTmpStorage, dRunLengthEncodeTmpStorageSize, 
                                            nextDoorData.dTransitToSampleMapKeys[deviceIdx],
                                            dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], dUniqueTransitsNumRuns[deviceIdx], totalThreads[deviceIdx]);

          CHK_CU(cudaGetLastError());
        }
        
        CUDA_SYNC_DEVICE_ALL(nextDoorData);

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          CHK_CU(cudaMemcpy(uniqueTransitNumRuns[deviceIdx], dUniqueTransitsNumRuns[deviceIdx], 
                            sizeof(*uniqueTransitNumRuns[deviceIdx]), cudaMemcpyDeviceToHost));
        
          void* dExclusiveSumTmpStorage = nullptr;
          size_t dExclusiveSumTmpStorageSize = 0;
          //Exclusive sum to obtain the start position of each transit (and its samples) in the map
          cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts[deviceIdx], 
                                        dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);

          assert(dExclusiveSumTmpStorageSize < temp_storage_bytes[deviceIdx]);
          dExclusiveSumTmpStorage = d_temp_storage[deviceIdx];

          cub::DeviceScan::ExclusiveSum(dExclusiveSumTmpStorage, dExclusiveSumTmpStorageSize, dUniqueTransitsCounts[deviceIdx],
                                        dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx]);

          CHK_CU(cudaGetLastError());
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);

        //printKernelTypes<App>(step, csr, dUniqueTransits, dUniqueTransitsCounts, dUniqueTransitsNumRuns);
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) 
            continue;

          partitionTransitsInKernels<App, 1024><<<thread_block_size((*uniqueTransitNumRuns[deviceIdx]), 1024), 1024>>>(step, dUniqueTransits[deviceIdx], dUniqueTransitsCounts[deviceIdx], 
              dTransitPositions[deviceIdx], *uniqueTransitNumRuns[deviceIdx], nextDoorData.INVALID_VERTEX, dGridKernelTransits[deviceIdx], dGridKernelTransitsNum[deviceIdx], 
              dThreadBlockKernelTransits[deviceIdx], dThreadBlockKernelTransitsNum[deviceIdx], dSubWarpKernelTransits[deviceIdx], dSubWarpKernelTransitsNum[deviceIdx], nullptr, 
              dIdentityKernelTransitsNum[deviceIdx], dKernelTypeForTransit[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx]);

          CHK_CU(cudaGetLastError());
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);
        int subWarpSize = subWarpSizeAtStep<App>(step);

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) 
            continue;
          CHK_CU(cudaMemcpy(hKernelTransitNums[deviceIdx], dKernelTransitNums[deviceIdx], NUM_KERNEL_TYPES * sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
          
          //std::cout << "hInvalidVertexStartPosInMap " << *invalidVertexStartPosInMap << " step " << step << std::endl;
          // GPUUtils::printDeviceArray(dGridKernelTransits, *gridKernelTransitsNum, ',');
          // getchar();
          // std::cout << "SubWarpSize at step " << step << " " << subWarpSize << std::endl;
          //From each Transit we sample stepSize(step) vertices
          totalThreads[deviceIdx] =  totalThreads[deviceIdx] * subWarpSize;
        }

        double loadBalancingT2 = convertTimeValToDouble(getTimeOfDay ());
        loadBalancingTime += (loadBalancingT2 - loadBalancingT1);

        bool noTransitsForAllDevices = true;
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          if (*uniqueTransitNumRuns[deviceIdx] > 0) {
            noTransitsForAllDevices = false;
          }
        }

        if (noTransitsForAllDevices)
          //End Sampling because no more transits exists  
          break;

        double identityKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          if (*uniqueTransitNumRuns[deviceIdx] == 0) 
            continue;
          // std::cout << "final totalThreads " << totalThreads << std::endl;
          const size_t maxThreadBlocksPerKernel = min(4096L, nextDoorData.maxThreadsPerKernel[deviceIdx]/256L);
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);

          if (*identityKernelTransitsNum[deviceIdx] > 0) {
            identityKernel<SampleType, App, 256, true><<<maxThreadBlocksPerKernel, 256>>>(step, 
              gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
              (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
              totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
              nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
              nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx]);
            CHK_CU(cudaGetLastError());
          }
        }

        CUDA_SYNC_DEVICE_ALL(nextDoorData);
        double identityKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
        identityKernelTime += (identityKernelTimeT2 - identityKernelTimeT1);

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          const int perThreadSamplesForSubWarpKernel = 1;
          int threadBlocks = DIVUP(DIVUP(*subWarpKernelTransitsNum[deviceIdx]*LoadBalancing::LoadBalancingThreshold::SubWarpLevel, perThreadSamplesForSubWarpKernel), 256);
          //std::cout << "subWarpKernelTransitsNum " << *subWarpKernelTransitsNum << " threadBlocks " << threadBlocks << std::endl;
          double subWarpKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
          if (useSubWarpKernel && *subWarpKernelTransitsNum[deviceIdx] > 0) {
            subWarpKernel<SampleType, App, 256,3*1024,false,false,false,perThreadSamplesForSubWarpKernel,true><<<threadBlocks, 256>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
              (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
              totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
              nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
              nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dSubWarpKernelTransits[deviceIdx], *subWarpKernelTransitsNum[deviceIdx]);
            CHK_CU(cudaGetLastError());
            CHK_CU(cudaDeviceSynchronize());
          }
          double subWarpKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
          subWarpKernelTime += (subWarpKernelTimeT2 - subWarpKernelTimeT1);

          double threadBlockKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());
          const int perThreadSamplesForThreadBlockKernel = 1;
          threadBlocks = DIVUP(*threadBlockKernelTransitsNum[deviceIdx], perThreadSamplesForThreadBlockKernel);
          if (useThreadBlockKernel && *threadBlockKernelTransitsNum[deviceIdx] > 0) {
            threadBlockKernel<SampleType, App, 256,3*1024-3,true,false,false,perThreadSamplesForThreadBlockKernel,true><<<threadBlocks, 32>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
              (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
              totalThreads[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
              nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
              nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
              nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dThreadBlockKernelTransits[deviceIdx], *threadBlockKernelTransitsNum[deviceIdx]);
            CHK_CU(cudaGetLastError());
            CHK_CU(cudaDeviceSynchronize());
          }
          double threadBlockKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
          threadBlockKernelTime += (threadBlockKernelTimeT2 - threadBlockKernelTimeT1);
        }

        double gridKernelTimeT1 = convertTimeValToDouble(getTimeOfDay ());

        for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
          auto device = nextDoorData.devices[deviceIdx];
          CHK_CU(cudaSetDevice(device));
          //Process more than one thread blocks positions written in dGridKernelTransits per thread block.
          //Processing more can improve the locality if thread blocks have common transits.
          const int perThreadSamplesForGridKernel = 16; // Works best for KHop
          //const int perThreadSamplesForGridKernel = 8;
          const size_t maxThreadBlocksPerKernel = min(4096L, nextDoorData.maxThreadsPerKernel[deviceIdx]/256L);
          const VertexID_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
          const size_t threadBlocks = DIVUP(*gridKernelTransitsNum[deviceIdx], perThreadSamplesForGridKernel);
          
          if (useGridKernel && *gridKernelTransitsNum[deviceIdx] > 0 && numberOfTransits<App>(step) > 1) {
            //FIXME: A Bug in Grid Kernel prevents it from being used when numberOfTransits for a sample at step are 1.
            // for (int threadBlocksExecuted = 0; threadBlocksExecuted < threadBlocks; threadBlocksExecuted += nextDoorData.maxThreadsPerKernel/256) {
              const bool CACHE_EDGES = true;
              const bool CACHE_WEIGHTS = false;
              const int CACHE_SIZE = (CACHE_EDGES || CACHE_WEIGHTS) ? 3*1024-10 : 0;
            
              switch (subWarpSizeAtStep<App>(step)) {
                case 32:
                  gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,false,256,32><<<maxThreadBlocksPerKernel, 256>>>(step,
                    gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks);
                    break;
                case 16:
                  gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,16><<<maxThreadBlocksPerKernel, 256>>>(step,
                    gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks);
                    break;
                case 8:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,8><<<maxThreadBlocksPerKernel, 256>>>(step,
                  gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks);
                  break;
                case 4:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,4><<<maxThreadBlocksPerKernel, 256>>>(step,
                  gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks);
                  break;
                case 2:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,2><<<maxThreadBlocksPerKernel, 256>>>(step,
                  gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks);
                  break;
                case 1:
                gridKernel<SampleType,App,256,CACHE_SIZE,CACHE_EDGES,CACHE_WEIGHTS,false,perThreadSamplesForGridKernel,true,true,256,1><<<maxThreadBlocksPerKernel, 256>>>(step,
                  gpuCSRPartition, deviceSampleStartPtr, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys[deviceIdx], (const VertexID_t*)nextDoorData.dTransitToSampleMapValues[deviceIdx],
                    totalThreads[deviceIdx],  nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, nextDoorData.dSampleInsertionPositions[deviceIdx],
                    nextDoorData.dCurandStates[deviceIdx], dKernelTypeForTransit[deviceIdx], dGridKernelTransits[deviceIdx], *gridKernelTransitsNum[deviceIdx], threadBlocks);
                  break;
                default:
                  //TODO: Add others
                    break;
              }
              CHK_CU(cudaGetLastError());
            // }
          }
        }
        CUDA_SYNC_DEVICE_ALL(nextDoorData);
        double gridKernelTimeT2 = convertTimeValToDouble(getTimeOfDay ());
        gridKernelTime += (gridKernelTimeT2 - gridKernelTimeT1);
      }
    }

    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      //Call SampleParallel Kernel to do sampling from collective neighborhood
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));

        sampleParallelKernel<SampleType, App, 256, false><<<min(1024L, nextDoorData.maxThreadsPerKernel[deviceIdx]/256L), 256>>>(step, gpuCSRPartition, 0,
                    nextDoorData.INVALID_VERTEX, totalThreads[deviceIdx], 
                    nextDoorData.dInitialSamples[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], nextDoorData.samples.size(),
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, 
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx],
                    nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dSampleInsertionPositions[deviceIdx], nextDoorData.dCurandStates[deviceIdx]);
        CHK_CU(cudaGetLastError());
      }

      CUDA_SYNC_DEVICE_ALL(nextDoorData);
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        CHK_CU(cudaFree(dCollectiveNeighborhoodCSRCols[deviceIdx]));
        CHK_CU(cudaFree(dCollectiveNeighborhoodCSRRows[deviceIdx]));
      }
    }
    if (step != App().steps() - 1) {
      double inversionT1 = convertTimeValToDouble(getTimeOfDay ());
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        //Invert sample->transit map by sorting samples based on the transit vertices
        cub::DeviceRadixSort::SortPairs(d_temp_storage[deviceIdx], temp_storage_bytes[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapValues[deviceIdx], nextDoorData.dTransitToSampleMapKeys[deviceIdx], 
                                        nextDoorData.dSamplesToTransitMapKeys[deviceIdx], nextDoorData.dTransitToSampleMapValues[deviceIdx], 
                                        totalThreads[deviceIdx], 0, nextDoorData.maxBits);
        CHK_CU(cudaGetLastError());
      }
      CUDA_SYNC_DEVICE_ALL(nextDoorData);
      double inversionT2 = convertTimeValToDouble(getTimeOfDay ());
      //std::cout << "inversionTime at step " << step << " : " << (inversionT2 - inversionT1) << std::endl; 
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

  std::cout << "Transit Parallel: End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  std::cout << "InversionTime: " << inversionTime <<", " << "LoadBalancingTime: " << loadBalancingTime << ", " << "GridKernelTime: " << gridKernelTime << ", ThreadBlockKernelTime: " << threadBlockKernelTime << ", SubWarpKernelTime: " << subWarpKernelTime << ", IdentityKernelTime: "<< identityKernelTime << std::endl;
  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    CHK_CU(cudaFree(d_temp_storage[deviceIdx]));
    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      CHK_CU(cudaFreeHost(hSumNeighborhoodSizes[deviceIdx]));
      CHK_CU(cudaFree(dSumNeighborhoodSizes[deviceIdx]));
      CHK_CU(cudaFree(dSampleNeighborhoodPos[deviceIdx]));
    }


    CHK_CU(cudaFree(dUniqueTransits[deviceIdx]));
    CHK_CU(cudaFree(dUniqueTransitsCounts[deviceIdx]));
    CHK_CU(cudaFree(dUniqueTransitsNumRuns[deviceIdx]));
    CHK_CU(cudaFree(dKernelTypeForTransit[deviceIdx]));
    CHK_CU(cudaFree(dTransitPositions[deviceIdx]));
    CHK_CU(cudaFree(dGridKernelTransits[deviceIdx]));
    CHK_CU(cudaFree(dThreadBlockKernelTransits[deviceIdx]));
    CHK_CU(cudaFree(dSubWarpKernelTransits[deviceIdx]));
  }
  
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
bool doSampleParallelSampling(CSR* csr, GPUCSRPartition gpuCSRPartition, NextDoorData<SampleType, App>& nextDoorData)
{
  //Size of each sample output
  int finalSampleSize = getFinalSampleSize<App>();
  int neighborsToSampleAtStep = App().initialSampleSize(csr);
  size_t numDevices = nextDoorData.devices.size();

  std::vector<EdgePos_t*> hSumNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dSumNeighborhoodSizes = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dSampleNeighborhoodPos = std::vector<EdgePos_t*>(nextDoorData.devices.size());
  std::vector<VertexID_t*> dCollectiveNeighborhoodCSRCols = std::vector<VertexID_t*>(nextDoorData.devices.size());
  std::vector<EdgePos_t*> dCollectiveNeighborhoodCSRRows = std::vector<EdgePos_t*>(nextDoorData.devices.size());

  if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
    for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
      CHK_CU(cudaSetDevice(device));
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
      CHK_CU(cudaMallocHost(&hSumNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t)));
      CHK_CU(cudaMalloc(&dSumNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t)));
      CHK_CU(cudaMalloc(&dSampleNeighborhoodPos[deviceIdx], sizeof(EdgePos_t)*perDeviceNumSamples));
    }
  }

  std::vector<size_t> totalThreads = std::vector<size_t>(nextDoorData.devices.size());
  
  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  double collectiveNeighborhoodTime = 0.0f;

  for (int step = 0; step < App().steps(); step++) {
    //Number of threads created are equal to number of new neighbors to be sampled at a step.
    //In collective neighborhood we sample stepSize(step) vertices at each step
    //Otherwise need to sample product.

    const size_t numTransits = (App().samplingType() == CollectiveNeighborhood) ? 1 : neighborsToSampleAtStep;
    neighborsToSampleAtStep = (App().samplingType() == CollectiveNeighborhood) ? App().stepSize(step) : neighborsToSampleAtStep * App().stepSize(step);
    // std::cout << "totalThreads " << totalThreads << std::endl;
    double collectiveNeighborhood_t0 = convertTimeValToDouble(getTimeOfDay());
    
    for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
      auto device = nextDoorData.devices[deviceIdx];
      CHK_CU(cudaSetDevice(device));
      const size_t perDeviceNumSamples = PartDivisionSize(nextDoorData.samples.size(), deviceIdx, numDevices);
      const size_t deviceSampleStartPtr = PartStartPointer(nextDoorData.samples.size(), deviceIdx, numDevices);
      totalThreads[deviceIdx] = perDeviceNumSamples*neighborsToSampleAtStep;
      if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
        assert(nextDoorData.devices.size() == 1);
        //FIXME: CollectiveNeighborhood Sampling for more than one GPU in Sample Parallel is not support 
        //Create collective neighborhood for all transits related to a sample
        collectiveNeighbrsSize<App><<<nextDoorData.samples.size(), N_THREADS>>>(step, gpuCSRPartition, 
                                                                            nextDoorData.INVALID_VERTEX,
                                                                            nextDoorData.dInitialSamples[deviceIdx], 
                                                                            nextDoorData.dFinalSamples[deviceIdx], 
                                                                            nextDoorData.samples.size(),
                                                                            dSampleNeighborhoodPos[deviceIdx],
                                                                            dSumNeighborhoodSizes[deviceIdx]);
        CHK_CU(cudaGetLastError());
        CHK_CU(cudaDeviceSynchronize());
        double __t1 = convertTimeValToDouble(getTimeOfDay());
        //TODO: Neighborhood is edges of all transit vertices. Hence, neighborhood size is (# of Transit Vertices)/(|G.V|) * |G.E|
        CHK_CU(cudaMemcpy(hSumNeighborhoodSizes[deviceIdx], dSumNeighborhoodSizes[deviceIdx], sizeof(EdgePos_t), cudaMemcpyDeviceToHost));
        CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRCols[deviceIdx], sizeof(VertexID_t)*(*hSumNeighborhoodSizes[deviceIdx])));
        CHK_CU(cudaMalloc(&dCollectiveNeighborhoodCSRRows[deviceIdx], sizeof(EdgePos_t)*App().initialSampleSize(csr)*nextDoorData.samples.size()));
        double __t2 = convertTimeValToDouble(getTimeOfDay());
        
        collectiveNeighborhood<App><<<nextDoorData.samples.size(), N_THREADS>>>(step, gpuCSRPartition, 
                                                                            nextDoorData.INVALID_VERTEX,
                                                                            nextDoorData.dInitialSamples[deviceIdx],
                                                                            nextDoorData.dFinalSamples[deviceIdx], 
                                                                            nextDoorData.samples.size(),
                                                                            dCollectiveNeighborhoodCSRRows[deviceIdx],
                                                                            dCollectiveNeighborhoodCSRCols[deviceIdx],
                                                                            dSampleNeighborhoodPos[deviceIdx],
                                                                            dSumNeighborhoodSizes[deviceIdx]);
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

      double collectiveNeighborhood_t1 = convertTimeValToDouble(getTimeOfDay());
      collectiveNeighborhoodTime += (collectiveNeighborhood_t1 - collectiveNeighborhood_t0);

      if (App().hasExplicitTransits() and step > 0) {
        const size_t totalThreads = App().numSamples(csr)*numTransits;
        for (int _thExecs = 0; _thExecs < totalThreads; _thExecs += nextDoorData.maxThreadsPerKernel[deviceIdx]) {
          const size_t currExecThreads = min(nextDoorData.maxThreadsPerKernel[deviceIdx], totalThreads - _thExecs);

          explicitTransitsKernel<SampleType, App, false><<<DIVUP(currExecThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, 
                                                                                                      nextDoorData.INVALID_VERTEX,
                                                                                                      _thExecs, currExecThreads,
                                                                                                      totalThreads,
                                                                                                      nextDoorData.dOutputSamples[deviceIdx],
                                                                                                      nextDoorData.samples.size(),
                                                                                                      nullptr,
                                                                                                      nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                                                                                                      nextDoorData.dCurandStates[deviceIdx]);
          
          CHK_CU(cudaGetLastError());
          CHK_CU(cudaDeviceSynchronize());
        }
      }

      //Perform SampleParallel Sampling
      sampleParallelKernel<SampleType, App, 256, false><<<min(1024L, nextDoorData.maxThreadsPerKernel[deviceIdx]/256L), 256>>>(step, gpuCSRPartition, 
                    deviceSampleStartPtr, nextDoorData.INVALID_VERTEX, totalThreads[deviceIdx], 
                    nextDoorData.dInitialSamples[deviceIdx], nextDoorData.dOutputSamples[deviceIdx], perDeviceNumSamples,
                    nextDoorData.dFinalSamples[deviceIdx], finalSampleSize, 
                    nextDoorData.dSamplesToTransitMapKeys[deviceIdx],
                    nextDoorData.dSamplesToTransitMapValues[deviceIdx],
                    nextDoorData.dSampleInsertionPositions[deviceIdx], nextDoorData.dCurandStates[deviceIdx]);
      CHK_CU(cudaGetLastError());
      CHK_CU(cudaDeviceSynchronize());
    }

    if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
      for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
        auto device = nextDoorData.devices[deviceIdx];
        CHK_CU(cudaSetDevice(device));
        CHK_CU(cudaFree(dCollectiveNeighborhoodCSRCols[deviceIdx]));
        CHK_CU(cudaFree(dCollectiveNeighborhoodCSRRows[deviceIdx]));
      }
    }
  }

  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());
  
  std::cout << "SampleParallel: End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
    std::cout << "Collective Neighborhood Computing " << collectiveNeighborhoodTime << " secs" << std::endl;
  }

  if (App().samplingType() == SamplingType::CollectiveNeighborhood) {
    for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
      auto device = nextDoorData.devices[deviceIdx];
      CHK_CU(cudaFreeHost(hSumNeighborhoodSizes[deviceIdx]));
      CHK_CU(cudaFree(dSumNeighborhoodSizes[deviceIdx]));
      CHK_CU(cudaFree(dSampleNeighborhoodPos[deviceIdx]));
    }
  }

  return true;
}

template<class SampleType, typename App>
std::vector<VertexID_t>& getFinalSamples(NextDoorData<SampleType, App>& nextDoorData)
{
  for(auto deviceIdx = 0; deviceIdx < nextDoorData.devices.size(); deviceIdx++) {
    auto device = nextDoorData.devices[deviceIdx];
    //Per Device Allocation
    CHK_CU(cudaSetDevice(device));
    const size_t numSamples = nextDoorData.samples.size();
    const size_t finalSampleSize = getFinalSampleSize<App>();

    const size_t perDeviceNumSamples = PartDivisionSize(numSamples, deviceIdx, nextDoorData.devices.size());
    const size_t deviceSampleStartPtr = PartStartPointer(numSamples, deviceIdx, nextDoorData.devices.size());

    CHK_CU(cudaMemcpy(&nextDoorData.hFinalSamples[0] + finalSampleSize * deviceSampleStartPtr, nextDoorData.dFinalSamples[deviceIdx], 
                      sizeof(nextDoorData.hFinalSamples[0]) * finalSampleSize * perDeviceNumSamples, cudaMemcpyDeviceToHost));
    CHK_CU(cudaMemcpy(&nextDoorData.samples[0] + deviceSampleStartPtr, nextDoorData.dOutputSamples[deviceIdx], 
                      perDeviceNumSamples*sizeof(SampleType), cudaMemcpyDeviceToHost));
  }

  return nextDoorData.hFinalSamples;
}

template<class SampleType, typename App>
bool nextdoor(const char* graph_file, const char* graph_type, const char* graph_format, 
             const int nruns, const bool chk_results, const bool print_samples,
             const char* kernelType, const bool enableLoadBalancing,
             bool (*checkResultsFunc)(NextDoorData<SampleType, App>&))
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
  NextDoorData<SampleType, App> nextDoorData;

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