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

/**User Defined Functions**/

//GraphSage 2-hop sampling
const bool has_random = true;
extern "C" {
  __host__ __device__ int steps();

  __host__ __device__ 
  int stepSize(int k);

  __device__ inline
  VertexID next(int step, const VertexID transit, const VertexID sample, 
                const CSR::Edge* transitEdges, const EdgePos_t numEdges,
                const EdgePos_t neighbrID, 
                curandState* state);
  __host__ __device__ int steps();
}

// __host__ __device__ 
// int stepSize(int k) {
//   return ((k == 0) ? 5 : 2);
// }

// __device__ inline
// VertexID next(int step, const VertexID transit, const VertexID sample, 
//               const CSR::Edge* transitEdges, const EdgePos_t numEdges,
//               const EdgePos_t neighbrID, 
//               curandState* state)
// {
//   EdgePos_t id = RandNumGen::rand_int(state, numEdges);
//   // if (sample == 100 && transit == 100) {
//   //   printf("113: id %ld transitEdges[id] %d\n", (long)id, transitEdges[id]);
//   // }
//   return transitEdges[id];
// }

/**********************/

__host__ __device__
EdgePos_t newNeighborsSize(int hop, EdgePos_t num_edges)
{
  return (stepSize(hop) == ALL_NEIGHBORS) ? num_edges : (EdgePos_t)stepSize(hop);
}

__host__ __device__
EdgePos_t stepSizeAtStep(int step)
{
  if (step == -1)
    return 0;

  EdgePos_t n = 1;
  for (int i = 0; i <= step; i++) {
    n = n * stepSize(i);
  }

  return n;
}


__host__ __device__ int numberOfTransits(int step) {
  return stepSizeAtStep(step);
}

#include "check_results.cu"

__global__ void samplingKernel(const int step, GPUCSRPartition graph, const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               curandState* randStates)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  //__shared__ VertexID newNeigbhors[N_THREADS];

  if (threadId >= transitToSamplesSize)
    return;
  
  EdgePos_t transitIdx = threadId/stepSize(step);
  EdgePos_t transitNeighborIdx = threadId % stepSize(step);
  
  VertexID_t sample = transitToSamplesValues[transitIdx];
  assert(sample < NumSamples);
  VertexID_t transit = transitToSamplesKeys[transitIdx];
  VertexID_t neighbor = invalidVertex;

  if (transit != invalidVertex) {
    // if (graph.device_csr->has_vertex(transit) == false)
    //   printf("transit %d\n", transit);
    assert(graph.device_csr->has_vertex(transit));

    EdgePos_t numTransitEdges = graph.device_csr->get_n_edges_for_vertex(transit);
    const CSR::Edge* transitEdges = graph.device_csr->get_edges(transit);

    if (numTransitEdges != 0) {
      curandState* randState = &randStates[threadId];
      neighbor = next(step, transit, sample, transitEdges, numTransitEdges, 
                      transitNeighborIdx, randState);
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

  EdgePos_t totalSizeOfSample = stepSizeAtStep(step - 1);

  samplesToTransitKeys[threadId] = sample;
  samplesToTransitValues[threadId] = neighbor;
  
  EdgePos_t insertionPos = 0; 
  if (numberOfTransits(step) > 1) {    
    insertionPos = utils::atomicAdd(&sampleInsertionPositions[sample], 1);
  } else {
    insertionPos = step;
  }

  // if (insertionPos < finalSampleSize) {
  //   printf("insertionPos %d finalSampleSize %d\n", insertionPos, finalSampleSize);
  // }
  assert(finalSampleSize > 0);
  if (insertionPos >= finalSampleSize) {
    printf("insertionPos %d finalSampleSize %ld sample %d\n", insertionPos, finalSampleSize, sample);
  }
  assert(insertionPos < finalSampleSize);
  finalSamples[sample*finalSampleSize + insertionPos] = neighbor;
  // if (sample == 100) {
  //   printf("neighbor for 100 %d insertionPos %ld transit %d\n", neighbor, (long)insertionPos, transit);
  // }
  //TODO: We do not need atomic instead store indices of transit in another array,
  //wich can be accessed based on sample and transitIdx.
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
  copyPartitionToGPU(full_partition, gpuCSRPartition);

  return gpuCSRPartition;
}

bool allocNextDoorDataOnGPU(CSR* csr, NextDoorData& data)
{
  //Initially each sample contains only one vertex
  //Allocate one sample for each vertex
  int maxV = 0;
  for (auto vertex : csr->iterate_vertices()) {
    data.samples.push_back(vertex);
    maxV = (maxV < vertex) ? vertex : maxV;
  }
  
  int maxBits = 0;
  while ((maxV >> maxBits) == 0) {
    maxBits++;
  }
  
  data.maxBits = maxBits + 1;

  //Size of each sample output
  size_t maxNeighborsToSample = 1;
  for (int step = 0; step < steps(); step++) {
    maxNeighborsToSample *= stepSize(step);
  }

  size_t finalSampleSize = 0;
  size_t neighborsToSampleAtStep = 1;
  for (int step = 0; step < steps(); step++) {
    neighborsToSampleAtStep *= stepSize(step);
    finalSampleSize += neighborsToSampleAtStep;
  }

  data.INVALID_VERTEX = csr->get_n_vertices();
  //Allocate storage for final samples on GPU
  data.hFinalSamples = std::vector<VertexID_t>(finalSampleSize*data.samples.size());

  CHK_CU(cudaMalloc(&data.dFinalSamples, sizeof(VertexID_t)*data.hFinalSamples.size()));
  gpu_memset(data.dFinalSamples, data.INVALID_VERTEX, data.hFinalSamples.size());
  //Samples to Transit Map
  CHK_CU(cudaMalloc(&data.dSamplesToTransitMapKeys, sizeof(VertexID_t)*data.hFinalSamples.size()));
  CHK_CU(cudaMalloc(&data.dSamplesToTransitMapValues, sizeof(VertexID_t)*data.hFinalSamples.size()));

  //Transit to Samples Map
  CHK_CU(cudaMalloc(&data.dTransitToSampleMapKeys, sizeof(VertexID_t)*data.hFinalSamples.size()));
  CHK_CU(cudaMalloc(&data.dTransitToSampleMapValues, sizeof(VertexID_t)*data.hFinalSamples.size()));

  //Same as initial values of samples for first iteration
  CHK_CU(cudaMemcpy(data.dTransitToSampleMapKeys, &data.samples[0], sizeof(VertexID_t)*data.samples.size(), 
                    cudaMemcpyHostToDevice));
  CHK_CU(cudaMemcpy(data.dTransitToSampleMapValues, &data.samples[0], sizeof(VertexID_t)*data.samples.size(), 
                    cudaMemcpyHostToDevice));

  //Insertion positions per transit vertex for each sample
  
  CHK_CU(cudaMalloc(&data.dSampleInsertionPositions, sizeof(EdgePos_t)*data.samples.size()));

  CHK_CU(cudaMalloc(&data.dCurandStates, maxNeighborsToSample*data.samples.size()*sizeof(curandState)));
  init_curand_states<<<thread_block_size(data.samples.size()*maxNeighborsToSample, 256UL), 256UL>>> (data.dCurandStates, data.samples.size()*maxNeighborsToSample);
  CHK_CU(cudaDeviceSynchronize());

  return true;
}

void freeDeviceData(NextDoorData& data) 
{
  CHK_CU(cudaFree(data.dSamplesToTransitMapKeys));
  CHK_CU(cudaFree(data.dSamplesToTransitMapValues));
  CHK_CU(cudaFree(data.dTransitToSampleMapKeys));
  CHK_CU(cudaFree(data.dTransitToSampleMapValues));
  CHK_CU(cudaFree(data.dSampleInsertionPositions));
  CHK_CU(cudaFree(data.dCurandStates));
  CHK_CU(cudaFree(data.dFinalSamples));
}

bool doSampling(GPUCSRPartition gpuCSRPartition, NextDoorData& nextDoorData)
{
  //Size of each sample output
  size_t maxNeighborsToSample = 1;
  for (int step = 0; step < steps(); step++) {
    maxNeighborsToSample *= stepSize(step);
  }

  size_t finalSampleSize = 0;
  size_t neighborsToSampleAtStep = 1;
  for (int step = 0; step < steps(); step++) {
    neighborsToSampleAtStep *= stepSize(step);
    finalSampleSize += neighborsToSampleAtStep;
  }
  
  neighborsToSampleAtStep = 1;
  CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapKeys, &nextDoorData.samples[0], sizeof(VertexID_t)*nextDoorData.samples.size(), 
                  cudaMemcpyHostToDevice));
  CHK_CU(cudaMemcpy(nextDoorData.dTransitToSampleMapValues, &nextDoorData.samples[0], sizeof(VertexID_t)*nextDoorData.samples.size(), 
                  cudaMemcpyHostToDevice));
  VertexID_t* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  
  //Check if the space runs out.
  //TODO: Use DoubleBuffer version that requires O(P) space.
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
            nextDoorData.dSamplesToTransitMapValues, nextDoorData.dTransitToSampleMapKeys, 
            nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dTransitToSampleMapValues, nextDoorData.hFinalSamples.size());
  
  CHK_CU(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  for (int step = 0; step < steps(); step++) {
    neighborsToSampleAtStep *= stepSize(step);
    const size_t totalThreads = nextDoorData.samples.size()*neighborsToSampleAtStep;
    
    //Sample neighbors of transit vertices
    samplingKernel<<<thread_block_size(totalThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, nextDoorData.INVALID_VERTEX,
                    (const VertexID_t*)nextDoorData.dTransitToSampleMapKeys, (const VertexID_t*)nextDoorData.dTransitToSampleMapValues,
                    totalThreads, nextDoorData.samples.size(),
                    nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dSamplesToTransitMapValues,
                    nextDoorData.dFinalSamples, finalSampleSize, nextDoorData.dSampleInsertionPositions,
                    nextDoorData.dCurandStates);
    CHK_CU(cudaGetLastError());
    CHK_CU(cudaDeviceSynchronize());

    if (step != steps() - 1) {
      //Invert sample->transit map by sorting samples based on the transit vertices
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                      nextDoorData.dSamplesToTransitMapValues, nextDoorData.dTransitToSampleMapKeys, 
                                      nextDoorData.dSamplesToTransitMapKeys, nextDoorData.dTransitToSampleMapValues, 
                                      totalThreads, 0, nextDoorData.maxBits);
    }
  }

  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());
  

  CHK_CU(cudaMemset(nextDoorData.dSampleInsertionPositions, 0, sizeof(EdgePos_t)*nextDoorData.samples.size()));

  std::cout << "End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  return true;
}

std::vector<VertexID_t>& getFinalSamples(NextDoorData& nextDoorData)
{
  CHK_CU(cudaMemcpy(&nextDoorData.hFinalSamples[0], nextDoorData.dFinalSamples, 
                    nextDoorData.hFinalSamples.size()*sizeof(nextDoorData.hFinalSamples[0]), cudaMemcpyDeviceToHost));
  return nextDoorData.hFinalSamples;
}

int nextdoor(const char* graph_file, const char* graph_type, const char* graph_format, 
             const int nruns, const bool chk_results, const bool print_samples)
{
  std::vector<Vertex> vertices;

  //Load Graph
  Graph graph;
  CSR* csr;
  if ((csr = loadGraph(graph, (char*)graph_file, (char*)graph_type, (char*)graph_format)) == nullptr) {
    return 1;
  }

  std::cout << "Graph has " <<graph.get_n_edges () << " edges and " << 
      graph.get_vertices ().size () << " vertices " << std::endl; 

  //graph.print(std::cout);
  GPUCSRPartition gpuCSRPartition = transferCSRToGPU(csr);

  NextDoorData nextDoorData;
  allocNextDoorDataOnGPU(csr, nextDoorData);
  
  for (int i = 0; i < nruns; i++)
    doSampling(gpuCSRPartition, nextDoorData);

  std::vector<VertexID_t>& hFinalSamples = getFinalSamples(nextDoorData);

  size_t maxNeighborsToSample = 1;
  for (int step = 0; step < steps(); step++) {
    maxNeighborsToSample *= stepSize(step);
  }

  size_t finalSampleSize = 0;
  size_t neighborsToSampleAtStep = 1;
  for (int step = 0; step < steps(); step++) {
    neighborsToSampleAtStep *= stepSize(step);
    finalSampleSize += neighborsToSampleAtStep;
  }
  
  size_t totalSampledVertices = 0;
  for (auto s : hFinalSamples) {
    totalSampledVertices += (int)(s != nextDoorData.INVALID_VERTEX);
  }

  if (print_samples) {
    for (size_t s = 0; s < hFinalSamples.size(); s += finalSampleSize) {
      std::cout << "Contents of sample " << s/finalSampleSize << " [";
      for(size_t v = s; v < s + finalSampleSize; v++)
        std::cout << hFinalSamples[v] << ", ";
      std::cout << "]" << std::endl;
    }
  }

  std::cout << "totalSampledVertices " << totalSampledVertices << std::endl;
  freeDeviceData(nextDoorData);
  if (chk_results)
    return check_result(csr, nextDoorData.INVALID_VERTEX, nextDoorData.samples, finalSampleSize, hFinalSamples);

  return 0;
}

#endif