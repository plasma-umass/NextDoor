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

#include <anyoption.h>
#include "sample.hpp"

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
#include "pinned_memory_alloc.hpp"
#include "sampler.cuh"
#include "rand_num_gen.cuh"

using namespace utils;
using namespace GPUUtils;

#define CHECK_RESULT

//For mico, 512 works best
const int N_THREADS = 256;

//TODO try for larger random walks to improve results

#define WARP_HOP

const int ALL_NEIGHBORS = -1;

/**User Defined Functions**/

//GraphSage 2-hop sampling
const bool has_random = true;
__host__ __device__ int steps() {return 2;}

__host__ __device__ 
int stepSize(int k) {
  return ((k == 0) ? 25 : 10);
}

__device__ inline
VertexID next(int step, const VertexID transit, const VertexID sample, 
              const CSR::Edge* transitEdges, const EdgePos_t numEdges,
              const EdgePos_t neighbrID, 
              curandState* state)
{
  EdgePos_t id = RandNumGen::rand_int(state, numEdges);
  return transitEdges[0];
}

/**********************/

#include "check_results.cu"

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

__global__ void samplingKernel(const int step, GPUCSRPartition graph, const VertexID_t invalidVertex,
                               const VertexID_t* transitToSamplesKeys, const VertexID_t* transitToSamplesValues,
                               const size_t transitToSamplesSize, const size_t NumSamples,
                               VertexID_t* samplesToTransitKeys, VertexID_t* samplesToTransitValues,
                               VertexID_t* finalSamples, const size_t finalSampleSize, EdgePos_t* sampleInsertionPositions,
                               curandState* randStates)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;

  if (threadId >= transitToSamplesSize)
    return;
  
  EdgePos_t transitIdx = threadId/stepSize(step);
  EdgePos_t transitNeighborIdx = threadId % stepSize(step);
  
  VertexID_t sample = transitToSamplesValues[transitIdx];
  assert(sample < NumSamples);
  VertexID_t transit = transitToSamplesKeys[transitIdx];
  VertexID_t neighbor = invalidVertex;

  if (transit != invalidVertex) {
    assert(graph.device_csr->has_vertex(transit));

    EdgePos_t numTransitEdges = graph.device_csr->get_n_edges_for_vertex(transit);
    const CSR::Edge* transitEdges = graph.device_csr->get_edges(transit);

    if (numTransitEdges != 0) {
      curandState* randState = &randStates[threadId];
      neighbor = next(step, transit, sample, transitEdges, numTransitEdges, 
                      transitNeighborIdx, randState);
    }
  }

  __syncwarp();

  EdgePos_t totalSizeOfSample = stepSizeAtStep(step - 1);

  samplesToTransitKeys[transitIdx*stepSize(step) + transitNeighborIdx] = sample;
  samplesToTransitValues[transitIdx*stepSize(step) + transitNeighborIdx] = neighbor;
  
  EdgePos_t insertionPos = 0;
  if (stepSize(step) > 1)
    insertionPos = utils::atomicAdd(&sampleInsertionPositions[sample], 1);

  assert(insertionPos < finalSampleSize);
  finalSamples[sample*finalSampleSize + insertionPos] = neighbor;

  //TODO: We do not need atomic instead store indices of transit in another array,
  //wich can be accessed based on sample and transitIdx.
}

__global__ void init_curand_states(curandState* states, size_t num_states)
{
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  if (thread_id < num_states)
    curand_init(1234, 0, 0, &states[thread_id]);
}

bool loadGraph(Graph& graph, AnyOption* opt) 
{
  char* graph_file = opt->getValue('g');
  char* graph_type = opt->getValue('t');
  char* graph_format = opt->getValue('f');

  if (graph_file == nullptr || graph_type == nullptr || 
      graph_format == nullptr) {
    opt->printUsage();
    delete opt;
    return 0;
  }

  //Load Graph
  if (strcmp(graph_type, "adj-list") == 0) {
    if (strcmp(graph_format, "text") == 0) {
      graph.load_from_adjacency_list(graph_file);
      return true;
    }
    else {
      printf ("graph_format '%s' not supported for graph_type '%s'\n", 
              graph_format, graph_type);
      return false;
    }
  } else if (strcmp(graph_type, "edge-list") == 0) {
    if (strcmp(graph_format, "binary") == 0) {
      graph.load_from_edge_list_binary(graph_file, true);
      return true;
    } else if (strcmp(graph_format, "text") == 0) {
      FILE* fp = fopen (graph_file, "r");
      if (fp == nullptr) {
        std::cout << "File '" << graph_file << "' not found" << std::endl;
        return false;
      }
      graph.load_from_edge_list_txt(fp, true);
      fclose (fp);
      return true;
    } else {
      printf ("graph_format '%s' not supported for graph_type '%s'\n", 
              graph_format, graph_type);
      return false;
    }
  } else {
    printf("Incorrect graph file type '%s'\n", graph_type);
    return false;
  }

  return false;
}

int main(int argc, char* argv[])
{
  std::vector<Vertex> vertices;

  AnyOption *opt = new AnyOption();
  opt->addUsage("usage: ");
  opt->addUsage("");
  opt->addUsage("-h --help        Prints this help");
  opt->addUsage("-g --graph-file  File containing graph");
  opt->addUsage("-t --graph-type <type> Format of graph file: 'adj-list' or 'edge-list'");
  opt->addUsage("-f --format <format> Format of graph file: 'binary' or 'text'");
  opt->addUsage("-chk --check-results Check results using an algorithm");
  opt->addUsage("-p --print-samples Print Samples");

  opt->setFlag("help", 'h');
  opt->setOption("graph-file",  'g');
  opt->setOption("graph-type", 't');
  opt->setOption("graph-format", 'f');
  opt->setFlag("print-samples", 'p');
  opt->setFlag("check-results", 'chk');

  opt->processCommandArgs(argc, argv);

  if (!opt->hasOptions()) {
    opt->printUsage();
    delete opt;
    return 0;
  }  

  //Load Graph
  Graph graph;
  if (loadGraph(graph, opt) == false) {
    return 1;
  }

  std::cout << "Graph has " <<graph.get_n_edges () << " edges and " << 
      graph.get_vertices ().size () << " vertices " << std::endl; 

  //Convert graph to CSR format
  CSR* csr = new CSR(graph.get_vertices().size(), graph.get_n_edges());
  csr_from_graph (csr, graph);

  double total_stream_time = 0;

  double_t kernelTotalTime = 0.0;
  std::vector<CSRPartition> csr_partitions;

  //Assume that whole graph can be stored in GPU Memory.
  //Hence, only one Graph Partition is created.
  CSRPartition full_partition = CSRPartition (0, csr->get_n_vertices () - 1, 0, csr->get_n_edges () - 1, 
                                              csr->get_vertices (), csr->get_edges ());
  csr_partitions.push_back (full_partition);

  assert(csr_partitions.size() == 1);
  
  //Copy full graph to GPU
  GPUCSRPartition gpuCSRPartition;
  copy_partition_to_gpu(csr_partitions[0], gpuCSRPartition);

  //Initially each sample contains only one vertex
  std::vector<VertexID_t> samples;

  //Allocate one sample for each vertex
  for (auto vertex : csr->iterate_vertices()) {
    samples.push_back(vertex);
  }
  
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

  //Allocate storage for final samples on GPU
  std::vector<VertexID_t> hFinalSamples(finalSampleSize*samples.size());

  VertexID_t* dFinalSamples;
  CHK_CU(cudaMalloc(&dFinalSamples, sizeof(VertexID_t)*hFinalSamples.size()));

  //Samples to Transit Map
  VertexID_t* dSamplesToTransitMapKeys;
  VertexID_t* dSamplesToTransitMapValues;
  CHK_CU(cudaMalloc(&dSamplesToTransitMapKeys, sizeof(VertexID_t)*hFinalSamples.size()));
  CHK_CU(cudaMalloc(&dSamplesToTransitMapValues, sizeof(VertexID_t)*hFinalSamples.size()));

  //Transit to Samples Map
  VertexID_t* dTransitToSampleMapKeys;
  VertexID_t* dTransitToSampleMapValues;
  CHK_CU(cudaMalloc(&dTransitToSampleMapKeys, sizeof(VertexID_t)*hFinalSamples.size()));
  CHK_CU(cudaMalloc(&dTransitToSampleMapValues, sizeof(VertexID_t)*hFinalSamples.size()));

  //Same as initial values of samples for first iteration
  CHK_CU(cudaMemcpy(dTransitToSampleMapKeys, &samples[0], sizeof(VertexID_t)*samples.size(), 
                    cudaMemcpyHostToDevice));
  CHK_CU(cudaMemcpy(dTransitToSampleMapValues, &samples[0], sizeof(VertexID_t)*samples.size(), 
                    cudaMemcpyHostToDevice));

  //Insertion positions per transit vertex for each sample
  EdgePos_t* dSampleInsertionPositions;
  CHK_CU(cudaMalloc(&dSampleInsertionPositions, sizeof(EdgePos_t)*samples.size()));

  const VertexID_t INVALID_VERTEX = graph.get_vertices().size();
  
  curandState* dCurandStates;
  
  CHK_CU(cudaMalloc(&dCurandStates, maxNeighborsToSample*samples.size()*sizeof(curandState)));
  init_curand_states<<<next_multiple(samples.size()*maxNeighborsToSample, 256), 256>>> (dCurandStates, samples.size()*maxNeighborsToSample);
  CHK_CU(cudaDeviceSynchronize());

  neighborsToSampleAtStep = 1;
  
  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  for (int step = 0; step < steps(); step++) {
    neighborsToSampleAtStep *= stepSize(step);
    const size_t totalThreads = samples.size()*neighborsToSampleAtStep;
    
    //Sample neighbors of transit vertices
    samplingKernel<<<next_multiple(totalThreads, N_THREADS), N_THREADS>>>(step, gpuCSRPartition, INVALID_VERTEX,
                   (const VertexID_t*)dTransitToSampleMapKeys, (const VertexID_t*)dTransitToSampleMapValues,
                   totalThreads, samples.size(),
                   dSamplesToTransitMapKeys, dSamplesToTransitMapValues,
                   dFinalSamples, finalSampleSize, dSampleInsertionPositions,
                   dCurandStates);
    CHK_CU(cudaGetLastError());
    CHK_CU(cudaDeviceSynchronize());

    if (step != steps() - 1) {
      //Invert sample->transit map by sorting samples based on the transit vertices
      VertexID_t* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      
      //Check if the space runs out.
      //TODO: Use DoubleBuffer version that requires O(P) space.
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                      dSamplesToTransitMapValues, dTransitToSampleMapKeys, dSamplesToTransitMapKeys, dTransitToSampleMapValues, totalThreads);
      
      CHK_CU (cudaMalloc(&d_temp_storage, temp_storage_bytes));

      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                      dSamplesToTransitMapValues, dTransitToSampleMapKeys, dSamplesToTransitMapKeys, dTransitToSampleMapValues, totalThreads);
    }
  }

  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());
  

  CHK_CU(cudaMemcpy(&hFinalSamples[0], dFinalSamples, hFinalSamples.size()*sizeof(hFinalSamples[0]), cudaMemcpyDeviceToHost));

  size_t totalSampledVertices = 0;
  for (auto s : hFinalSamples) {
    totalSampledVertices += (int)(s != INVALID_VERTEX);
  }

  std::cout << "totalSampledVertices " << totalSampledVertices << std::endl;
  if (opt->getFlag('check-results'))
    assert(check_result(csr, INVALID_VERTEX, samples, finalSampleSize, hFinalSamples));

  if (opt->getFlag('print-samples')) {
    for (size_t s = 0; s < hFinalSamples.size(); s += finalSampleSize) {
      std::cout << "Contents of sample " << s/finalSampleSize << " [";
      for(size_t v = s; v < s + finalSampleSize; v++)
        std::cout << hFinalSamples[v] << ", ";
      std::cout << "]" << std::endl;
    }
  }
  
  // std::cout << "GPU Time: " << gpu_time << " secs" << std::endl;
  std::cout << "End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  // std::cout << "Total " << N_HOPS << "-hop neighbors " << total_neighbors << std::endl;

  // std::cout << "Results are correct? " <<check_result(csr, additions_sizes, neighbors) << std::endl;
  // std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  // std::cout << "Time spent in Streams " << total_stream_time << std::endl;
}
