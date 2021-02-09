#include "testBase.h"

#include <stdlib.h>
#include <nlohmann/json.hpp>

#define VERTICES_PER_SAMPLE 105
#define NUM_CLUSTERS 20

nlohmann::json partitionsJson;

class SubGraphSample 
{
public:
  int *adjMatrixTotalLen;

  int adjMatrixLength;
  int adjMatrixPos;
  VertexID_t vertices[VERTICES_PER_SAMPLE*NUM_CLUSTERS];
  int adjacencyMatrixLen;
  int *adjacencyMatrixRow;
  int *adjacencyMatrixCol;
  int *adjacencyMatrixVal;
};

int * dRowStorage;
int * dColStorage;
int* dAdjMatrixTotalLen;

struct SubGraphSamplingAppI {
  __host__ __device__ int steps() {return 2;}

  __host__ __device__ 
  int stepSize(int k) {
    if (k == 0) return 1;
    return 1;
  }

  template<class SampleType>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx, 
                SampleType* sample, const float max_weight,
                const CSR::Edge* transitEdges, const float* transitEdgeWeights,
                const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
  {
    VertexID_t v1 = transits[0];

    if (step == 0) {
      ::atomicAdd(&sample->adjMatrixLength, numEdges);
      return v1;
    }

    for (int v2Idx = 0; v2Idx < VERTICES_PER_SAMPLE; v2Idx++) 
    //int v2Idx = neighbrID;
    {
      VertexID_t v2 = sample->vertices[v2Idx];
      bool hasEdge = csr->has_edge_logn(v1, v2);
      if (hasEdge) {
        int len = ::atomicAdd(&sample->adjacencyMatrixLen, 1) + sample->adjMatrixPos;
        //int cooIdx = step * NUM_SAMPLED_VERTICES + len;
        sample->adjacencyMatrixRow[len] = v1;
        sample->adjacencyMatrixCol[len] = v2;
        //sample->adjacencyMatrixVal[len] = 1.0f;
      }
    }

    return -1;
  }

  template<class SampleType, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool DECREASE_GM_LOADS>
  __device__ inline
  VertexID nextCached(int step, const VertexID transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                const CSR::Edge* transitEdges, const float* transitEdgeWeights,
                const EdgePos_t numEdges, const EdgePos_t neighbrID, 
                curandState* state, VertexID_t* cachedEdges, float* cachedWeights,
                bool* globalLoadBV)
  {
    EdgePos_t id = RandNumGen::rand_int(state, numEdges);
    if (CACHE_EDGES)
      return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(id, transitEdges, cachedEdges, globalLoadBV);
    return transitEdges[id];
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return AdjacencyMatrix;
  }

  __host__ EdgePos_t numSamples(CSR* graph)
  {
    return graph->get_n_vertices() / (VERTICES_PER_SAMPLE * NUM_CLUSTERS);
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return true;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    if (transitIdx == 0) {
      if (step == 1) {
        sample.adjMatrixPos = ::atomicAdd(sample.adjMatrixTotalLen, sample.adjMatrixLength);
      }
    }


    return sample.vertices[transitIdx];
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;

    for (int i = 0; i < VERTICES_PER_SAMPLE*NUM_CLUSTERS; i++) {
      VertexID_t v = sampleIdx * VERTICES_PER_SAMPLE*NUM_CLUSTERS + i;
      initialValue.push_back(v);
      sample.vertices[i] = v;
    }

    return initialValue;
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    sample.adjacencyMatrixLen = 0;
    sample.adjMatrixLength = 0;
    sample.adjMatrixTotalLen = dAdjMatrixTotalLen;
    sample.adjacencyMatrixRow = dRowStorage;
    sample.adjacencyMatrixCol = dColStorage;

    return sample;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph) { return VERTICES_PER_SAMPLE*NUM_CLUSTERS;}
};

#define RUNS 1
#define CHECK_RESULTS true

template<class SampleType, typename App>
bool checkSubGraphResult(NextDoorData<SampleType, App>& nextDoorData)
{
  //Check result by traversing all sampled neighbors and making
  //sure that if neighbors at kth-hop is an adjacent vertex of one
  //of the k-1th hop neighbors.
  CSR* csr = nextDoorData.csr;
  auto& initialSamples = nextDoorData.initialContents;
  auto finalSampleSize = getFinalSampleSize<SubGraphSamplingAppI>();
  auto& finalSamples = nextDoorData.hFinalSamples;
  auto INVALID_VERTEX = nextDoorData.INVALID_VERTEX;
  auto& samples = nextDoorData.samples;
  int maxSteps = 4;

  //First create the adjacency matrix.
  std::cout << "checking results" << std::endl;
  AdjMatrix adj_matrix;

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  size_t numNeighborsToSampleAtStep = 0;
  bool foundError = false;
  int sampleIdx = 0;
  for (SampleType& sample : samples) {
    //Go through all edges between two vertices and see if they exist in the graph
    for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
      VertexID_t v1 = sample.adjacencyMatrixRow[e];
      VertexID_t v2 = sample.adjacencyMatrixCol[e];

      if (!foundError && adj_matrix[v1].count(v2) == 0) {
        printf("Sample '%d': no edge '%d' -> '%d' in graph\n", sampleIdx, v1, v2);
        foundError = true;
      }
    }

    //Go through edges between two vertices in graph and see if they exist in sample
    for (int vidx1 = 0; vidx1 < VERTICES_PER_SAMPLE; vidx1++) {
      VertexID_t v1 = sample.vertices[vidx1];
      for (int vidx2 = 0; vidx2 < VERTICES_PER_SAMPLE; vidx2++) {
        VertexID_t v2 = sample.vertices[vidx2];

        if (adj_matrix[v1].count(v2) == 1 and v1 != v2) {
          bool foundEdge = false;
          //Edge in Graph. Check if it is in Sample.
          for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
            if (sample.adjacencyMatrixRow[e] == v1 && sample.adjacencyMatrixCol[e] == v2) {
              foundEdge = true;
              break;
            }
          }

          if (!foundEdge) {
            if (!foundError) {
              printf("Sample '%d': Edge '%d'->'%d' exists in graph but not in sample of length '%d'\n", sampleIdx, v1, v2, sample.adjacencyMatrixLen);
            }
            foundError = true;
          }
        }
      }
    }

    sampleIdx++;
  }

  if (foundError) return false;
  return true;
}

void foo(const char* graph_file, const char* graph_type, const char* graph_format, 
  const int nruns, const bool chk_results, const bool print_samples,
  const char* kernelType, const bool enableLoadBalancing,
  bool (*checkResultsFunc)(NextDoorData<SubGraphSample, SubGraphSamplingAppI>&))
{
  Graph graph; 
  CSR* csr;
  if ((csr = loadGraph(graph, (char*)graph_file, (char*)"adj-list", (char*)"text")) == nullptr) {
    return;
  }

  std::cout << "Graph has " <<graph.get_n_edges () << " edges and " << 
      graph.get_vertices ().size () << " vertices " << std::endl;
  
  std::string parts_file = "/mnt/homes/abhinav/nextdoor-experiments/cluster_gcn/reddit-parts-txt";
  std::ifstream partitionsFile(parts_file);
  partitionsFile >> partitionsJson;
  partitionsFile.close();
  size_t maximumSize = 0;

  for (auto& item : partitionsJson.items()) {
    maximumSize = std::max(item.value().size(), maximumSize);
  }

  std::cout << "maximumSize " << maximumSize << std::endl; 

  GPUCSRPartition gpuCSRPartition = transferCSRToGPU(csr);
  
  NextDoorData<SubGraphSample, SubGraphSamplingAppI> nextDoorData;
  nextDoorData.csr = csr;
  nextDoorData.gpuCSRPartition = gpuCSRPartition;
  CHK_CU(cudaMalloc(&dAdjMatrixTotalLen, sizeof(int)));
  CHK_CU(cudaMemset(dAdjMatrixTotalLen, 0, sizeof(int)));
  CHK_CU(cudaMalloc(&dRowStorage, sizeof(VertexID_t) * graph.get_n_edges()));
  CHK_CU(cudaMalloc(&dColStorage, sizeof(VertexID_t) * graph.get_n_edges()));
  //CHK_CU(cudaMalloc(&SubGraphSample::rowStorage, sizeof(VertexID_t) * graph.get_n_edges()));
  allocNextDoorDataOnGPU<SubGraphSample, SubGraphSamplingAppI>(csr, nextDoorData);
  

  for (int i = 0; i < 1; i++) {
    if (strcmp(kernelType, "TransitParallel") == 0)
      doTransitParallelSampling<SubGraphSample, SubGraphSamplingAppI>(csr, gpuCSRPartition, nextDoorData, enableLoadBalancing);
    else if (strcmp(kernelType, "SampleParallel") == 0)
      doSampleParallelSampling<SubGraphSample, SubGraphSamplingAppI>(csr, gpuCSRPartition, nextDoorData);
    else
      abort();
  }

  // int hTotalLen = 0;

  // CHK_CU(cudaMemcpy(&hTotalLen, dAdjMatrixTotalLen, sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout<<hTotalLen<<std::endl;
}

#define SubGraphAPP_TEST(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(SubGraphSampling, TestName) { \
    foo(Path, (char*)"adj-list", (char*)"text", 1, false, false, "SampleParallel", false, chkResultsFunc);\
  }

// APP_TEST(DeepWalk, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "TransitParallel") 
// APP_TEST(DeepWalk, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, PpiTP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, PpiSP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "SampleParallel")
SubGraphAPP_TEST(RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)
//APP_TEST(SubGraphSample, SubGraph, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
// SubGraphAPP_TEST(RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
// //APP_TEST(SubGraph, DeepWalk, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult, "TransitParallel", true)
// SubGraphAPP_TEST(LiveJournalTP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)
// //APP_TEST(SubGraphSample, SubGraph, LiveJournalLB, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
// SubGraphAPP_TEST(LiveJournalSP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
// SubGraphAPP_TEST(OrkutTP, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)
// //APP_TEST(SubGraphSample, SubGraph, OrkutLB, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
// SubGraphAPP_TEST(OrkutSP, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)