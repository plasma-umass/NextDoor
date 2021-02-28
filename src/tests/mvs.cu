#include "testBase.h"

#include <stdlib.h>

#define VERTICES_IN_BATCH 32
#define VERTICES_PER_SAMPLE (VERTICES_IN_BATCH)
#define NUM_SAMPLES 1500000

class MVSSample 
{
public:
  int *row;
  int *col;
  int length;
  int posInStorage;
  VertexID_t vertices[VERTICES_PER_SAMPLE];
};

int *dRowStorage = nullptr;
int *dColStorage = nullptr;
int storagePosition;
std::vector<std::vector<VertexID_t>> batches;

struct MVSSamplingApp {
  __host__ __device__ int steps() {return 1;}

  __host__ __device__ 
  int stepSize(int k) {
    return 32;
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    for (int e = neighbrID; e < numEdges; e += stepSize(0)) {
      int p = ::atomicAdd(&sample->length, 1);
      sample->row[p] = transits[0];
      sample->col[p] = transitEdges[e];
      // if (transits[0] == 0 and sampleIdx == 0) {
      //   printf("e %d numEdges %d\n", e, numEdges);
      // }
    }
    return -1;
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
    return NUM_SAMPLES;
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;
    EdgePos_t totalEdges = 0;
    size_t storageStartPos = storagePosition;
    for (int i = 0; i < VERTICES_IN_BATCH; i++) {
      VertexID_t v = (sampleIdx * VERTICES_IN_BATCH + i) % graph->get_n_vertices();
      initialValue.push_back(v);
      totalEdges += graph->n_edges_for_vertex(v);
      sample.vertices[i] = v;
    }

    storagePosition += totalEdges;

    sample.row = dRowStorage + storageStartPos;
    sample.col = dColStorage + storageStartPos;
    sample.posInStorage = storageStartPos;
    sample.length = 0;
    return initialValue;
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    return sample;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph) { return VERTICES_PER_SAMPLE;}
};

#define RUNS 1
#define CHECK_RESULTS true

template<class SampleType, typename App>
bool checkMVSResult(NextDoorData<SampleType, App>& nextDoorData)
{
  //Check result by traversing all sampled neighbors and making
  //sure that if neighbors at kth-hop is an adjacent vertex of one
  //of the k-1th hop neighbors.
  CSR* csr = nextDoorData.csr;
  auto& initialSamples = nextDoorData.initialContents;
  auto finalSampleSize = getFinalSampleSize<MVSSamplingApp>();
  auto& finalSamples = nextDoorData.hFinalSamples;
  auto INVALID_VERTEX = nextDoorData.INVALID_VERTEX;
  auto& samples = nextDoorData.samples;
  int maxSteps = 4;
  storagePosition = 0;

  //First create the adjacency matrix.
  std::cout << "checking results" << std::endl;
  AdjMatrix adj_matrix(csr->get_n_vertices(), std::unordered_set<VertexID> ());

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  size_t numNeighborsToSampleAtStep = 0;
  bool foundError = false;
  int sampleIdx = 0;
  int* hRowStorage = new int[csr->get_n_edges()*DIVUP(NUM_SAMPLES*VERTICES_PER_SAMPLE, csr->get_n_vertices())];
  int* hColStorage = new int[csr->get_n_edges()*DIVUP(NUM_SAMPLES*VERTICES_PER_SAMPLE, csr->get_n_vertices())];

  CHK_CU(cudaMemcpy(hRowStorage, dRowStorage, csr->get_n_edges()*DIVUP(NUM_SAMPLES*VERTICES_PER_SAMPLE, 
         csr->get_n_vertices())*sizeof(CSR::Edge), cudaMemcpyDeviceToHost));
  CHK_CU(cudaMemcpy(hColStorage, dColStorage, csr->get_n_edges()*DIVUP(NUM_SAMPLES*VERTICES_PER_SAMPLE, 
         csr->get_n_vertices())*sizeof(CSR::Edge), cudaMemcpyDeviceToHost));

  #pragma omp parallel for shared(foundError)
  for (int sampleIdx = 0; sampleIdx < samples.size(); sampleIdx++) {
    auto sample = samples[sampleIdx];
    //Go through all edges between two vertices and see if they exist in the graph
    for (int e = 0; e < sample.length; e++) {
      VertexID_t v1 = hRowStorage[e + sample.posInStorage];
      VertexID_t v2 = hColStorage[e + sample.posInStorage];

      if (!foundError && adj_matrix[v1].count(v2) == 0) {
        printf("Sample '%d': no edge '%d' -> '%d' in graph\n", sampleIdx, v1, v2);
        foundError = true;
      }
    }

    // if (sampleIdx == 49) {
    // for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
    //   VertexID_t v1 = hRowStorage[e + sample.adjMatrixPos];
    //   VertexID_t v2 = hColStorage[e + sample.adjMatrixPos];

    //   printf("Sample '%d': '%d' -> '%d' at '%d'\n", sampleIdx, v1, v2, e);
    // } 
    // }

    //Go through edges between two vertices in graph and see if they exist in sample
    for (int vidx1 = 0; vidx1 < VERTICES_PER_SAMPLE; vidx1++) {
      VertexID_t v1 = sample.vertices[vidx1];
      for (int vidx2 = 0; vidx2 < VERTICES_PER_SAMPLE; vidx2++) {
        VertexID_t v2 = sample.vertices[vidx2];

        if (adj_matrix[v1].count(v2) == 1 and v1 != v2) {
          bool foundEdge = false;
          //Edge in Graph. Check if it is in Sample.
          for (int e = 0; e < sample.length; e++) {
            if (hRowStorage[e + sample.posInStorage] == v1 && hColStorage[e + sample.posInStorage] == v2) {
              foundEdge = true;
              break;
            }
          }

          if (!foundEdge) {
            if (!foundError) {
              printf("Sample '%d': Edge '%d'->'%d' exists in graph but not in sample of length '%d' sample.adjMatrixPos '%d'\n", sampleIdx, v1, v2, sample.length, sample.posInStorage);
            }
            foundError = true;
          }
        }
      }
    }
  }

  printf("Results Checked? %d\n", !foundError);
  if (foundError) return false;
  return true;
}

bool foo(const char* graph_file, const char* graph_type, const char* graph_format, 
  const int nruns, const bool chk_results, const bool print_samples,
  const char* kernelType, const bool enableLoadBalancing,
  bool (*checkResultsFunc)(NextDoorData<MVSSample, MVSSamplingApp>&))
{
  Graph graph; 
  CSR* csr;
  if ((csr = loadGraph(graph, (char*)graph_file, (char*)graph_type, (char*)graph_format)) == nullptr) {
    return false;
  }

  std::cout << "Graph has " <<graph.get_n_edges () << " edges and " << 
      graph.get_vertices ().size () << " vertices " << std::endl;
  
  // std::string parts_file = "/mnt/homes/abhinav/nextdoor-experiments/cluster_gcn/reddit-parts-txt";
  // std::ifstream partitionsFile(parts_file);
  // partitionsFile >> partitionsJson;
  // partitionsFile.close();
  // size_t maximumSize = 0;

  // for (auto& item : partitionsJson.items()) {
  //   maximumSize = std::max(item.value().size(), maximumSize);
  // }

  // std::cout << "maximumSize " << maximumSize << std::endl; 
  
  //Create Clusters
  // batches = std::vector<std::vector<VertexID_t>>(csr->get_n_vertices()/VERTICES_IN_BATCH);
  // for (int batchIdx = 0; batchIdx < csr->get_n_vertices()/VERTICES_IN_BATCH; batchIdx++) {
  //   for (int v = 0; v < VERTICES_IN_BATCH; v++) {
  //     batches[batchIdx].push_back(batchIdx * VERTICES_IN_BATCH + v);
  //   }
  // }

  NextDoorData<MVSSample, MVSSamplingApp> nextDoorData;
  nextDoorData.csr = csr;
  CHK_CU(cudaMalloc(&dRowStorage, sizeof(VertexID_t) * graph.get_n_edges()*DIVUP(NUM_SAMPLES*VERTICES_PER_SAMPLE, csr->get_n_vertices())));
  CHK_CU(cudaMalloc(&dColStorage, sizeof(VertexID_t) * graph.get_n_edges()*DIVUP(NUM_SAMPLES*VERTICES_PER_SAMPLE, csr->get_n_vertices())));
  
  GPUCSRPartition gpuCSRPartition = transferCSRToGPU(csr);
  nextDoorData.gpuCSRPartition = gpuCSRPartition;
  //CHK_CU(cudaMalloc(&SubGraphSample::rowStorage, sizeof(VertexID_t) * graph.get_n_edges()));
  allocNextDoorDataOnGPU<MVSSample, MVSSamplingApp>(csr, nextDoorData);
  

  for (int i = 0; i < RUNS; i++) {
    if (strcmp(kernelType, "TransitParallel") == 0)
      doTransitParallelSampling<MVSSample, MVSSamplingApp>(csr, gpuCSRPartition, nextDoorData, enableLoadBalancing);
    else if (strcmp(kernelType, "SampleParallel") == 0)
      doSampleParallelSampling<MVSSample, MVSSamplingApp>(csr, gpuCSRPartition, nextDoorData);
    else
      abort();
  }

  getFinalSamples(nextDoorData);

  // int hTotalLen = 0;

  // CHK_CU(cudaMemcpy(&hTotalLen, dAdjMatrixTotalLen, sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout<<hTotalLen<<std::endl;

  bool toRet = true;
  if (chk_results) {
    toRet = checkResultsFunc(nextDoorData);
  }

  CHK_CU(cudaFree(dRowStorage));
  CHK_CU(cudaFree(dColStorage));

  return toRet;
}

#define MVSAPP_TEST(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(MVSSampling, TestName) { \
    EXPECT_TRUE(foo(Path, (char*)"adj-list", (char*)"text", 1, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc));\
  }

#define MVSAPP_TEST_BINARY(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing)\
  TEST(MVSSampling, TestName) { \
  bool b = foo(Path, "edge-list", "binary", Runs, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc);\
  EXPECT_TRUE(b);\
}


// APP_TEST(DeepWalk, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "TransitParallel") 
// APP_TEST(DeepWalk, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, PpiTP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, PpiSP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "SampleParallel")
//SubGraphAPP_TEST(RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
//SubGraphAPP_TEST(RedditTP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)
// MVSAPP_TEST(RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", true)
// MVSAPP_TEST_BINARY(LiveJournalSP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, 
                  //  checkMVSResult, "SampleParallel", false)
MVSAPP_TEST_BINARY(LiveJournalLB, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, 
                    checkMVSResult, "TransitParallel", true)
//MVSAPP_TEST_BINARY(LiveJournalTP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", false)

//APP_TEST(SubGraphSample, SubGraph, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
// SubGraphAPP_TEST(RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
// //APP_TEST(SubGraph, DeepWalk, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult, "TransitParallel", true)
// SubGraphAPP_TEST(LiveJournalTP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)
// //APP_TEST(SubGraphSample, SubGraph, LiveJournalLB, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
// SubGraphAPP_TEST(LiveJournalSP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
// SubGraphAPP_TEST(OrkutTP, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)
// //APP_TEST(SubGraphSample, SubGraph, OrkutLB, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
// SubGraphAPP_TEST(OrkutSP, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)

/**
 * [==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from SubGraphSampling
[ RUN      ] SubGraphSampling.LiveJournalSP
Graph Binary Loaded
Graph has 68555726 edges and 4847569 vertices 
Final Size of each sample: 33
Maximum Neighbors Sampled at each step: 32
Number of Samples: 1500000
2002:free memory 4006215680
Maximum Threads Per Kernel: 5242880
2023:free memory 3442081792
SampleParallel: End to end time 0.192252 secs
checking results
* [==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from SubGraphSampling
[ RUN      ] SubGraphSampling.LiveJournalTP
Graph Binary Loaded
Graph has 68555726 edges and 4847569 vertices 
Final Size of each sample: 33
Maximum Neighbors Sampled at each step: 32
Number of Samples: 1500000
2002:free memory 4006215680
Maximum Threads Per Kernel: 5242880
2023:free memory 3442081792
step 0
step 1
Transit Parallel: End to end time 0.11525 secs
InversionTime: 0.012413, LoadBalancingTime: 0, GridKernelTime: 0, ThreadBlockKernelTime: 0, SubWarpKernelTime: 0, IdentityKernelTime: 0
checking results

 * 
*/