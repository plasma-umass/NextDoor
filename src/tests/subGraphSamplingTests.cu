#include "testBase.h"

#include <stdlib.h>

#define VERTICES_IN_CLUSTERS 16
#define CLUSTERS_IN_SAMPLE 2
#define VERTICES_PER_SAMPLE (VERTICES_IN_CLUSTERS*CLUSTERS_IN_SAMPLE)

class SubGraphSample 
{
public:
  int *adjMatrixTotalLen;

  int adjMatrixLength;
  int adjMatrixPos;
  VertexID_t vertices[VERTICES_IN_CLUSTERS*CLUSTERS_IN_SAMPLE];
  int adjacencyMatrixLen;
  int *adjacencyMatrixRow;
  int *adjacencyMatrixCol;
};

int * dRowStorage;
int * dColStorage;
int* dAdjMatrixTotalLen;
std::vector<std::vector<VertexID_t>> clusters;

struct SubGraphSamplingAppI {
  __host__ __device__ int steps() {return 2;}

  __host__ __device__ 
  int stepSize(int k) {
    if (k == 0) return 1;
    return VERTICES_PER_SAMPLE;
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    VertexID_t v1 = transits[0];
    if (step == 0) {
      ::atomicAdd(&sample->adjMatrixLength, numEdges);
      return v1;
    }

    int v2Idx = neighbrID; //for (int v2Idx = 0; v2Idx < VERTICES_PER_SAMPLE; v2Idx++) //
    {
      VertexID_t v2 = sample->vertices[v2Idx];
      bool hasEdge = utils::binarySearch(transitEdges, v2, numEdges);
      // if (sampleIdx == 1929) {
      //   printf("sampleIdx %d v1 %d v2 %d hasEdge %d v2Idx %d\n", sampleIdx, v1, v2, hasEdge, v2Idx);
      // }

      if (hasEdge) {
        int len = ::atomicAdd(&sample->adjacencyMatrixLen, 1) + sample->adjMatrixPos;
        //int cooIdx = step * NUM_SAMPLED_VERTICES + len;
        sample->adjacencyMatrixRow[len] = v1;
        sample->adjacencyMatrixCol[len] = v2;
        //sample->adjacencyMatrixVal[len] = 1.0f;
        // if (sampleIdx == 49 && v1==1569 && v2==1570) { 
        //   printf("sampleIdx %d v1 %d v2 %d hasEdge %d v2Idx %d len %d %d %d %d\n", sampleIdx, v1, v2, hasEdge, v2Idx, len, sample->adjacencyMatrixLen, 
        //          sample->adjMatrixLength, sample->adjMatrixPos);
        // }
        // if (sampleIdx == 1929 || (len >= 32765 && len <= 32766)) { //v1==76921 && v2==205491 && 
        //   printf("sampleIdx %d v1 %d v2 %d hasEdge %d v2Idx %d len %d %d %d\n", sampleIdx, v1, v2, hasEdge, v2Idx, len, sample->adjacencyMatrixLen, sample->adjMatrixLength);
        // }
      }

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
    return (graph->get_n_edges() > 100000000) ? 800000 : min(1500000, (graph->get_n_vertices()*8)/VERTICES_IN_CLUSTERS);
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return true;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    if (transitIdx == 0 && step == 1) {
      sample.adjMatrixPos = ::atomicAdd(sample.adjMatrixTotalLen, sample.adjMatrixLength);
    }

    VertexID_t v = sample.vertices[transitIdx];

    return v;
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;
    int idx = 0;
    for (int i = 0; i < CLUSTERS_IN_SAMPLE; i++) {
      VertexID_t clusterIdx = (sampleIdx * CLUSTERS_IN_SAMPLE + i) % clusters.size();//rand() % graph->get_n_vertices();
      //initialValue.insert(initialValue.begin(), clusters[clusterIdx].begin(), clusters[clusterIdx].end());
      for (auto v : clusters[clusterIdx]) {
        sample.vertices[idx] = v;
        idx++;
        initialValue.push_back(v);
      }
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

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph) { return VERTICES_PER_SAMPLE;}
};

#define RUNS 1
#define CHECK_RESULTS false
#include "../check_results.cu"

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
  AdjMatrix adj_matrix(csr->get_n_vertices(), std::unordered_set<VertexID> ());

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  size_t numNeighborsToSampleAtStep = 0;
  bool foundError = false;
  int sampleIdx = 0;
  int* hRowStorage = new int[csr->get_n_edges()*DIVUP(App().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())];
  int* hColStorage = new int[csr->get_n_edges()*DIVUP(App().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())];

  CHK_CU(cudaMemcpy(hRowStorage, dRowStorage, csr->get_n_edges()*DIVUP(App().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())*sizeof(CSR::Edge), cudaMemcpyDeviceToHost));
  CHK_CU(cudaMemcpy(hColStorage, dColStorage, csr->get_n_edges()*DIVUP(App().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())*sizeof(CSR::Edge), cudaMemcpyDeviceToHost));

  #pragma omp parallel for shared(foundError)
  for (int sampleIdx = 0; sampleIdx < samples.size(); sampleIdx++) {
    auto sample = samples[sampleIdx];
    //Go through all edges between two vertices and see if they exist in the graph
    for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
      VertexID_t v1 = hRowStorage[e + sample.adjMatrixPos];
      VertexID_t v2 = hColStorage[e + sample.adjMatrixPos];

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
          for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
            if ( hRowStorage[e + sample.adjMatrixPos]== v1 && hColStorage[e + sample.adjMatrixPos] == v2) {
              foundEdge = true;
              break;
            }
          }

          if (!foundEdge) {
            if (!foundError) {
              printf("Sample '%d': Edge '%d'->'%d' exists in graph but not in sample of length '%d' sample.adjMatrixPos '%d'\n", sampleIdx, v1, v2, sample.adjacencyMatrixLen, sample.adjMatrixPos);
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
  bool (*checkResultsFunc)(NextDoorData<SubGraphSample, SubGraphSamplingAppI>&))
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
  clusters = std::vector<std::vector<VertexID_t>>(csr->get_n_vertices()/VERTICES_IN_CLUSTERS);
  for (int clusterIdx = 0; clusterIdx < csr->get_n_vertices()/VERTICES_IN_CLUSTERS; clusterIdx++) {
    for (int v = 0; v < VERTICES_IN_CLUSTERS; v++) {
      clusters[clusterIdx].push_back(clusterIdx * VERTICES_IN_CLUSTERS + v);
    }
  }
  printCudaMemInfo();
  GPUCSRPartition gpuCSRPartition = transferCSRToGPU(csr);
  
  NextDoorData<SubGraphSample, SubGraphSamplingAppI> nextDoorData;
  nextDoorData.csr = csr;
  nextDoorData.gpuCSRPartition = gpuCSRPartition;
  CHK_CU(cudaMalloc(&dAdjMatrixTotalLen, sizeof(int)));
  CHK_CU(cudaMemset(dAdjMatrixTotalLen, 0, sizeof(int)));
  CHK_CU(cudaMalloc(&dRowStorage, sizeof(VertexID_t) * graph.get_n_edges()*DIVUP(SubGraphSamplingAppI().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())));
  CHK_CU(cudaMalloc(&dColStorage, sizeof(VertexID_t) * graph.get_n_edges()*DIVUP(SubGraphSamplingAppI().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())));
  //CHK_CU(cudaMalloc(&SubGraphSample::rowStorage, sizeof(VertexID_t) * graph.get_n_edges()));
  allocNextDoorDataOnGPU<SubGraphSample, SubGraphSamplingAppI>(csr, nextDoorData);
  
  for (int i = 0; i < RUNS; i++) {
    if (strcmp(kernelType, "TransitParallel") == 0)
      doTransitParallelSampling<SubGraphSample, SubGraphSamplingAppI>(csr, gpuCSRPartition, nextDoorData, enableLoadBalancing);
    else if (strcmp(kernelType, "SampleParallel") == 0)
      doSampleParallelSampling<SubGraphSample, SubGraphSamplingAppI>(csr, gpuCSRPartition, nextDoorData);
    else
      abort();
  }

  getFinalSamples(nextDoorData);

  // int hTotalLen = 0;

  // CHK_CU(cudaMemcpy(&hTotalLen, dAdjMatrixTotalLen, sizeof(int), cudaMemcpyDeviceToHost));
  // std::cout<<hTotalLen<<std::endl;
  bool toRet = false;
  if (chk_results) {
    toRet = checkResultsFunc(nextDoorData);
  }

  CHK_CU(cudaFree(dRowStorage));
  CHK_CU(cudaFree(dColStorage));
  CHK_CU(cudaFree(dAdjMatrixTotalLen));
  freeDeviceData(nextDoorData);
  return true;
}

#define SubGraphAPP_TEST(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(ClusterGCNSampling, TestName) { \
    EXPECT_TRUE(foo(Path, (char*)"adj-list", (char*)"text", 1, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc));\
  }

#define SubGraphAPP_TEST_BINARY(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing)\
  TEST(ClusterGCNSampling, TestName) { \
  bool b = foo(Path, "edge-list", "binary", Runs, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc);\
  EXPECT_TRUE(b);\
}

SubGraphAPP_TEST_BINARY(LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
SubGraphAPP_TEST_BINARY(LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
SubGraphAPP_TEST_BINARY(LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)

SubGraphAPP_TEST_BINARY(OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
SubGraphAPP_TEST_BINARY(OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
SubGraphAPP_TEST_BINARY(OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)

SubGraphAPP_TEST_BINARY(PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
SubGraphAPP_TEST_BINARY(PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
SubGraphAPP_TEST_BINARY(PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)

SubGraphAPP_TEST_BINARY(RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
SubGraphAPP_TEST_BINARY(RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
SubGraphAPP_TEST_BINARY(RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)

SubGraphAPP_TEST_BINARY(PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "SampleParallel", false)
SubGraphAPP_TEST_BINARY(PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
SubGraphAPP_TEST_BINARY(PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", false)