#include "testBase.h"

#include <stdlib.h>

#include "../apps/mvs/mvsSampling.cu"

#define RUNS 1
#define CHECK_RESULTS true
#include "../check_results.cu"

template<class SampleType, typename App>
bool checkMVSResult(NextDoorData<SampleType, App>& nextDoorData)
{
  //Check result by traversing all sampled neighbors and making
  //sure that if neighbors at kth-hop is an adjacent vertex of one
  //of the k-1th hop neighbors.
  CSR* csr = nextDoorData.csr;
  // auto& initialSamples = nextDoorData.initialContents;
  auto finalSampleSize = getFinalSampleSize<MVSSamplingApp>();
  // auto& finalSamples = nextDoorData.hFinalSamples;
  // auto INVALID_VERTEX = nextDoorData.INVALID_VERTEX;
  auto& samples = nextDoorData.samples;
  storagePosition = 0;

  //First create the adjacency matrix.
  std::cout << "checking results" << std::endl;
  AdjMatrix adj_matrix(csr->get_n_vertices(), std::unordered_set<VertexID> ());

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  bool foundError = false;
  int* hRowStorage = new int[csr->get_n_edges()*DIVUP(MVSSamplingApp().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())];
  int* hColStorage = new int[csr->get_n_edges()*DIVUP(MVSSamplingApp().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())];

  CHK_CU(cudaMemcpy(hRowStorage, dRowStorage, csr->get_n_edges()*DIVUP(MVSSamplingApp().numSamples(csr)*VERTICES_PER_SAMPLE, 
         csr->get_n_vertices())*sizeof(CSR::Edge), cudaMemcpyDeviceToHost));
  CHK_CU(cudaMemcpy(hColStorage, dColStorage, csr->get_n_edges()*DIVUP(MVSSamplingApp().numSamples(csr)*VERTICES_PER_SAMPLE, 
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
  nextDoorData.devices = {0};
  CHK_CU(cudaMalloc(&dRowStorage, sizeof(VertexID_t) * graph.get_n_edges()*DIVUP(MVSSamplingApp().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())));
  CHK_CU(cudaMalloc(&dColStorage, sizeof(VertexID_t) * graph.get_n_edges()*DIVUP(MVSSamplingApp().numSamples(csr)*VERTICES_PER_SAMPLE, csr->get_n_vertices())));
  
  std::vector<GPUCSRPartition> gpuCSRPartitions = transferCSRToGPUs(nextDoorData, csr);
  nextDoorData.gpuCSRPartitions = gpuCSRPartitions;
  //CHK_CU(cudaMalloc(&SubGraphSample::rowStorage, sizeof(VertexID_t) * graph.get_n_edges()));
  allocNextDoorDataOnGPU<MVSSample, MVSSamplingApp>(csr, nextDoorData);
  

  for (int i = 0; i < RUNS; i++) {
    if (strcmp(kernelType, "TransitParallel") == 0)
      doTransitParallelSampling<MVSSample, MVSSamplingApp>(csr, nextDoorData, enableLoadBalancing);
    else if (strcmp(kernelType, "SampleParallel") == 0)
      doSampleParallelSampling<MVSSample, MVSSamplingApp>(csr, nextDoorData);
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
  freeDeviceData(nextDoorData);

  return toRet;
}

#define MVSAPP_TEST(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(MVS, TestName) { \
    EXPECT_TRUE(foo(Path, (char*)"adj-list", (char*)"text", 1, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc));\
  }

#define MVSAPP_TEST_BINARY(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing)\
  TEST(MVS, TestName) { \
  bool b = foo(Path, "edge-list", "binary", Runs, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc);\
  EXPECT_TRUE(b);\
}


MVSAPP_TEST_BINARY(LiveJournalSP, LJ1_PATH, RUNS, true, checkMVSResult, "SampleParallel", false)
MVSAPP_TEST_BINARY(LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", true)
MVSAPP_TEST_BINARY(LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", false)

MVSAPP_TEST_BINARY(OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "SampleParallel", false)
MVSAPP_TEST_BINARY(OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", true)
MVSAPP_TEST_BINARY(OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", false)

MVSAPP_TEST_BINARY(PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "SampleParallel", false)
MVSAPP_TEST_BINARY(PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", true)
MVSAPP_TEST_BINARY(PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", false)

MVSAPP_TEST_BINARY(RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "SampleParallel", false)
MVSAPP_TEST_BINARY(RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", true)
MVSAPP_TEST_BINARY(RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", false)

MVSAPP_TEST_BINARY(PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "SampleParallel", false)
MVSAPP_TEST_BINARY(PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", true)
MVSAPP_TEST_BINARY(PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkMVSResult, "TransitParallel", false)