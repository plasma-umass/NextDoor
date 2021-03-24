#include "testBase.h"

#include <stdlib.h>
#include "../apps/clustergcn/clusterGCNSampling.cu"
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
  // auto& initialSamples = nextDoorData.initialContents;
  auto finalSampleSize = getFinalSampleSize<SubGraphSamplingAppI>();
  // auto& finalSamples = nextDoorData.hFinalSamples;
  // auto INVALID_VERTEX = nextDoorData.INVALID_VERTEX;
  auto& samples = nextDoorData.samples;
  // int maxSteps = 4;

  //First create the adjacency matrix.
  std::cout << "checking results" << std::endl;
  AdjMatrix adj_matrix(csr->get_n_vertices(), std::unordered_set<VertexID> ());

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  // size_t numNeighborsToSampleAtStep = 0;
  bool foundError = false;
  // int sampleIdx = 0;
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


#define SubGraphAPP_TEST(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(ClusterGCN, TestName) { \
    EXPECT_TRUE(foo(Path, (char*)"adj-list", (char*)"text", 1, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc));\
  }

#define SubGraphAPP_TEST_BINARY(TestName,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing)\
  TEST(ClusterGCN, TestName) { \
  bool b = foo(Path, "edge-list", "binary", Runs, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc);\
  EXPECT_TRUE(b);\
}

SubGraphAPP_TEST_BINARY(LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)

SubGraphAPP_TEST_BINARY(OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)

SubGraphAPP_TEST_BINARY(PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)

SubGraphAPP_TEST_BINARY(RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)

SubGraphAPP_TEST_BINARY(PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSubGraphResult, "TransitParallel", true)
