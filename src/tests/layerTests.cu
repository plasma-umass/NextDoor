#include "testBase.h"

__host__ __device__ int steps() {return 1;}

__host__ __device__ 
int stepSize(int k) {
  return 1;
}

__device__ inline
VertexID next(int step, const VertexID transit, const VertexID sample, 
              const float max_weight,
              const CSR::Edge* transitEdges, const float* transitEdgeWeights,
              const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
{
  EdgePos_t id = RandNumGen::rand_int(state, numEdges);
  return transitEdges[id];
}

template<int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool DECREASE_GM_LOADS>
__device__ inline
VertexID nextCached(int step, const VertexID transit, const VertexID sample, 
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
  return SamplingType::CollectiveNeighborhood;
}

__host__ __device__ OutputFormat outputFormat()
{
  return AdjacencyMatrix;
}

#define VERTICES_PER_SAMPLE 2

__host__ EdgePos_t numSamples(CSR* graph)
{
  return graph->get_n_vertices() / VERTICES_PER_SAMPLE;
}

__host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph)
{
  std::vector<VertexID_t> initialValue;

  for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
    initialValue.push_back(sampleIdx * VERTICES_PER_SAMPLE + i);
  }
}

#define RUNS 1
#define CHECK_RESULTS true

//APP_TEST(KHop, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", false)
APP_TEST(KHop, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "SampleParallel", false)
// APP_TEST(KHop, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST(KHop, LiveJournalTP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", false)
// APP_TEST(KHop, LiveJournalLB, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST(KHop, LiveJournalSP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "SampleParallel", false)
// APP_TEST(KHop, OrkutTP, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", false)
// APP_TEST(KHop, OrkutLB, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST(KHop, OrkutSP, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, "SampleParallel", false)

// APP_TEST(KHop, Citeseer, GRAPH_PATH"/citeseer.graph", 1, true, "TransitParallel")
// APP_TEST(KHop, Mico, GRAPH_PATH"/micro.graph", 1, false, "TransitParallel")
// APP_TEST(KHop, Reddit, GRAPH_PATH"/reddit_sampled_matrix", 1, false, "TransitParallel")


// TEST(KHop, Citeseer) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/citeseer.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Mico) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/micro.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Reddit) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/reddit_sampled_matrix", "adj-list", "text", CHECK_RESULTS, false));
// }

