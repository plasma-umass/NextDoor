#include "testBase.h"

#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64

__host__ __device__ int steps() {return NUM_LAYERS;}

__host__ __device__ 
int stepSize(int k) {
  return NUM_SAMPLED_VERTICES;
}

class LayerSample 
{
public:
  int adjacencyMatrixLen[NUM_LAYERS];
  int adjacencyMatrixRow[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
  int adjacencyMatrixCol[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
  int adjacencyMatrixVal[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
};

template<class SampleType>
__device__ inline
VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx, 
              SampleType* sample, const float max_weight,
              const CSR::Edge* transitEdges, const float* transitEdgeWeights,
              const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
{
  EdgePos_t id = RandNumGen::rand_int(state, csr->get_n_vertices());
  for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
    VertexID transit = transits[i];
    bool hasEdge = csr->has_edge_logn(transit, id);
    if (hasEdge) {
      int len = ::atomicAdd(&sample->adjacencyMatrixLen[step], 1);
      //int cooIdx = step * NUM_SAMPLED_VERTICES + len;
      sample->adjacencyMatrixRow[step][len] = i;
      sample->adjacencyMatrixCol[step][len] = neighbrID;
      sample->adjacencyMatrixVal[step][len] = 1.0f;
    }
  }

  return id;
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
  return SamplingType::CollectiveNeighborhood;
}

__host__ __device__ OutputFormat outputFormat()
{
  return AdjacencyMatrix;
}

__host__ EdgePos_t numSamples(CSR* graph)
{
  return graph->get_n_vertices() / VERTICES_PER_SAMPLE / 10;
}

__host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph)
{
  std::vector<VertexID_t> initialValue;

  for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
    initialValue.push_back(sampleIdx * VERTICES_PER_SAMPLE + i);
  }

  return initialValue;
}

__host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
{
  return VERTICES_PER_SAMPLE;
}

#define RUNS 1
#define CHECK_RESULTS true

//APP_TEST(KHop, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", false)
//APP_TEST(LayerSample, FastGCN, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "SampleParallel", false)
// APP_TEST(KHop, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST(KHop, LiveJournalTP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", false)
// APP_TEST(KHop, LiveJournalLB, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", true)
APP_TEST(LayerSample, FastGCN, LiveJournalSP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "SampleParallel", false)
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

