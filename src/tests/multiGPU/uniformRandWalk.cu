#include "testBase.h"

__host__ __device__ int steps() {return 3;}

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
  EdgePos_t x = RandNumGen::rand_int(state, numEdges);
  if (CACHE_EDGES)
    return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(x, transitEdges, cachedEdges, globalLoadBV);
  else 
    return transitEdges[x];
}

__host__ int samplingType()
{
  return SamplingType::Individual;
}

//nvprof bin/test_rw_10.2_x86_64 by-pass --graph-file=/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix --walks-per-node=1 --walk-length=10 --walk-mode=0

//APP_TEST(UniformRandWalk, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "TransitParallel") 
// APP_TEST(UniformRandWalk, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(UniformRandWalk, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, false, "TransitParallel")
// APP_TEST(UniformRandWalk, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(UniformRandWalk, PpiTP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "TransitParallel")
// APP_TEST(UniformRandWalk, PpiSP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "SampleParallel")
// APP_TEST(UniformRandWalk, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", 10, false, "TransitParallel", false)
//APP_TEST(UniformRandWalk, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", 1, true, "TransitParallel", true)
// APP_TEST(UniformRandWalk, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", 10, false, "SampleParallel")
// APP_TEST(UniformRandWalk, OrkutTP, GRAPH_PATH"/com-orkut-weighted.graph", 10, false, "TransitParallel", false)
// APP_TEST(UniformRandWalk, OrkutLB, GRAPH_PATH"/com-orkut-weighted.graph", 10, false, "TransitParallel", true)
// APP_TEST(UniformRandWalk, OrkutSP, GRAPH_PATH"/com-orkut-weighted.graph", 10, false, "SampleParallel", false)
// APP_TEST(UniformRandWalk, LiveJournalTP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", 10, false, "TransitParallel", false)
APP_TEST(UniformRandWalk, LiveJournalLB, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", 1, true, "TransitParallel", true)
// APP_TEST(UniformRandWalk, LiveJournalSP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", 10, false, "SampleParallel", false)