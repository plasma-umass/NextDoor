#include "testBase.h"

__host__ __device__ int steps() {return 10;}

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
  if (numEdges == 1) {
    return transitEdges[0];
  }
  
  EdgePos_t x = RandNumGen::rand_int(state, numEdges);
  float y = curand_uniform(state)*max_weight;

  while (y > transitEdgeWeights[x]) {
    x = RandNumGen::rand_int(state, numEdges);
    y = curand_uniform(state)*max_weight;
  }

  return transitEdges[x];
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
  if (numEdges == 1) {
    if (CACHE_EDGES)
      return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(0, transitEdges, cachedEdges, globalLoadBV);
    return transitEdges[0];
  }
  
  EdgePos_t x = RandNumGen::rand_int(state, numEdges);
  float y = curand_uniform(state)*max_weight;
  float weight;
  if (CACHE_WEIGHTS) {
    weight = cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(x, transitEdgeWeights, cachedWeights, globalLoadBV);
  } else {
    weight = transitEdgeWeights[x];
  }

  while (y > weight) {
    x = RandNumGen::rand_int(state, numEdges);
    y = curand_uniform(state)*max_weight;
    if (CACHE_WEIGHTS) {
      weight = cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(x, transitEdgeWeights, cachedWeights, globalLoadBV);
    } else {
      weight = transitEdgeWeights[x];
    }
  }

  if (CACHE_EDGES)
    return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(x, transitEdges, cachedEdges, globalLoadBV);
  else
    return transitEdges[x];
}


/*
  float2 randNums = curand_normal2(state);
  EdgePos_t x =  min((EdgePos_t)(randNums.x*numEdges), numEdges-1);
  float y = randNums.y*max_weight;

  while (y > transitEdgeWeights[x]) {
    randNums = curand_normal2(state);
    x =  min((EdgePos_t)(randNums.x*numEdges), numEdges-1);
    y = randNums.y*max_weight;
  }

  return transitEdges[x];
*/

//nvprof bin/test_rw_10.2_x86_64 by-pass --graph-file=/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix --walks-per-node=1 --walk-length=10 --walk-mode=0

#define RUNS 1
#define CHECK_RESULTS true

// APP_TEST(DeepWalk, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "TransitParallel") 
// APP_TEST(DeepWalk, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, PpiTP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, PpiSP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "SampleParallel")
//APP_TEST(DeepWalk, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", 10, false, "TransitParallel", false)
//APP_TEST(DeepWalk, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", 10, false, "SampleParallel", false)
APP_TEST(DeepWalk, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST(DeepWalk, LiveJournalTP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", 10, false, "TransitParallel", false)
APP_TEST(DeepWalk, LiveJournalLB, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", true)
//APP_TEST(DeepWalk, LiveJournalSP, GRAPH_PATH"/soc-LiveJournal1-weighted.graph", 10, false, "SampleParallel", false)
//APP_TEST(DeepWalk, OrkutTP, GRAPH_PATH"/com-orkut-weighted.graph", 10, false, "TransitParallel", false)
APP_TEST(DeepWalk, OrkutLB, GRAPH_PATH"/com-orkut-weighted.graph", RUNS, CHECK_RESULTS, "TransitParallel", true)
//APP_TEST(DeepWalk, OrkutSP, GRAPH_PATH"/com-orkut-weighted.graph", 10, false, "SampleParallel", false)










































































































