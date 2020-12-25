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
  
  //TODO: Use curand_normal2
  EdgePos_t x = RandNumGen::rand_int(state, numEdges);
  float y = curand_uniform(state)*max_weight;

  while (y > transitEdgeWeights[x]) {
    x = RandNumGen::rand_int(state, numEdges);
    y = curand_uniform(state)*max_weight;
  }

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

APP_TEST(DeepWalk, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, "TransitParallel")
APP_TEST(DeepWalk, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, "TransitParallel")
APP_TEST(DeepWalk, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, "TransitParallel")
APP_TEST(DeepWalk, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, "SampleParallel")
APP_TEST(DeepWalk, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", 10, "TransitParallel")
APP_TEST(DeepWalk, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", 10, "SampleParallel")
