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
  EdgePos_t id = RandNumGen::rand_int(state, numEdges);
  return transitEdges[id];
}

//nvprof bin/test_rw_10.2_x86_64 by-pass --graph-file=/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix --walks-per-node=1 --walk-length=10 --walk-mode=0

APP_TEST(UniformRandWalk, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, "TransitParallel") 
APP_TEST(UniformRandWalk, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, "SampleParallel") 
APP_TEST(UniformRandWalk, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, "TransitParallel")
APP_TEST(UniformRandWalk, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, "SampleParallel") 
APP_TEST(UniformRandWalk, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", 10, "TransitParallel")
APP_TEST(UniformRandWalk, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", 10, "SampleParallel")