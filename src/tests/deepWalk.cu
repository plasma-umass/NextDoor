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
  EdgePos_t x = RandNumGen::rand_int(state, numEdges);
  float y = curand_uniform(state)*max_weight;
  while (y > transitEdgeWeights[x]) {
    x = RandNumGen::rand_int(state, numEdges);
    y = curand_uniform(state)*max_weight;
  }
  return transitEdges[x];
}

//nvprof bin/test_rw_10.2_x86_64 by-pass --graph-file=/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix --walks-per-node=1 --walk-length=10 --walk-mode=0

APP_TEST(DeepWalk, Citeseer, GRAPH_PATH"/citeseer-weighted.graph", 10) //SP: 0.5 ms
APP_TEST(DeepWalk, Mico, GRAPH_PATH"/micro-weighted.graph", 10) //SP: 1.05 ms
APP_TEST(DeepWalk, Reddit, GRAPH_PATH"/reddit_sampled_matrix", 10) //SP: 1.43 ms
