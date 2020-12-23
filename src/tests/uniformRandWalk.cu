#include "testBase.h"

__host__ __device__ int steps() {return 10;}

__host__ __device__ 
int stepSize(int k) {
  return 1;
}

__device__ inline
VertexID next(int step, const VertexID transit, const VertexID sample, 
              const CSR::Edge* transitEdges, const EdgePos_t numEdges,
              const EdgePos_t neighbrID, 
              curandState* state)
{
  EdgePos_t id = RandNumGen::rand_int(state, numEdges);
  return transitEdges[id];
}

APP_TEST(UniformRandWalk, Citeseer, GRAPH_PATH"/citeseer.graph")
APP_TEST(UniformRandWalk, Mico, GRAPH_PATH"/micro.graph")
APP_TEST(UniformRandWalk, Reddit, GRAPH_PATH"/reddit_sampled_matrix")