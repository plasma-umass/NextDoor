#include "gtest/gtest.h"
#include "../nextdoor.cu"
__host__ __device__ int steps() {return 2;}

__host__ __device__ 
int stepSize(int k) {
  return ((k == 0) ? 5 : 2);
}

__device__ inline
VertexID next(int step, const VertexID transit, const VertexID sample, 
              const CSR::Edge* transitEdges, const EdgePos_t numEdges,
              const EdgePos_t neighbrID, 
              curandState* state)
{
  EdgePos_t id = RandNumGen::rand_int(state, numEdges);
  // if (sample == 100 && transit == 100) {
  //   printf("113: id %ld transitEdges[id] %d\n", (long)id, transitEdges[id]);
  // }
  return transitEdges[id];
}

#define GRAPH_PATH "../GPUesque-for-eval/input/"

// Tests factorial of 0.
TEST(KHop, Citeseer) {
  EXPECT_TRUE(nextdoor(GRAPH_PATH"/citeseer.graph", "adj-list", "text", true, false));
}