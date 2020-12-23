#include "testBase.h"

extern "C" {

  __host__ __device__ int steps() {return 2;}

  __host__ __device__ 
  int stepSize(int k) {
    return ((k == 0) ? 10 : 25);
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

}

APP_TEST(KHop, Citeseer, GRAPH_PATH"/citeseer.graph", 1)
APP_TEST(KHop, Mico, GRAPH_PATH"/micro.graph", 1)
APP_TEST(KHop, Reddit, GRAPH_PATH"/reddit_sampled_matrix", 1)


// TEST(KHop, Citeseer) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/citeseer.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Mico) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/micro.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Reddit) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/reddit_sampled_matrix", "adj-list", "text", CHECK_RESULTS, false));
// }

