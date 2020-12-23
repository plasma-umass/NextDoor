#include "gtest/gtest.h"
#include "../nextdoor.cu"

#ifndef __TEST_BASE_H__
#define __TEST_BASE_H__

#define GRAPH_PATH "../GPUesque-for-eval/input/"
#define CHECK_RESULTS true 

#define APP_TEST(App,Graph,Path,Runs) \
  TEST(App, Graph) { \
  EXPECT_TRUE(nextdoor(Path, "adj-list", "text", Runs, CHECK_RESULTS, false));\
}

// TEST(KHop, Mico) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/micro.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Reddit) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/reddit_sampled_matrix", "adj-list", "text", CHECK_RESULTS, false));
// }

#endif