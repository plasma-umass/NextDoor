#include "gtest/gtest.h"
#include "../nextdoor.cu"

#ifndef __TEST_BASE_H__
#define __TEST_BASE_H__

#define GRAPH_PATH "../GPUesque-for-eval/input/"

#define APP_TEST(App,Graph,Path,Runs,CheckResults,KernelType,LoadBalancing) \
  TEST(App, Graph) { \
  EXPECT_TRUE(nextdoor(Path, "adj-list", "text", Runs, CheckResults, false, KernelType, LoadBalancing));\
}

// TEST(KHop, Mico) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/micro.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Reddit) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/reddit_sampled_matrix", "adj-list", "text", CHECK_RESULTS, false));
// }

#endif