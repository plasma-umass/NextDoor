#include "gtest/gtest.h"
#include "../nextdoor.cu"
#ifndef __TEST_BASE_H__
#define __TEST_BASE_H__

#define GRAPH_PATH "../GPUesque-for-eval/input/"
#define GRAPH_PARENT_DIR "./input/"
#define LJ1_PATH GRAPH_PARENT_DIR"LJ1.data"
#define ORKUT_PATH GRAPH_PARENT_DIR"orkut.data"
#define PATENTS_PATH GRAPH_PARENT_DIR"patents.data"
#define REDDIT_PATH GRAPH_PARENT_DIR"reddit.data"
#define PPI_PATH GRAPH_PARENT_DIR"ppi.data"

#define APP_TEST(SampleClass,AppName,App,Graph,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(AppName, Graph) { \
  bool b = nextdoor<SampleClass, App>(Path, "adj-list", "text", Runs, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc);\
  EXPECT_TRUE(b);\
}

#define APP_TEST_BINARY(SampleClass,AppName,App,Graph,Path,Runs,CheckResults,chkResultsFunc,KernelType,LoadBalancing) \
  TEST(AppName, Graph) { \
  bool b = nextdoor<SampleClass, App>(Path, "edge-list", "binary", Runs, CheckResults, false, KernelType, LoadBalancing, chkResultsFunc);\
  EXPECT_TRUE(b);\
}

// TEST(KHop, Mico) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/micro.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Reddit) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/reddit_sampled_matrix", "adj-list", "text", CHECK_RESULTS, false));
// }

#endif