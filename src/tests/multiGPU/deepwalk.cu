#include "testBase.h"
#include <stdlib.h>

#define RUNS 1
#define CHECK_RESULTS false
#define VERTICES_IN_SAMPLE 0
#include "../apps/randomwalks/randomWalks.cu"
#include "../check_results.cu"
#define COMMA ,

/**DeepWalk**/
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)