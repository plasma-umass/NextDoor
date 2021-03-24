#include "testBase.h"
#include <stdlib.h>

#define RUNS 1
#define CHECK_RESULTS true
#define VERTICES_IN_SAMPLE 0
#include "apps/randomwalks/randomWalks.cu"
#include "check_results.cu"
#define COMMA ,

// /**DeepWalk**/
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)
