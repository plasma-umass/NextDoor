#include "testBase.h"
#include <stdlib.h>

#define RUNS 1
#define CHECK_RESULTS false
#define VERTICES_IN_SAMPLE 0
#include "../apps/randomwalks/randomWalks.cu"
#include "../check_results.cu"
#define COMMA ,

/**PPR**/
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalLB, LJ1_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutLB, ORKUT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsLB, PATENTS_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditLB, REDDIT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)

APP_TEST_BINARY(DummySample, PPR, PPRApp, PPILB, PPI_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)