#include "testBase.h"
#include <stdlib.h>

#define RUNS 1
#define CHECK_RESULTS false
#define VERTICES_IN_SAMPLE 0
#include "apps/randomwalks/randomWalks.cu"
#include "check_results.cu"
#define COMMA ,

/**PPR**/
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalTP, LJ1_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalLB, LJ1_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalSP, LJ1_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutTP, ORKUT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutLB, ORKUT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutSP, ORKUT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsTP, PATENTS_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsLB, PATENTS_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsSP, PATENTS_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditTP, REDDIT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditLB, REDDIT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditSP, REDDIT_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, PPITP, PPI_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PPILB, PPI_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PPISP, PPI_PATH, RUNS, false, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)
