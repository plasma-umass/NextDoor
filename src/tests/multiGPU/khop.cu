#include "testBase.h"
#include "../apps/khop/khop.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS false
#define COMMA ,

APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
