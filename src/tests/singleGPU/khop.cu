#include "testBase.h"
#include "../apps/khop/khop.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS true
#define COMMA ,

APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, RedditSP, REDDIT_PATH, RUNS, false, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)

APP_TEST_BINARY(KHopSample, KHop, KHopApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, PPISP, PPI_PATH, RUNS, false, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)
