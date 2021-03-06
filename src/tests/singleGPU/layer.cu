#include "testBase.h"
#include "../apps/layer/layerSampling.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS false
#define COMMA ,

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, nullptr, "SampleParallel", false)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", true)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", false)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", false)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", true)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, nullptr, "SampleParallel", false)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", false)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", true)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, nullptr, "SampleParallel", false)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", false)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", true)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, nullptr, "SampleParallel", false)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, nullptr, "TransitParallel", false)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PPILB, PPI_PATH, RUNS, false, nullptr, "TransitParallel", true)
APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, nullptr, "SampleParallel", false)