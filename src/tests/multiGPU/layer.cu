#include "testBase.h"
#include "../apps/layer/layerSampling.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS false
#define COMMA ,

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<LayerSample COMMA LayerSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<LayerSample COMMA LayerSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<LayerSample COMMA LayerSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<LayerSample COMMA LayerSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(LayerSample, Layer, LayerSamplingApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<LayerSample COMMA LayerSamplingApp>, "TransitParallel", true)
