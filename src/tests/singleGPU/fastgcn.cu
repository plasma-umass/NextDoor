#include "testBase.h"

#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64
#define NUM_SAMPLES 10000
#include "../apps/fastgcn/fastgcnSampling.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS false
#define COMMA ,


APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "SampleParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "SampleParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "SampleParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "SampleParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", true)

APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "SampleParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", false)
APP_TEST_BINARY(FastGCNSample, FastGCN, FastGCNSamplingApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<FastGCNSample COMMA FastGCNSamplingApp>, "TransitParallel", true)