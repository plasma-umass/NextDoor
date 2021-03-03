#include "testBase.h"

#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64
#define NUM_SAMPLES 10000
#include "../apps/ladies/ladiessampling.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS false
#define COMMA ,

APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, LADIES, ImportanceSampleApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)
