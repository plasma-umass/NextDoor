#include "testBase.h"

#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64
#define NUM_SAMPLES 10000
#include "../apps/importanceSampling.cu"
#include "../check_results.cu"

#define RUNS 1
#define CHECK_RESULTS false
#define COMMA ,


APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)

APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "SampleParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", false)
APP_TEST_BINARY(ImportanceSample, ImportanceSampling, ImportanceSampleApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkAdjacencyMatrixResult<ImportanceSample COMMA ImportanceSampleApp>, "TransitParallel", true)
