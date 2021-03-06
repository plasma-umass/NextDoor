#include "testBase.h"
#include <stdlib.h>

#define RUNS 1
#define CHECK_RESULTS false
#define VERTICES_IN_SAMPLE 0
#include "apps/randomwalks/randomWalks.cu"
#include "check_results.cu"
#define COMMA ,

/**Node2Vec**/
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", false)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", true)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "SampleParallel", false)

APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", false)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", true)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "SampleParallel", false)

APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", false)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", true)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "SampleParallel", false)

APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", false)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", true)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "SampleParallel", false)

APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", false)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "TransitParallel", true)
APP_TEST_BINARY(Node2VecSample, Node2Vec, Node2VecApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<Node2VecSample COMMA Node2VecApp>, "SampleParallel", false)
