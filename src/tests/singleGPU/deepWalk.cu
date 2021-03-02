#include "testBase.h"
#include <stdlib.h>

#define RUNS 1
#define CHECK_RESULTS true
#define VERTICES_IN_SAMPLE 0
#include "apps/randomWalks.cu"
#include "check_results.cu"
#define COMMA ,

/**DeepWalk**/
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

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
