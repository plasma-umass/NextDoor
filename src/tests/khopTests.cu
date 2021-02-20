#include "testBase.h"

struct KHopApp {

  __host__ __device__ int steps() {return 2;}

  __host__ __device__ 
  int stepSize(int k) {
    return ((k == 0) ? 10 : 15);
  }

  template<class SampleType>

  __device__ inline
  VertexID next(int step,CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
    SampleType* sample, 
    const float max_weight,
    const CSR::Edge* transitEdges, const float* transitEdgeWeights,
    const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
  {
    if (numEdges == 0) {
      return transitEdges[0];
    }
    EdgePos_t id = RandNumGen::rand_int(state, numEdges);
    return transitEdges[id];
  }

  template<class SampleType, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool DECREASE_GM_LOADS>
  __device__ inline
  VertexID nextCached(int step, const VertexID transit, const VertexID sampleIdx, 
                SampleType* sample, const float max_weight,
                const CSR::Edge* transitEdges, const float* transitEdgeWeights,
                const EdgePos_t numEdges, const EdgePos_t neighbrID, 
                curandState* state, VertexID_t* cachedEdges, float* cachedWeights,
                bool* globalLoadBV)
  {
    EdgePos_t id = RandNumGen::rand_int(state, numEdges);
    if (CACHE_EDGES)
      return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS>(id, transitEdges, cachedEdges, globalLoadBV);
    return transitEdges[id];
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return SampledVertices;
  }

  #define VERTICES_PER_SAMPLE 1

  __host__ __device__ EdgePos_t numSamples(CSR* graph)
  {
    return graph->get_n_vertices() / VERTICES_PER_SAMPLE;
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;

    for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
      initialValue.push_back(sampleIdx * VERTICES_PER_SAMPLE + i);
    }

    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
  {
    return VERTICES_PER_SAMPLE;
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __host__ __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;

    return sample;
  }
};

class KHopSample
{

};

#define RUNS 1
#define CHECK_RESULTS true
#define COMMA ,

// APP_TEST(KHopSample, KHop, KHopApp, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
// APP_TEST(KHopSample, KHop, KHopApp, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
// APP_TEST(KHopSample, KHop, KHopApp, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)
// APP_TEST(KHop, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalTP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)

// APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalTP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalLB, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalSP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutTP, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutLB, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutSP, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)

// APP_TEST(KHop, Citeseer, GRAPH_PATH"/citeseer.graph", 1, true, "TransitParallel")
// APP_TEST(KHop, Mico, GRAPH_PATH"/micro.graph", 1, false, "TransitParallel")
// APP_TEST(KHop, Reddit, GRAPH_PATH"/reddit_sampled_matrix", 1, false, "TransitParallel")


// TEST(KHop, Citeseer) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/citeseer.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Mico) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/micro.graph", "adj-list", "text", CHECK_RESULTS, false));
// }

// TEST(KHop, Reddit) {
//   EXPECT_TRUE(nextdoor(GRAPH_PATH"/reddit_sampled_matrix", "adj-list", "text", CHECK_RESULTS, false));
// }

