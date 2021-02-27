#include "testBase.h"

struct KHopApp {

  __host__ __device__ int steps() {return 2;}

  __host__ __device__ 
  int stepSize(int k) {
    return ((k == 0) ? 10 : 25);
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    EdgePos_t id = RandNumGen::rand_int(state, numEdges);
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
      initialValue.push_back((rand())%graph->get_n_vertices());
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

#define RUNS 10
#define CHECK_RESULTS false
#define COMMA ,

// APP_TEST(KHopSample, KHop, KHopApp, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
// APP_TEST(KHopSample, KHop, KHopApp, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
// APP_TEST(KHopSample, KHop, KHopApp, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)
// APP_TEST(KHop, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, "TransitParallel", true)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalTP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)

// APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalTP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalLB, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, 
                  checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, LiveJournalSP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "SampleParallel", false)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutTP, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", false)
// APP_TEST_BINARY(KHopSample, KHop, KHopApp, OrkutLB, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<KHopSample COMMA KHopApp>, "TransitParallel", true)
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

/*
 RUN      ] KHop.LiveJournalTP
Graph Binary Loaded
Graph has 68555726 edges and 4847569 vertices 
Final Size of each sample: 260
Maximum Neighbors Sampled at each step: 10
Number of Samples: 4847569
Maximum Threads Per Kernel: 41943040
Transit Parallel: End to end time 0.382718 secs
InversionTime: 0.01246, LoadBalancingTime: 0, GridKernelTime: 0, ThreadBlockKernelTime: 0, SubWarpKernelTime: 0, IdentityKernelTime: 0
totalSampledVertices 1012703170
[       OK ] KHop.LiveJournalTP (53033 ms)
[ RUN      ] KHop.LiveJournalSP
Graph Binary Loaded
Graph has 68555726 edges and 4847569 vertices 
Final Size of each sample: 260
Maximum Neighbors Sampled at each step: 10
Number of Samples: 4847569
Maximum Threads Per Kernel: 41943040
SampleParallel: End to end time 0.226597 secs
totalSampledVertices 1012646495
[       OK ] KHop.LiveJournalSP (50316 ms)
*/