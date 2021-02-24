#include "testBase.h"

#include <stdlib.h>    

struct DeepWalkApp {
  __host__ __device__ int steps() {return 100;}

  __host__ __device__ 
  int stepSize(int k) {
    return 1;
  }

  template<class SampleType>
  __device__ inline
  VertexID next(int step,CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                const CSR::Edge* transitEdges, const float* transitEdgeWeights,
                const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
  {
    if (numEdges == 1) {
      return transitEdges[0];
    }
    
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);
    float y = curand_uniform(state)*max_weight;

    while (y > transitEdgeWeights[x]) {
      x = RandNumGen::rand_int(state, numEdges);
      y = curand_uniform(state)*max_weight;
    }

    return transitEdges[x];
  }

  template<class SampleType, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool DECREASE_GM_LOADS, bool ONDEMAND_CACHING, int STATIC_CACHE_SIZE>
  __device__ inline
  VertexID nextCached(int step, const VertexID transit, const VertexID sampleIdx, 
    SampleType* sample,
                const float max_weight,
                const CSR::Edge* transitEdges, const float* transitEdgeWeights,
                const EdgePos_t numEdges, const EdgePos_t neighbrID, 
                curandState* state, VertexID_t* cachedEdges, float* cachedWeights,
                bool* globalLoadBV)
  {
    if (numEdges == 1) {
      if (CACHE_EDGES)
        return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS, CSR::Edge, ONDEMAND_CACHING, STATIC_CACHE_SIZE>(0, transitEdges, cachedEdges, globalLoadBV);
      return transitEdges[0];
    }
    
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);
    float y = curand_uniform(state)*max_weight;
    float weight;
    if (CACHE_WEIGHTS) {
      weight = cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS, float, ONDEMAND_CACHING, STATIC_CACHE_SIZE>(x, transitEdgeWeights, cachedWeights, globalLoadBV);
    } else {
      weight = transitEdgeWeights[x];
    }

    while (y > weight) {
      x = RandNumGen::rand_int(state, numEdges);
      y = curand_uniform(state)*max_weight;
      if (CACHE_WEIGHTS) {
        weight = cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS, float, ONDEMAND_CACHING, STATIC_CACHE_SIZE>(x, transitEdgeWeights, cachedWeights, globalLoadBV);
      } else {
        weight = transitEdgeWeights[x];
      }
    }

    if (CACHE_EDGES)
      return cacheAndGet<CACHE_SIZE, DECREASE_GM_LOADS, CSR::Edge, ONDEMAND_CACHING, STATIC_CACHE_SIZE >(x, transitEdges, cachedEdges, globalLoadBV);
    else
      return transitEdges[x];
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
      // initialValue.push_back(rand() % graph->get_n_vertices());
      initialValue.push_back(sampleIdx);
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

/*
  float2 randNums = curand_normal2(state);
  EdgePos_t x =  min((EdgePos_t)(randNums.x*numEdges), numEdges-1);
  float y = randNums.y*max_weight;

  while (y > transitEdgeWeights[x]) {
    randNums = curand_normal2(state);
    x =  min((EdgePos_t)(randNums.x*numEdges), numEdges-1);
    y = randNums.y*max_weight;
  }

  return transitEdges[x];
*/

//nvprof bin/test_rw_10.2_x86_64 by-pass --graph-file=/mnt/homes/abhinav/GPUesque-for-eval/input/reddit_sampled_matrix --walks-per-node=1 --walk-length=10 --walk-mode=0

class DummySample
{

};

#define RUNS 1
#define CHECK_RESULTS true

// APP_TEST(DeepWalk, DeepWalkApp, CiteseerTP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "TransitParallel") 
// APP_TEST(DeepWalk, DeepWalkApp, CiteseerSP, GRAPH_PATH"/citeseer-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, DeepWalkApp, MicoTP, GRAPH_PATH"/micro-weighted.graph", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, DeepWalkApp, MicoSP, GRAPH_PATH"/micro-weighted.graph", 10, false, "SampleParallel") 
// APP_TEST(DeepWalk, DeepWalkApp, PpiTP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "TransitParallel")
// APP_TEST(DeepWalk, DeepWalkApp, PpiSP, GRAPH_PATH"/ppi_sampled_matrix", 10, false, "SampleParallel")
#define COMMA ,
// APP_TEST(DummySample, DeepWalk, DeepWalkApp, RedditTP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS,  checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
//APP_TEST(DummySample, DeepWalk, DeepWalkApp, RedditSP, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)
// APP_TEST(DummySample, DeepWalk, DeepWalkApp, RedditLB, GRAPH_PATH"/reddit_sampled_matrix", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalTP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalLB, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalSP, "/mnt/homes/abhinav/KnightKing/build/bin/LJ1.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutTP, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutLB, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutSP, "/mnt/homes/abhinav/KnightKing/build/bin/orkut.data", RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)