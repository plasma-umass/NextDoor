#include "testBase.h"

#include <stdlib.h>    

struct RandomWalkApp {
  __host__ __device__ int steps() {return 100;}

  __host__ __device__ 
  int stepSize(int k) {
    return 1;
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

struct DeepWalkApp : public RandomWalkApp {
  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
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
};

struct PPRApp : public RandomWalkApp {
  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    const float walkEndProb = 0.01f;
    float p = curand_uniform(state);
    if (p < walkEndProb) {
      return -1;
    }

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
};

class DummySample
{

};

struct Node2VecApp : public RandomWalkApp {
  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    if (numEdges == 1 || step == 0) {
      sample->t = *transits;
      return transitEdges[0];
    }  
    
    const float p = 2.0f;
    const float q = 0.5f;

    do {
      EdgePos_t x = RandNumGen::rand_int(state, numEdges);
      VertexID v = transitEdges[x];
      float y = curand_uniform(state)*max(max(p, 1/q), 1.0f);
      const CSR::Edge* tEdges = csr->get_edges(sample->t);
      EdgePos_t tNumEdges = csr->get_n_edges_for_vertex(sample->t);
      float h;
      if (x == sample->t) {
        h = p;
      } else if (utils::binarySearch(tEdges, v, tNumEdges)) {
        h = 1/q;
      } else {
        h = 1.0f;
      }

      if (y < h) {
        sample->t = *transits;
        return v;
      }
    } while (true);
  }
};

class Node2VecSample {
public:
  VertexID t;
};


#define RUNS 1
#define CHECK_RESULTS false
#define VERTICES_IN_SAMPLE 0
#include "../check_results.cu"
#define COMMA ,

/**DeepWalk**/
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", false)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "TransitParallel", true)
// APP_TEST_BINARY(DummySample, DeepWalk, DeepWalkApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA DeepWalkApp>, "SampleParallel", false)

/**PPR**/
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalTP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalLB, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, LiveJournalSP, LJ1_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutTP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutLB, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, OrkutSP, ORKUT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsTP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsLB, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PatentsSP, PATENTS_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditTP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditLB, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, RedditSP, REDDIT_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

APP_TEST_BINARY(DummySample, PPR, PPRApp, PPITP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", false)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PPILB, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "TransitParallel", true)
APP_TEST_BINARY(DummySample, PPR, PPRApp, PPISP, PPI_PATH, RUNS, CHECK_RESULTS, checkSampledVerticesResult<DummySample COMMA PPRApp>, "SampleParallel", false)

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
