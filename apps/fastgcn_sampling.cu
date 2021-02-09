#include "libNextDoor.hpp"
#include "main.cu"

#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 64
#define VERTICES_PER_SAMPLE 64

class LayerSample 
{
public:
  int adjacencyMatrixLen[NUM_LAYERS];
  int adjacencyMatrixRow[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
  int adjacencyMatrixCol[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
  int adjacencyMatrixVal[NUM_LAYERS][VERTICES_PER_SAMPLE*NUM_SAMPLED_VERTICES];
};

struct FastGCNApp {
  __host__ __device__ int steps() {return NUM_LAYERS;}

  __host__ __device__ 
  int stepSize(int k) {
    return NUM_SAMPLED_VERTICES;
  }

  template<class SampleType>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx, 
                SampleType* sample, const float max_weight,
                const CSR::Edge* transitEdges, const float* transitEdgeWeights,
                const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
  {
    EdgePos_t id = RandNumGen::rand_int(state, csr->get_n_vertices());
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
      VertexID transit = transits[i];
      bool hasEdge = csr->has_edge_logn(transit, id);
      if (hasEdge) {
        int len = ::atomicAdd(&sample->adjacencyMatrixLen[step], 1);
        //int cooIdx = step * NUM_SAMPLED_VERTICES + len;
        sample->adjacencyMatrixRow[step][len] = i;
        sample->adjacencyMatrixCol[step][len] = neighbrID;
        sample->adjacencyMatrixVal[step][len] = 1.0f;
      }
    }

    return id;
  }

  template<class SampleType, int CACHE_SIZE, bool CACHE_EDGES, bool CACHE_WEIGHTS, bool DECREASE_GM_LOADS>
  __device__ inline
  VertexID nextCached(int step, const VertexID transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
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
    return SamplingType::CollectiveNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return AdjacencyMatrix;
  }

  __host__ EdgePos_t numSamples(CSR* graph)
  {
    return graph->get_n_vertices() / VERTICES_PER_SAMPLE / (graph->get_n_vertices() > 1000000 ? 100 : 1);
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
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    for (int i = 0; i < NUM_LAYERS; i++) {
      sample.adjacencyMatrixLen[i] = 0;
    }
    return sample;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    return -1;
  }
};

static NextDoorData<LayerSample, FastGCNApp> nextDoorData;

#include "check_results.cu"
int main(int argc, char* argv[]) {
  return appMain<LayerSample, FastGCNApp>(argc, argv, checkAdjacencyMatrixResult<LayerSample, FastGCNApp>);
}

#ifdef PYTHON_2
#include "nextDoorModule.cu"
#endif

#ifdef PYTHON_3
#include "nextDoorModule.cu"
#endif