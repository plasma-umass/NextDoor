#include <stdlib.h>

#define NUM_ROOT_VERTICES 100

class MultiRWSample
{
public:
  int rootVertices[NUM_ROOT_VERTICES];
  int lastRootIdx;
};

struct MultiRWApp {
  __host__ __device__ int steps() {return 100;}

  __host__ __device__ 
  int stepSize(int k) {
    return 1;
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    if (numEdges == 0) {
      return -1;
    }
    if (numEdges == 1) {
      VertexID_t v = transitEdges[0];
      if (step > 0) {
        sample->rootVertices[sample->lastRootIdx] = v;
      }

      return transitEdges[0];
    }
    
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);
    VertexID_t v = transitEdges[x];

    if (step > 0) {
      sample->rootVertices[sample->lastRootIdx] = v;
    }

    return v;
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return SampledVertices;
  }

  __host__ __device__ EdgePos_t numSamples(CSR* graph)
  {
    return graph->get_n_vertices();
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;
    initialValue.push_back(sample.rootVertices[0]);

    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
  {
    return 1;
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return true;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    CSRPartition* csr = (CSRPartition*)&csrPartitionBuff[0];
    //Use rejection sampling to sample based on the degree of vertices.
    int x = RandNumGen::rand_int(randState, NUM_ROOT_VERTICES);
    //printf("x %d\n", x);
    sample.lastRootIdx = x;
    return sample.rootVertices[x];
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    //printf("sample %d\n", sampleID);
    for (int i = 0; i < NUM_ROOT_VERTICES; i++) {
      sample.rootVertices[i] = rand() % graph->get_n_vertices();
      // if (sampleID + i < graph->get_n_vertices()) {
      //   sample.rootVertices[i] = sampleID + i;
      // } else {
      //   sample.rootVertices[i] = sampleID;
      // }
    }
    return sample;
  }
};
