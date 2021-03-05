#define NUM_LAYERS 5
#define NUM_SAMPLED_VERTICES 8
#define VERTICES_PER_SAMPLE 8
#define NUM_SAMPLES 10000

class LayerSample 
{
public:
};

struct LayerSamplingApp {
  __host__ __device__ int steps() {return NUM_LAYERS;}

  __host__ __device__ 
  int stepSize(int k) {
    return NUM_SAMPLED_VERTICES;
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx,
    SampleType* sample, 
    const float max_weight,
    EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
    const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    EdgePos_t v = RandNumGen::rand_int(state, NUM_SAMPLED_VERTICES);
    if (csr->get_n_edges_for_vertex(v) > 0) {
      EdgePos_t e = RandNumGen::rand_int(state, csr->get_n_edges_for_vertex(v));
      return csr->get_edges(v)[e];
    }

    return v;
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
    return NUM_SAMPLES;//graph->get_n_vertices() / VERTICES_PER_SAMPLE / (graph->get_n_vertices() > 1000000 ? 100 : 1);
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
    return -1;
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;

    for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
      initialValue.push_back((sampleIdx * VERTICES_PER_SAMPLE + i) % graph->get_n_vertices());
    }

    return initialValue;
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    
    return sample;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
  {
    return VERTICES_PER_SAMPLE;
  }
};