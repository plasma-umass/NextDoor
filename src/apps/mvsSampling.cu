#define VERTICES_IN_BATCH 32
#define VERTICES_PER_SAMPLE (VERTICES_IN_BATCH)

class MVSSample 
{
public:
  int *row;
  int *col;
  int length;
  int posInStorage;
  VertexID_t vertices[VERTICES_PER_SAMPLE];
};

int *dRowStorage = nullptr;
int *dColStorage = nullptr;
int storagePosition;
std::vector<std::vector<VertexID_t>> batches;

struct MVSSamplingApp {
  __host__ __device__ int steps() {return 1;}

  __host__ __device__ 
  int stepSize(int k) {
    return 32;
  }

  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    //TODO: Optimize this using warp shuffles
    for (int e = neighbrID; e < numEdges; e += stepSize(0)) {
      int p = ::atomicAdd(&sample->length, 1);
      sample->row[p] = transits[0];
      sample->col[p] = transitEdges[e];
      // if (transits[0] == 0 and sampleIdx == 0) {
      //   printf("e %d numEdges %d\n", e, numEdges);
      // }
    }
    return -1;
  }

  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return AdjacencyMatrix;
  }

  __host__ EdgePos_t numSamples(CSR* graph)
  {
    return (graph->get_n_edges() > 100000000) ? 800000 : min(1500000, DIVUP(graph->get_n_vertices()*8, VERTICES_PER_SAMPLE));
  }

  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
  }

  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;
    EdgePos_t totalEdges = 0;
    size_t storageStartPos = storagePosition;
    for (int i = 0; i < VERTICES_IN_BATCH; i++) {
      VertexID_t v = (sampleIdx * VERTICES_IN_BATCH + i) % graph->get_n_vertices();
      initialValue.push_back(v);
      totalEdges += graph->n_edges_for_vertex(v);
      sample.vertices[i] = v;
    }

    storagePosition += totalEdges;

    sample.row = dRowStorage + storageStartPos;
    sample.col = dColStorage + storageStartPos;
    sample.posInStorage = storageStartPos;
    sample.length = 0;
    return initialValue;
  }

  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;
    return sample;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph) { return VERTICES_PER_SAMPLE;}
};
