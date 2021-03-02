/**
 * Uniform Random Walk Example
*/

struct UniformRandomWalkSample {
  //A Sample struct can contain extra information required
  //for the application. 
};

//An App struct defines the functions needed to execute the 
//application.
struct UniformRandomWalkApp {
  //steps() returns the length of random walk
  __host__ __device__ int steps() {return 100;}

  //At each step only 1 neighbor of a vertex is sampled
  //stepSize reflects that
  __host__ __device__ 
  int stepSize(int k) {
    return 1;
  }

  //next function samples one neighbor and returns it
  template<typename SampleType, typename EdgeArray, typename WeightArray>
  __device__ inline
  VertexID next(int step, CSRPartition* csr, const VertexID* transit, const VertexID sampleIdx,
                SampleType* sample, 
                const float max_weight,
                EdgeArray& transitEdges, WeightArray& transitEdgeWeights,
                const EdgePos_t numEdges, const VertexID_t neighbrID, curandState* state)
  {
    EdgePos_t x = RandNumGen::rand_int(state, numEdges);
    return transitEdges[x];
  }

  //Type of Sampling. Random Walk samples 1 vertex from the 
  //individual neighborhood of individual transit.
  __host__ __device__ int samplingType()
  {
    return SamplingType::IndividualNeighborhood;
  }

  __host__ __device__ OutputFormat outputFormat()
  {
    return SampledVertices;
  }

  //Number of Random Walks. Each Sample represents one walk.
  //Here, we have as many random walks as the number of vertices.
  __host__ __device__ EdgePos_t numSamples(CSR* graph)
  {
    return graph->get_n_vertices();
  }

  /**Next two functions are used to specify initialization of each sample (walk).
   * One vertex is assigned to one sample.**/
  template<class SampleType>
  __host__ std::vector<VertexID_t> initialSample(int sampleIdx, CSR* graph, SampleType& sample)
  {
    std::vector<VertexID_t> initialValue;
    for (int i = 0; i < VERTICES_PER_SAMPLE; i++) {
      initialValue.push_back(sampleIdx);
    }

    return initialValue;
  }

  __host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
  {
    return 1;
  }

  //
  __host__ __device__ bool hasExplicitTransits()
  {
    return false;
  }

  template<class SampleType>
  __host__ __device__ VertexID_t stepTransits(int step, const VertexID_t sampleID, SampleType& sample, int transitIdx, curandState* randState)
  {
  }

  //Initialize the Random Walk Sample.
  template<class SampleType>
  __host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
  {
    SampleType sample;

    return sample;
  }
};

int main(int argc, char* argv[]) {
  //Call the main function of NextDoor to execute application.
  return appMain<UniformRandomWalkSample, UniformRandomWalkApp>(argc, argv, nullptr);
}