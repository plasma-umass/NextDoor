#include <stdlib.h>

#include "libNextDoor.hpp"
#include "main.cu"
#include "check_results.cu"

#define VERTICES_PER_SAMPLE 64

__host__ __device__ int steps() {return 1;}

__host__ __device__ 
int stepSize(int k) {
  return 1;
}

class SubGraphSample 
{
public:
  VertexID_t vertices[VERTICES_PER_SAMPLE];
  int adjacencyMatrixLen;
  int adjacencyMatrixRow[VERTICES_PER_SAMPLE*VERTICES_PER_SAMPLE];
  int adjacencyMatrixCol[VERTICES_PER_SAMPLE*VERTICES_PER_SAMPLE];
  int adjacencyMatrixVal[VERTICES_PER_SAMPLE*VERTICES_PER_SAMPLE];
};

template<class SampleType>
__device__ inline
VertexID next(int step, CSRPartition* csr, const VertexID* transits, const VertexID sampleIdx, 
              SampleType* sample, const float max_weight,
              const CSR::Edge* transitEdges, const float* transitEdgeWeights,
              const EdgePos_t numEdges, const EdgePos_t neighbrID, curandState* state)
{
  VertexID_t v1 = transits[0];
  for (int v2Idx = 0; v2Idx < VERTICES_PER_SAMPLE; v2Idx++) {
    VertexID_t v2 = sample->vertices[v2Idx];
    bool hasEdge = csr->has_edge_logn(v1, v2);
    if (hasEdge) {
      int len = ::atomicAdd(&sample->adjacencyMatrixLen, 1);
      //int cooIdx = step * NUM_SAMPLED_VERTICES + len;
      sample->adjacencyMatrixRow[len] = v1;
      sample->adjacencyMatrixCol[len] = v2;
      sample->adjacencyMatrixVal[len] = 1.0f;
    }
  }

  return -1;
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
  return SamplingType::IndividualNeighborhood;
}

__host__ __device__ OutputFormat outputFormat()
{
  return AdjacencyMatrix;
}

__host__ EdgePos_t numSamples(CSR* graph)
{
  return graph->get_n_vertices() / VERTICES_PER_SAMPLE / 10;
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
    VertexID_t v = sampleIdx * VERTICES_PER_SAMPLE + i;
    initialValue.push_back(v);
    sample.vertices[i] = v;
  }

  return initialValue;
}

template<class SampleType>
__host__ SampleType initializeSample(CSR* graph, const VertexID_t sampleID)
{
  SampleType sample;
  sample.adjacencyMatrixLen = 0;

  return sample;
}

__host__ __device__ EdgePos_t initialSampleSize(CSR* graph)
{
  return VERTICES_PER_SAMPLE;
}

#define RUNS 1
#define CHECK_RESULTS true

template<class SampleType>
bool checkSubGraphResult(NextDoorData<SampleType>& nextDoorData)
{
  //Check result by traversing all sampled neighbors and making
  //sure that if neighbors at kth-hop is an adjacent vertex of one
  //of the k-1th hop neighbors.
  CSR* csr = nextDoorData.csr;
  auto& initialSamples = nextDoorData.initialContents;
  auto finalSampleSize = getFinalSampleSize();
  auto& finalSamples = nextDoorData.hFinalSamples;
  auto INVALID_VERTEX = nextDoorData.INVALID_VERTEX;
  auto& samples = nextDoorData.samples;
  int maxSteps = 4;

  //First create the adjacency matrix.
  std::cout << "checking results" << std::endl;
  AdjMatrix adj_matrix;

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  size_t numNeighborsToSampleAtStep = 0;
  bool foundError = false;
  int sampleIdx = 0;
  for (SampleType& sample : samples) {
    //Go through all edges between two vertices and see if they exist in the graph
    for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
      VertexID_t v1 = sample.adjacencyMatrixRow[e];
      VertexID_t v2 = sample.adjacencyMatrixCol[e];

      if (!foundError && adj_matrix[v1].count(v2) == 0) {
        printf("Sample '%d': no edge '%d' -> '%d' in graph\n", sampleIdx, v1, v2);
        foundError = true;
      }
    }

    //Go through edges between two vertices in graph and see if they exist in sample
    for (int vidx1 = 0; vidx1 < VERTICES_PER_SAMPLE; vidx1++) {
      VertexID_t v1 = sample.vertices[vidx1];
      for (int vidx2 = 0; vidx2 < VERTICES_PER_SAMPLE; vidx2++) {
        VertexID_t v2 = sample.vertices[vidx2];

        if (adj_matrix[v1].count(v2) == 1 and v1 != v2) {
          bool foundEdge = false;
          //Edge in Graph. Check if it is in Sample.
          for (int e = 0; e < sample.adjacencyMatrixLen; e++) {
            if (sample.adjacencyMatrixRow[e] == v1 && sample.adjacencyMatrixCol[e] == v2) {
              foundEdge = true;
              break;
            }
          }

          if (!foundEdge) {
            if (!foundError) {
              printf("Sample '%d': Edge '%d'->'%d' exists in graph but not in sample of length '%d'\n", sampleIdx, v1, v2, sample.adjacencyMatrixLen);
            }
            foundError = true;
          }
        }
      }
    }

    sampleIdx++;
  }

  if (foundError) return false;
  return true;
}

static NextDoorData<SubGraphSample> nextDoorData;

int main(int argc, char* argv[]) {
  return appMain<SubGraphSample>(argc, argv, checkSubGraphResult<SubGraphSample>);
}

//#include "nextDoorModule.cu"