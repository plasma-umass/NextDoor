#include <omp.h>
#include "libNextDoor.hpp"

int numNeighborsSampledAtStep(int step)
{
  int n = 1;

  if (step < 0) {
    return 0;
  }

  if (stepSize(0) == 1) {
    n = 1;
  } else {
    n = stepSize(0);
  }

  for(int i = 1; i <= step; i++) {
    if (stepSize(i) == 1) {
      n += stepSize(i);
    } else {
      n += n * stepSize(i);
    }
  }

  return n;
}

typedef std::unordered_map<VertexID, std::unordered_set<VertexID>> AdjMatrix;

void csrToAdjMatrix(CSR* csr, AdjMatrix& adjMatrix)
{
  for (VertexID v : csr->iterate_vertices()) {
    adjMatrix[v] = std::unordered_set<VertexID> ();
    for (EdgePos_t i = csr->get_start_edge_idx(v); 
         i <= csr->get_end_edge_idx(v); i++) {
      VertexID e = csr->get_edges()[i];
      adjMatrix[v].insert(e);
    }
  }
}

template<class SampleType>
bool checkAdjacencyMatrixResult(CSR* csr, const VertexID_t INVALID_VERTEX, std::vector<VertexID_t>& initialSamples, 
                                const size_t finalSampleSize, std::vector<VertexID_t>& hFinalSamples, 
                                std::vector<SampleType>& samples, int maxSteps)
{
  std::cout << "checking results" << std::endl;
  AdjMatrix adjMatrix;

  csrToAdjMatrix(csr, adjMatrix);
  size_t numNeighborsToSampleAtStep = 0;

  for (int step = 0; step < min(maxSteps, steps()); step++) {
    bool foundError = false;
    std::cout << "Step: "<< step << " finalSampleSize " << finalSampleSize << " numNeighborsToSampleAtStep " << numNeighborsToSampleAtStep << std::endl;
    const size_t startIdxForCurrStep = (step == 0) ? 0 : (numNeighborsToSampleAtStep + stepSize(step));
    for (size_t s = 0; s < hFinalSamples.size(); s += finalSampleSize) {
      const size_t sampleId = s/finalSampleSize;
      size_t contentsLength = 0;

      //Two kinds of check are performed here.
      //1. If there is an edge in sample's adjacency matrix then there is same edge in the Graph.
      //2. All edges that can exist between vertices of two layers in graph also exists between 
      // sample's adjacency matrix.

      //Check first condition
      for (EdgePos_t v = 0; v < samples[sampleId].adjacencyMatrixLen[step]; v++) {
        VertexID_t col = samples[sampleId].adjacencyMatrixCol[step][v];
        VertexID_t row = samples[sampleId].adjacencyMatrixRow[step][v];
        VertexID_t transit = hFinalSamples[s + startIdxForCurrStep + col];
        VertexID_t prevVertex = -1;
        
        if (step == 0) {
          prevVertex = initialSamples[sampleId * initialSampleSize(nullptr) + row];
        } else {
          prevVertex = hFinalSamples[s + numNeighborsToSampleAtStep + row];
        }
        contentsLength += (int)(transit != INVALID_VERTEX);

        if (!foundError && transit != INVALID_VERTEX &&
          adjMatrix[prevVertex].count(transit) == 0) {
          std::cout << "col: " << col << " row: " << row << std::endl;
          printf("%s:%d Invalid '%d' in Sample '%ld' at for previous step vertex '%d' Step '%d'\n", __FILE__, __LINE__, transit, sampleId, prevVertex, step);
          foundError = true;
        }
      }

      //Check second condition
      for (EdgePos_t v = 0; v < (EdgePos_t)stepSize(step); v++) {
        VertexID_t transit = hFinalSamples[s + startIdxForCurrStep + v];
        EdgePos_t prevSZ = (step == 0) ? initialSampleSize(nullptr) : stepSize(step - 1);
        for (EdgePos_t prevVertexIdx = 0; prevVertexIdx < prevSZ; prevVertexIdx++) {
          VertexID_t prevVertex = -1;
          if (step == 0) {
            prevVertex = initialSamples[sampleId * initialSampleSize(nullptr) + prevVertexIdx];
          } else {
            prevVertex = hFinalSamples[s + numNeighborsToSampleAtStep + prevVertexIdx];
          }
          if (adjMatrix[prevVertex].count(transit) == 1) {
            //Edge exist in graph. So, search for that there is an edge in the sample.
            bool foundEdge = false;
            for (int e = 0; e < samples[sampleId].adjacencyMatrixLen[step]; e++) {
              VertexID_t col = samples[sampleId].adjacencyMatrixCol[step][e];
              VertexID_t row = samples[sampleId].adjacencyMatrixRow[step][e];
              VertexID_t v1 = -1;
              if (step == 0) {
                v1 = initialSamples[sampleId * initialSampleSize(nullptr) + row];
              } else {
                v1 = hFinalSamples[s + numNeighborsToSampleAtStep + row];
              }
              VertexID_t v2 = hFinalSamples[s + startIdxForCurrStep + col];

              if (v1 == prevVertex && v2 == transit) {
                foundEdge = true;
                break;
              }
            }

            if (!foundError && !foundEdge) {
              printf("Edge '%d'->'%d' exists in Graph but not in sample %d at step %d\n", prevVertex, transit, sampleId, step);
              foundError = true;
            }
          }
        }
      }
      

      // if (!foundError && contentsLength == 0) {
      //   printf("Step %d: '%ld' vertices sampled for sample '%ld' but sum of edges of all vertices in sample is '%ld'\n", 
      //           step, contentsLength, sampleId, adjMatrix[initialVal].size());
      //   foundError = true;
      // }
    }

    if (foundError)
      return false;
    
    if (step >= 1)
      numNeighborsToSampleAtStep += stepSize(step);
  }

  return true;
}

bool checkSampledVerticesResult(CSR* csr, const VertexID_t INVALID_VERTEX, std::vector<VertexID_t>& initialSamples, 
                                const size_t finalSampleSize, std::vector<VertexID_t>& finalSamples, int maxSteps)
{
  //Check result by traversing all sampled neighbors and making
  //sure that if neighbors at kth-hop is an adjacent vertex of one
  //of the k-1th hop neighbors.

  //First create the adjacency matrix.
  std::cout << "checking results" << std::endl;
  AdjMatrix adj_matrix;

  csrToAdjMatrix(csr, adj_matrix);

  //Now check the correctness
  size_t numNeighborsToSampleAtStep = 0;
  
  for (int step = 0; step < min(maxSteps, steps()); step++) {
    if (step == 0) { 
      bool foundError = false;
      #pragma omp parallel for shared(foundError)
      for (size_t s = 0; s < finalSamples.size(); s += finalSampleSize) {
        std::unordered_set<VertexID_t> uniqueNeighbors;
        // printf("omp_get_num_threads() %d\n", omp_get_num_threads());
        const size_t sampleId = s/finalSampleSize;
        const VertexID_t initialVal = initialSamples[sampleId];
        size_t contentsLength = 0;
        if (stepSize(step) != ALL_NEIGHBORS) {
          for (size_t v = s + numNeighborsToSampleAtStep; v < s + stepSize(step); v++) {
            VertexID_t transit = finalSamples[v];
            uniqueNeighbors.insert(transit);
            contentsLength += (int)(transit != INVALID_VERTEX);

            if (!foundError && transit != INVALID_VERTEX &&
                adj_matrix[initialVal].count(transit) == 0) {
              printf("%s:%d Invalid '%d' in Sample '%ld' at Step '%d'\n", __FILE__, __LINE__, transit, sampleId, step);
              foundError = true;
            }
          }

          if (!foundError && contentsLength == 0 && adj_matrix[initialVal].size() > 0) {
            printf("Step %d: '%ld' vertices sampled for sample '%ld' but sum of edges of all vertices in sample is '%ld'\n", 
                    step, contentsLength, sampleId, adj_matrix[initialVal].size());
            foundError = true;
          }
        } else {
          for (size_t v = s + numNeighborsToSampleAtStep; v < s + adj_matrix[initialVal].size(); v++) {
            VertexID_t transit = finalSamples[v];
            uniqueNeighbors.insert(transit);
            contentsLength += (int)(transit != INVALID_VERTEX);

            if (!foundError && transit != INVALID_VERTEX &&
                adj_matrix[initialVal].count(transit) == 0) {
              printf("%s:%d Invalid '%d' in Sample '%ld' at Step '%d'\n", __FILE__, __LINE__, transit, sampleId, step);
              foundError = true;
            }
          }

          if (!foundError && adj_matrix[initialVal].size() != contentsLength) {
            printf("%s:%d Sample '%ld' has %ld neighbors but %ld are sampled at Step '%d'\n", __FILE__, __LINE__, sampleId, 
                   adj_matrix[initialVal].size(), contentsLength, step);
            foundError = true;
          }
        }
      }

      if (foundError) return false;
    } else {
      bool foundError = false;
      #pragma omp parallel for shared(foundError)
      for (size_t s = 0; s < finalSamples.size(); s += finalSampleSize) {
        const size_t sampleId = s/finalSampleSize;
        size_t contentsLength = 0;
        size_t sumEdgesOfNeighborsAtPrevStep = 0;
        
        for (size_t v = s + numNeighborsSampledAtStep(step-2); v < s + numNeighborsSampledAtStep(step-1); v++) {
          sumEdgesOfNeighborsAtPrevStep +=  adj_matrix[finalSamples[v]].size();
        }
        
        // if (sampleId == 48) {
        //   printf("step %d start %d end %d\n", step, numNeighborsSampledAtStep(step-1),
        //          ((step == steps() - 1) ? finalSampleSize : numNeighborsSampledAtStep(step)));
        // }
        for (size_t v = s + numNeighborsSampledAtStep(step-1); 
             v < s + ((step == steps() - 1) ? finalSampleSize : numNeighborsSampledAtStep(step)); v++) {
          VertexID_t transit = finalSamples[v];
          contentsLength += (int)(transit != INVALID_VERTEX);
          
          bool found = false;
          if (transit != INVALID_VERTEX) {

            for (size_t v1 = s + numNeighborsSampledAtStep(step-2); v1 < s + numNeighborsSampledAtStep(step-1); v1++) {
              if (adj_matrix[finalSamples[v1]].count(transit) > 0) {
                found = true;
                break;
              }
            }

            if (!foundError && found == false) {
              printf("%s:%d Invalid '%d' in Sample '%ld' at Step '%d'\n", __FILE__, __LINE__, transit, sampleId, step);
              std::cout << "Contents of sample : [";
              for (size_t v2 = s; v2 < s + finalSampleSize; v2++) {
                std::cout << finalSamples[v2] << ", ";
              }
              std::cout << "]" << std::endl;
              foundError = true;
            }
          }
        }

        if (!foundError && contentsLength == 0 && sumEdgesOfNeighborsAtPrevStep > 0) {
          printf("Step %d: '%ld' vertices sampled for sample '%ld' but sum of edges of all vertices in sample is '%ld'\n", 
                  step, contentsLength, sampleId, sumEdgesOfNeighborsAtPrevStep);
          std::cout << "Contents of sample : [";
          for (size_t v2 = s; v2 < s + finalSampleSize; v2++) {
            std::cout << finalSamples[v2] << ", ";
          }
          std::cout << "]" << std::endl;
          foundError = true;
        }
      }

      if (foundError) return false;
    }

    numNeighborsToSampleAtStep = stepSizeAtStep(step);
  }

  return true;
}