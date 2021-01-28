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

bool checkAdjacencyMatrixResult(CSR* csr, const VertexID_t INVALID_VERTEX, std::vector<VertexID_t>& initialSamples, 
                                const size_t finalSampleSize,
                                std::vector<VertexID_t>& hFinalSamples, 
                                std::vector<VertexID_t>& hFinalSamplesCSRRow, 
                                std::vector<VertexID_t>& hFinalSamplesCSRCol, int maxSteps)
{
  std::cout << "checking results" << std::endl;
  AdjMatrix adjMatrix;

  csrToAdjMatrix(csr, adjMatrix);
  size_t numNeighborsToSampleAtStep = 0;

  for (int step = 0; step < min(maxSteps, steps()); step++) {
    if (step == 0) {
      bool foundError = false;

      for (size_t s = 0; s < hFinalSamples.size(); s += finalSampleSize) {
        const size_t sampleId = s/finalSampleSize;
        const VertexID_t initialVal = initialSamples[sampleId];
        size_t contentsLength = 0;

        for (size_t v = 0; v < stepSize(step); v++) {
          EdgePos_t idx = v + s + numNeighborsToSampleAtStep;
          VertexID_t col = hFinalSamplesCSRCol[idx];
          //VertexID_t row = hFinalSamplesCSRRow[idx];
          VertexID_t transit = hFinalSamples[s + col];
          contentsLength += (int)(transit != INVALID_VERTEX);

          if (!foundError && transit != INVALID_VERTEX &&
            adjMatrix[initialVal].count(transit) == 0) {
            printf("%s:%d Invalid '%d' in Sample '%ld' at Step '%d'\n", __FILE__, __LINE__, transit, sampleId, step);
            foundError = true;
          }
        }

        if (!foundError && contentsLength == 0 && adjMatrix[initialVal].size() > 0) {
          printf("Step %d: '%ld' vertices sampled for sample '%ld' but sum of edges of all vertices in sample is '%ld'\n", 
                  step, contentsLength, sampleId, adjMatrix[initialVal].size());
          foundError = true;
        }
      }
    }

    numNeighborsToSampleAtStep += stepSize(step);
  }
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
  size_t numNeighborsSampledAtPrevStep = 0;
  
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
            printf("%s:%d Sample '%d' has %ld neighbors but %ld are sampled at Step '%d'\n", __FILE__, __LINE__, sampleId, 
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
    numNeighborsSampledAtPrevStep = stepSizeAtStep(step-1);
  }

  return true;
}