#include "sample.hpp"
#include "csr.hpp"
#include "utils.hpp"
#include "sampler.cuh"
#include "rand_num_gen.cuh"

struct NextDoorData {

};

CSR* loadGraph(Graph& graph, char* graph_file, char* graph_type, char* graph_format);
GPUCSRPartition transferCSRToGPU(CSR* csr);
// std::vector<VertexID_t> createInitialSamples(CSR* csr);
bool allocNextDoorDataOnGPU(CSR* csr, NextDoorData& data);
bool doSampling(NextDoorData& data);
std::vector<VertexID_t>* getFinalSamples(NextDoorData& data);