#include <cstdlib>

#ifndef __SAMPLER_CUH__
#define __SAMPLER_CUH__

class Sampler {
protected:
  Sampler* d_ptr;
public:
  Sampler() : d_ptr(nullptr) {}
  virtual Sampler* device_cpy() = 0;
  virtual size_t size() = 0;
  virtual void free()
  {
    if (d_ptr) {
      cudaFree(d_ptr);
      d_ptr = nullptr;
    }
  }
};

class GraphSageSampler {
  virtual Sampler* device_cpy() {return nullptr;}
  virtual size_t size() {return sizeof(GraphSageSampler);}
};

class Node2VecSampler : public Sampler {
private:
  VertexID last_stop;
 
  Node2VecSampler(Node2VecSampler& x) : last_stop(x.last_stop) {}

public:
  float p = 2.0f;
  float q = 0.5f;
  Node2VecSampler():last_stop(-1) {}

  __device__ __forceinline__
  void set_last_stop(VertexID t) {last_stop = t;}
  
  __device__ __host__ __forceinline__
  VertexID get_last_stop() const {return last_stop;}

  virtual size_t size() {return sizeof(Node2VecSampler);}
 
  virtual Sampler* device_cpy()
  {
    CHK_CU(cudaMalloc(&d_ptr, sizeof(Node2VecSampler)));
    CHK_CU(cudaMemcpy(d_ptr, this, sizeof(Node2VecSampler), cudaMemcpyHostToDevice));

    return d_ptr;
  }

  //TODO: Test for last_stop_edges
};

void copy_sampler_to_gpu (Sampler* samplers, size_t num_samplers, Sampler*& device_samplers)
{
  CHK_CU(cudaMalloc(&device_samplers, samplers[0].size()*num_samplers));
  CHK_CU(cudaMemcpy(device_samplers, samplers, samplers[0].size()*num_samplers, cudaMemcpyHostToDevice));
}

void copy_back_sampler_from_gpu (Sampler* samplers, size_t num_samplers, Sampler*& device_samplers)
{
  CHK_CU(cudaMemcpy(samplers, device_samplers, samplers[0].size()*num_samplers, cudaMemcpyDeviceToHost));
}

#endif