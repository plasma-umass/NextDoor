#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include "csr.hpp"

#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#define DIVUP(x,y) (((x) + ((y) - 1))/(y))
#define ROUNDUP(x,y) (DIVUP(x,y)*(y))
#define FULL_WARP_MASK 0xffffffff
#define WARP_SIZE (32)
#define CHK_CU(x) if (utils::is_cuda_error (x, __LINE__) == true) {assert(false);}
#define MAX(x,y) (((x)<(y))?(y):(x))
#define MIN(x,y) (((x)>(y))?(y):(x))
#define CUDA_SYNC_DEVICE_ALL(nextDoorData) {\
for(auto device : nextDoorData.devices) {\
  CHK_CU(cudaSetDevice(device));\
  CHK_CU(cudaDeviceSynchronize());\
}\
}

#define PartDivisionSize(Total,PartIdx,NumParts) (PartIdx < NumParts - 1) ? Total/NumParts : Total - (NumParts - 1)*(Total/NumParts)
#define PartStartPointer(Total,PartIdx,NumParts) PartIdx * (Total/NumParts)

namespace utils {
  template<typename F, typename T> 
  __host__ __device__ __forceinline__
  bool binarySearch (F& array, T x, int size) 
  {
    EdgePos_t l = 0;
    EdgePos_t r = size - 1;
    
    while (l <= r) {
      EdgePos_t m = l + (r-l)/2;
      if (array[m] == x)
        return true;
      
      if (array[m] < x)
        l = m + 1;
      else
        r = m - 1;
    }
    return false;
  }

  template<typename T> 
  __host__ __device__ __forceinline__
  bool linearSearch (const T* array, T x, int size) 
  {
    for (EdgePos_t i = 0; i < size; i++) {
      if (array[i] == x) {
        return true;
      }
    }

    return false;
  }

  __device__
  EdgePos_t atomicAdd (EdgePos_t* ptr, const EdgePos_t val)
  {
    if (sizeof(EdgePos_t) == 8) {
      return (EdgePos_t)::atomicAdd((unsigned long long*)ptr, (unsigned long long)val);
    } else {
      return (EdgePos_t)::atomicAdd((int*)ptr, (int)val);
    }
  }

  double convertTimeValToDouble (struct timeval _time)
  {
    return ((double)_time.tv_sec) + ((double)_time.tv_usec)/1000000.0f;
  }

  template<class T>
  void print_container(T const &s)
  {
      std::copy(s.begin(),
              s.end(),
              std::ostream_iterator<int>(std::cout, " "));
  }

  struct timeval getTimeOfDay ()
  {
    struct timeval _time;

    if (gettimeofday (&_time, NULL) == -1) {
      fprintf (stderr, "gettimeofday returned -1\n");
      perror ("");
      abort ();
    }

    return _time;
  }

  __device__ __host__ bool intervals_intersect (int x1, int x2, int y1, int y2)
  {
    return x1 <= y2 && y1 <= x2;
  }


  bool is_cuda_error (cudaError_t error, int line) 
  {
    //cudaError_t error = cudaGetLastError ();
    if (error != cudaSuccess) {
      const char* error_string = cudaGetErrorString (error);
      std::cout << "Cuda Error: " << error_string << " " << line <<
      std::endl;
      return true;
    }

    return false;
  }

  void get_num_roots_distribution(int* roots_distribution, int* ranges, int num_ranges,
                                  EdgePos_t* src_num_roots, VertexRange src_range)
  {
    for (VertexID src : src_range) {
      EdgePos_t num_roots = src_num_roots[2*src + 1];
      if (num_roots <= ranges[0]) {
      } else {
        for (int i = 1; i < num_ranges - 1; i++) {
          if (num_roots >= ranges[i-1] and num_roots < ranges[i]) {
            roots_distribution[i] += num_roots;
            break;
          }
        }
        
        if (num_roots >= ranges[num_ranges]) {
          roots_distribution[num_ranges] += num_roots;
        }
      }
    }
  }

  template<class T1, class T2>
  inline T1 next_multiple(const T1 val, const T2 divisor)
  {
    if (val%divisor == 0) return val;
    return (val/divisor + 1)*divisor;
  }

  template<class T>
  inline T thread_block_size(const T total, const T tb_size)
  {
    if (total%tb_size == 0)
      return total/tb_size;
    return total/tb_size +1;
  }

  template<class T1, class T2, class T3>
  inline void set_till_next_multiple(T1& val, const T2 divisor, T3* mem, 
                                     const T3 init_val)
  {
    while (val%divisor != 0)
      mem[val++] = -1;
  }

  //Kernel to memset global memory
  template <class T>
  __global__ void memset_kernel(T* mem, T val, size_t nelems)
  {
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    if (id < nelems) {
      mem[id] = val;
    }
  }

  template <class T>
  void gpu_memset(T* mem, T val, size_t nelems)
  {
    const size_t threads = 256;
    const size_t blocks = thread_block_size(nelems, threads);

    memset_kernel<<<blocks, threads>>>(mem, val, nelems);
    CHK_CU(cudaDeviceSynchronize());
  }

  template<class T>
  class RangeIterator
  {
    public:
      class iterator 
      {
      private:
         T it;

      public: 
        iterator(T _it) : it(_it) {}
        iterator operator++() {it++; return *this;}
        iterator operator--() {it--; return *this;}
        T operator*() {return it;}

        bool operator==(const iterator& rhs) {return it == rhs.it;}
        bool operator!=(const iterator& rhs) {return it != rhs.it;}
      };

    private:
      T first;
      T last;

    public:
      RangeIterator(T _first, T _last) : first(_first), last(_last) {}

      iterator begin() const
      {
        return iterator(first);
      }

      iterator end() const
      {
        return iterator(last+1);
      }
  };

  template<class T>
  size_t sizeof_vector(const T& vec)
  {
    return vec.size()*sizeof(vec[0]);
  }

#if 0
  void bfs (CSR* csr) 
  {
    std::queue <VertexID> bfs_queue;
    bool* seen = new bool [csr->get_n_vertices ()];
    memset (seen, 0, csr->get_n_vertices ());

    for (VertexID v = 0; v < csr->get_n_vertices (); v++) {
      if (seen[v] == true) 
        continue;
        
      bfs_queue.push (v);

      while (!bfs_queue.empty ()) {
        VertexID  v = bfs_queue.front ();
        bfs_queue.pop ();
        EdgePos_t s = csr->get_start_edge_idx (v);
        const EdgePos_t e = csr->get_end_edge_idx (v);

        while (s <= e) {
          EdgePos_t u = csr->get_edges ()[s];
          if (seen [u] == false) {
            bfs_queue.push (u);
            seen[u] = true;
          }
          s++;
        }
      }
    }
  }
#endif

  enum StorageLocationType
  {
    Host,
    Device, 
  };

  template<class T>
  class Array
  {
    private:
      T* data_;
      
      StorageLocationType storageLocationType_;

      size_t nelems_;
    public:
      Array (StorageLocationType storageLocation, size_t nelems) : Array(nullptr, storageLocation, nelems)
      {
      }

      Array (T* ptr, StorageLocationType storageLocation, size_t nelems) : data_(ptr), storageLocationType_(storageLocation), nelems_(nelems)
      {
      }

      Array & operator=(const Array& a) 
      {
        nelems_ = a.nelems();
        storageLocationType_ = a.location();
        allocate();
        copy(a.data(), a.nelems());
        memcpy(data_, a.data(), sizeof(T)*nelems_);
      }

      void copy(T* dst, size_t nelems)
      {
        assert(nelems == nelems_);

        if (location() == Device) {
          assert(false);
        } else {
          memcpy(data_, dst, sizeof(T)*nelems);
        }
      }

      T* data() const {return data_;}
      inline size_t nelems() const {return nelems_;}
      inline StorageLocationType location() const {return storageLocationType_;}
      T& operator[](size_t idx) 
      {
        assert(idx < nelems());

        return data_[idx];
      }

      void allocate()
      {
        if (location() == Device) {
          assert(false); //TODO
        } else {
          data_ = new T[nelems()];
        }
      }

      void free()
      {
        if (location() == Device) {
          assert(false); //TODO
        } else {
          delete[] data_;
        }
      }

      ~Array()
      {
        free();
      }
  };
}

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  abort();}} while(0)

namespace GPUUtils {
  typedef uint32_t ShMemEdgePos_t;

  enum SourceVertexExec_t
  {
    BlockLevel,
    DeviceLevel
  };

  const uint FULL_MASK = 0xffffffff;

  void printCudaMemInfo() 
  {
    size_t free = 0, total = 0;
    CHK_CU(cudaMemGetInfo(&free, &total));
    printf("free memory %ld\n", free);
  }

  __device__ inline int get_warp_mask_and_participating_threads (int condition, int& participating_threads, int& first_active_thread)
  {
    uint warp_mask = __ballot_sync(FULL_MASK, condition);
    first_active_thread = -1;
    participating_threads = 0;
    int qq = 0;
    while (qq < 32) {
      if ((warp_mask & (1U << qq)) == (1U << qq)) {
        if (first_active_thread == -1) {
          first_active_thread = qq;
        }
        participating_threads++;
      }
      qq++;
    }

    return warp_mask;
  }
  
  CSRPartition copyPartitionToGPU(CSRPartition& part, GPUCSRPartition& gpuCSRPartition)
  {
    //TODO: Store gpuCSRPartition in Constant Memory  
    CHK_CU(cudaMalloc(&gpuCSRPartition.device_vertex_array, sizeof(CSR::Vertex)*part.get_n_vertices ()));
    CHK_CU(cudaMalloc(&gpuCSRPartition.device_edge_array, sizeof(CSR::Edge)*part.get_n_edges ()));
    CHK_CU(cudaMalloc(&gpuCSRPartition.device_weights_array, sizeof(float)*part.get_n_edges ()));
    CHK_CU(cudaMemcpy(gpuCSRPartition.device_vertex_array, part.vertices, sizeof(CSR::Vertex)*part.get_n_vertices (), cudaMemcpyHostToDevice));
    CHK_CU(cudaMemcpy(gpuCSRPartition.device_edge_array, part.edges, sizeof(CSR::Edge)*part.get_n_edges (), cudaMemcpyHostToDevice));
    CHK_CU(cudaMemcpy(gpuCSRPartition.device_weights_array, part.weights, sizeof(float)*part.get_n_edges (), cudaMemcpyHostToDevice));

    CSRPartition device_csr_partition_value = CSRPartition(part.first_vertex_id,
                                                           part.last_vertex_id, 
                                                           part.first_edge_idx, part.last_edge_idx, 
                                                           gpuCSRPartition.device_vertex_array, 
                                                           gpuCSRPartition.device_edge_array,
                                                           gpuCSRPartition.device_weights_array);
    return device_csr_partition_value;
  }
  
  __host__ __device__ int num_edges_to_warp_size (const EdgePos_t n_edges, SourceVertexExec_t src_vertex_exec) 
  {
    //Different warp sizes gives different performance. 32 is worst. adapative is a litter better.
    //Best is 4.
  #ifdef RANDOM_WALK
    return 1;
  #else
    if (src_vertex_exec == SourceVertexExec_t::BlockLevel) {
      //TODO: setting this to 4,8,or 16 gives error.
      if (n_edges < 4) 
        return 2;
      if (n_edges < 8)
        return 4;
      if (n_edges < 16)
        return 8;
      if (n_edges < 32)
        return 16;
      else
        return 32;
    } else {
      return 32;
    }
  #endif
  }

  float* gen_rand_on_gpu(size_t n_rands)
  {
    float* device_rand;
    cudaMalloc(&device_rand, n_rands*sizeof(float));
    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, clock()));
    CURAND_CALL(curandGenerateUniform(gen, device_rand, n_rands));

    return device_rand;
  }


  template<typename T>
  void printDeviceArray(T* array, int nelems, char sep)
  {
    T* tmp = new T[nelems];

    CHK_CU(cudaMemcpy(tmp, array, nelems*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nelems; i++) {
      if (tmp[i] == 4549983 || tmp[i] == 4017644)
      std::cout << i << ":" << tmp[i] << sep;
    }

    std::cout << std::endl;

    delete tmp;
  }

  template<typename T1, typename T2>
  void printKeyValuePairs(T1* keys, T2* values, int nelems, char sep)
  {
    for (int i = 0; i < nelems; i++) {
      if (values[i] == 4439958 || keys[i] == 4549983)
        std::cout << i << ": [" << keys[i] << ", " << values[i] << "]" << sep;
    }

    std::cout << std::endl;
  }

  template<typename T1, typename T2>
  void printDeviceKeyValuePairs(T1* keys, T2* values, int nelems, char sep)
  {
    T1* tmp1 = new T1[nelems];
    T2* tmp2 = new T2[nelems];

    CHK_CU(cudaMemcpy(tmp1, keys, nelems*sizeof(T1), cudaMemcpyDeviceToHost));
    CHK_CU(cudaMemcpy(tmp2, values, nelems*sizeof(T2), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nelems; i++) {
      if (tmp1[i] == 4549983 || tmp1[i] == 4017644)
        std::cout << i << ": [" << tmp1[i] << ", " << tmp2[i] << "]" << sep;
    }

    std::cout << std::endl;
    delete tmp1;
    delete tmp2;
  }

  template<typename T>
  T* copyDeviceMemToHostMem(T* devPtr, size_t nelems)
  {
    T* hPtr = new T[nelems];

    CHK_CU(cudaMemcpy(hPtr, devPtr, nelems*sizeof(T), cudaMemcpyDeviceToHost));

    return hPtr;
  }
}

namespace LoadBalancing {
  enum LoadBalancingThreshold{
    GridLevel = 256,
    BlockLevel = 32,
    SubWarpLevel = 8,
    IdentityKernel = 0
  };

  enum LoadBalancingTBSizes {
    GridLevelTBSize = 512,
    BlockLevelTBSize = 512,
    SubWarpLevelTBSize = 32,
  };

  const bool EnableLoadBalancing = false;

  bool is_grid_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots >= LoadBalancingThreshold::GridLevel;
  }

  bool is_block_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots < LoadBalancingThreshold::GridLevel and num_roots >= LoadBalancingThreshold::BlockLevel;
  }

  bool is_subwarp_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots < LoadBalancingThreshold::SubWarpLevel;
  }

  void num_gpu_threads(const VertexRange src_range, const EdgePos_t* src_num_roots, 
                       EdgePos_t& num_grid_threads, EdgePos_t& num_block_threads, 
                       EdgePos_t& num_subwarp_threads)
  {
    num_grid_threads = 0;
    num_block_threads = 0;
    num_subwarp_threads = 0;
    for (VertexID src : src_range) {
      EdgePos_t num_roots = src_num_roots[2*src + 1];
      if (EnableLoadBalancing and is_grid_level_assignment(num_roots)) {
        num_grid_threads += num_roots;
        num_grid_threads = utils::next_multiple(num_grid_threads, 
                                                GridLevelTBSize);
      } else if (is_block_level_assignment(num_roots)) {
        num_block_threads += num_roots;
        // num_block_threads = utils::next_multiple(num_block_threads, 
        //                                          BlockLevelTBSize);
      } else if (is_subwarp_level_assignment(num_roots)) {
        num_subwarp_threads += num_roots;
        // num_subwarp_threads = utils::next_multiple(num_subwarp_threads, 
        //                                            SubWarpLevelTBSize);
      }
    }
  }

};

#endif