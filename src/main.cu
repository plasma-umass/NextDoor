#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <unordered_set>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <string.h>
#include <assert.h>
#include <tuple>

//citeseer.graph
//const int N = 3312;
//const int N_EDGES = 9074;
//micro.graph
const int N = 100000;
const int N_EDGES = 2160312;
typedef uint32_t VertexID;

#include "csr.hpp"
#include "utils.hpp"

//#define USE_FIXED_THREADS
#define MAX_CUDA_THREADS (96*96)
#define THREAD_BLOCK_SIZE 256
#define WARP_SIZE 32
//#define ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING false

#define ENABLE_GRAPH_PARTITION_IN_SHARED_MEM
#if defined (ALL_THREAD_BLOCK_EMBEDDINGS_IN_SHARED_MEM_PER_VERTEX)  && !defined (EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK) 
  #error "For ALL_THREAD_BLOCK_EMBEDDINGS_IN_SHARED_MEM_PER_VERTEX, EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK should be defined "
#endif
#define USE_EMBEDDING_IN_LOCAL_MEM
//#define PROCESS_EMBEDDINGS_PER_VERTEX
#define GPU_QUERY_WAIT_TIME 1000UL

//#define ADD_TO_OUTPUT
//#define SHARED_MEM_NON_COALESCING
/**
  * The commit performing better is 698368fa19d023e3cb09705d820d333f79d0bf46.
  */
#ifdef SHARED_MEM_NON_COALESCING
  #ifndef USE_EMBEDDING_IN_SHARED_MEM
    #error "USE_EMBEDDING_IN_SHARED_MEM must be enabled with SHARED_MEM_NON_COALESCING"
  #endif
#endif
#ifdef USE_EMBEDDING_IN_SHARED_MEM
  #ifdef USE_FIXED_THREADS
    #error "USE_FIXED_THREADS cannot be enabled with USE_EMBEDDING_IN_SHARED_MEM"
  #endif
#endif

#define NEW_EMBEDDING_BUFFER_SIZE 128*1024*1024 //Size in terms of Bytes //Setting it to 128 MB makes citeseer performs a lot better

#define GRAPH_PARTITION_SIZE (48*1024) //24 KB is the size of each partition of graph

//#define USE_CONSTANT_MEM

typedef uint8_t SharedMemElem;

#define ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING false


//#define ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING true

const int N_THREADS = 256;

class GlobalMemAllocator
{
  static uint64_t memory_length;
  static uint64_t bump_pointer;
  static char* global_mem_ptr;

  public:

    static void initialize (char* _global_mem_ptr, uint64_t _memory_length) {
      global_mem_ptr = _global_mem_ptr;
      memory_length = _memory_length;
      bump_pointer = 0;
    }

    static uint64_t alloc (size_t sz) {
      assert (bump_pointer + sz < memory_length);

      uint64_t to = bump_pointer;

      bump_pointer += sz;

      return to;
    }

    static uint64_t allocated () {
      return bump_pointer;
    }
    static uint64_t alloc_vertices_array (size_t n_vertices) {
      return alloc (sizeof (VertexID)*n_vertices);
    }
    
    static void* get_global_mem_ptr () {
      return global_mem_ptr;
    }

};

uint64_t GlobalMemAllocator::memory_length;
uint64_t GlobalMemAllocator::bump_pointer;
char* GlobalMemAllocator::global_mem_ptr;

class VectorVertexEmbedding
{
private:
  uint64_t array_start_idx;
  uint32_t filled_size;
  uint32_t size;
  VertexID* array;

public:
  __host__
  VectorVertexEmbedding (uint32_t _max_size, uint64_t _array_start_idx, bool filled = false)
  {
    size = _max_size;
    filled_size = filled ? size : 0;
    array_start_idx = _array_start_idx;
    array = (VertexID*)((char*) GlobalMemAllocator::get_global_mem_ptr () + array_start_idx);
  }

  __host__ 
  std::vector<VertexID> to_vector ()
  {
    std::vector<VertexID> v;
    
    for (int i = 0; i < get_n_vertices (); i++) {
      v.push_back (get_vertex (i));
    }

    return v;
  }

  __host__
  void* get_array () {return array;}
  __device__ __host__
  uint64_t get_array_start_idx () { return array_start_idx;}
  
  __host__ __device__
  void add (int v)
  {
  #if DEBUG
    if (!(size != 0 and filled_size < size)) {
      printf ("filled_size %d, size %d\n", filled_size, size);
      //assert (size != 0 and filled_size < size);
      assert (false);
    }
  #endif
  
    add_unsorted (v);
  }

  __host__ __device__
  void add_last_in_sort_order () 
  {
    int v = array[filled_size-1];
    remove_last ();
    add (v);
  }

  __host__ __device__
  void add_unsorted (int v) 
  {
    array[filled_size++] = v;
  }
  
  __host__ __device__
  void remove (int v)
  {
    printf ("Do not support remove\n");
    assert (false);
  }
  
  // __host__ __device__
  // const bool has_logn (int v)
  // {
  //   int l = 0;
  //   int r = filled_size-1;
    
  //   while (l <= r) {
  //     int m = l+(r-l)/2;
      
  //     if (array[m] == v)
  //       return true;
      
  //     if (array[m] < v)
  //       l = m + 1;
  //     else
  //       r = m - 1;
  //   }
    
  //   return false;
  // }
  
  __host__ __device__
  bool has (int v) const
  {
    for (int i = 0; i < filled_size; i++) {
      if (array[i] == v) {
        return true;
      }
    }
    
    return false;
  }
  
  __host__ __device__
  size_t get_n_vertices () const
  {
    return filled_size;
  }
  
  __host__ __device__
  int get_vertex (int index, void* global_storage_start) const
  {
    return ((VertexID*)((char*)global_storage_start + array_start_idx))[index];
  }

  __device__
  int get_vertex (int index, void* global_storage_start, uint64_t global_start_idx) const
  {
    assert (array_start_idx >= global_start_idx);
    return ((VertexID*)((char*)global_storage_start + (array_start_idx - global_start_idx)))[index];
  }

  __host__ 
  int get_vertex (int index) const
  {
    return array[index];
  }
  
  __host__ __device__
  int get_last_vertex () const
  {
    return array[filled_size-1];
  }
  
  __host__ __device__
  size_t get_max_size () const
  {
    return size;
  }
  
  __host__ __device__
  void clear ()
  {
    filled_size = 0;
  }
  
  __host__ __device__
  void remove_last () 
  {
    assert (filled_size > 0);
    filled_size--;
  }
  __host__ __device__
  ~VectorVertexEmbedding ()
  {
    //delete[] array;
  }

  void print () 
  {
    std::cout << "[";
    for (int i = 0; i < filled_size; i++) {
      std::cout << get_vertex (i) << ", ";
    }
    std::cout << "]";
  }
};

std::vector<VectorVertexEmbedding> get_initial_embedding_vector (CSR* csr)
{
  VectorVertexEmbedding embedding (0, 0UL);
  std::vector <VectorVertexEmbedding> embeddings;

  embeddings.push_back (embedding);

  return embeddings;
}

__host__
void vector_embedding_from_one_less_size (VectorVertexEmbedding const & in,
                                          VectorVertexEmbedding& out)
{
  //TODO: Optimize here, filled_size++ in add is being called several times
  //but can be called only once too
  //if  (false and vec_emb1.get_n_vertices () != size) {
  //  printf ("vec_emb1.get_n_vertices () %ld != size %d\n", vec_emb1.get_n_vertices (), size);
  //  assert (false);
  //}
  assert (in.get_n_vertices () <= out.get_n_vertices ());
  for (int i = 0; i < in.get_n_vertices (); i++) {
    out.add (in.get_vertex (i));
  }
}

std::vector<VectorVertexEmbedding> get_extensions_vector (VectorVertexEmbedding& embedding, CSR* csr)
{
  std::vector<VectorVertexEmbedding> extensions;
  size_t size;
  
  size = embedding.get_n_vertices ();

  if (size == 0) {
    for (int u = 0; u < N; u++) {
      uint64_t ptr =  GlobalMemAllocator::alloc_vertices_array(1);
      VectorVertexEmbedding extension(1,ptr);
      extension.add(u);
      extensions.push_back(extension);
    }
  } else {
    for (int i = 0; i < size; i++) {
      int u = embedding.get_vertex (i);
      for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
        int v = csr->get_edges () [e];
        if (embedding.has (v) == false) {
          VectorVertexEmbedding extension(1, GlobalMemAllocator::alloc_vertices_array(size + 1));
          vector_embedding_from_one_less_size (embedding, extension);
          extension.add(v);
          extensions.push_back(extension);
        }
      }
    }
  }

  return extensions;
}

void run_single_step_initial_vector (std::vector<VectorVertexEmbedding>& input_embeddings,
                                     CSR* csr,
                                     std::vector<VectorVertexEmbedding>& output_embeddings,
                                     std::vector<VectorVertexEmbedding>& next_step_embeddings)
{
  for (int i = 0; i < input_embeddings.size (); i++) {
    VectorVertexEmbedding& embedding = input_embeddings[i];
    std::vector<VectorVertexEmbedding> extensions = get_extensions_vector (embedding, csr);
    for (auto extension : extensions) {
        output_embeddings.push_back (extension);
        next_step_embeddings.push_back (extension);
      }
   }
}

std::vector <std::vector <VertexID>> n_hop_cpu (CSR* csr, const int N_HOPS)
{
  std::vector <std::vector <VertexID>> hops = std::vector<std::vector<VertexID>> (csr->get_n_vertices ());

  int hop = 0;

  for (int vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    int start_edge_idx = csr->get_start_edge_idx (vertex);
    int end_edge_idx = csr->get_end_edge_idx (vertex);
    if (start_edge_idx != -1) {
      for (int edge = start_edge_idx; edge <= end_edge_idx; edge++) {
        hops[vertex].push_back (csr->get_edges()[edge]);
      }
    }
  }

  for (int vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    int hop = 1;
    std::vector <VertexID> vertex_hops[N_HOPS + 1];
    vertex_hops[0].insert (vertex_hops[0].begin(), hops[vertex].begin (), hops[vertex].end ());
    while (hop < N_HOPS) {
      for (int hop_vertex : vertex_hops[hop - 1]) {
        int start_edge_idx = csr->get_start_edge_idx (hop_vertex);
        int end_edge_idx = csr->get_end_edge_idx (hop_vertex);
        
        if (start_edge_idx != -1) {
          for (int edge = start_edge_idx; edge <= end_edge_idx; edge++) {
            int v = csr->get_edges()[edge];
            vertex_hops[hop].push_back (v);
          }
        }
      }

      hops[vertex].insert (hops[vertex].begin (), vertex_hops[hop].begin (), vertex_hops[hop].end ());
      hop++;
    }
  }

  return hops;
}

#define MAX_LOAD_PER_TB (N_THREADS)
#define MAX_VERTICES_PER_TB 10
#if MAX_VERTICES_PER_TB < 1
  #error "MAX_VERTICES_PER_TB should be greater than or equal to 1"
#endif

#define WARP_HOP

const uint FULL_MASK = 0xffffffff;

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

__device__ int n_edges_to_warp_size (const int n_edges) 
{
  //Different warp sizes gives different performance. 32 is worst. adapative is a litter better.
  //Best is 4.
  return 4;
  if (n_edges <= 4) 
    return 2;
  else if (n_edges > 4 && n_edges <= 8)
    return 4;
  else if (n_edges > 8 && n_edges <= 16)
    return 8;
  else if (n_edges > 16 && n_edges <= 32) 
    return 16;
  else
    return 32;
}

#define MAX_EDGES (2*MAX_LOAD_PER_TB)
#undef USE_PARTITION_FOR_SHMEM
#define MAX_HOP_VERTICES_IN_SH_MEM (MAX_VERTICES_PER_TB)
#define ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM

__global__ void get_max_lengths_for_vertices_first_iter (CSRPartition* void_csr,
                                                          int start_vertex, int end_vertex,
                                                          unsigned long long int* embeddings_additions_iter,
                                                          void* void_map_orig_embedding_to_additions)
{
  CSRPartition* csr = (CSRPartition*)void_csr;

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  VertexID vertex = thread_idx + start_vertex;
  if (vertex > end_vertex) {
    return;
  }

  int* map_orig_embedding_to_additions = (int*) void_map_orig_embedding_to_additions;
  unsigned long long int new_edges = 0;

  /*Perform a single hop for all vertices in the input embedding*/
  const int start_edge_idx = csr->get_start_edge_idx (vertex);
  const int end_edge_idx = csr->get_end_edge_idx (vertex);

  if (end_edge_idx != -1) {
    int e = (end_edge_idx - start_edge_idx) + 1;
    if (e < 0) {
      printf ("v %d s %d e %d\n", vertex, start_edge_idx, end_edge_idx);
    }
    assert (e >= 0);
    new_edges += e;
  }
  

  unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
  map_orig_embedding_to_additions[2*thread_idx] = additions_start_iter;
  map_orig_embedding_to_additions[2*thread_idx+1] = new_edges;
}

__global__ void get_max_lengths_for_vertices_single_step (CSRPartition* void_csr,
                                                          int start_vertex, int end_vertex,
                                                          unsigned long long int* void_embeddings_additions_iter,
                                                          void* void_map_orig_embedding_to_additions_prev_iter,
                                                          void* void_map_orig_embedding_to_additions_next_iter,
                                                          void* void_map_orig_embedding_to_additions_first_iter,
                                                          int* edges_to_prev_iter_additions,
                                                          int common_vertex_with_previous_partition,
                                                          int common_vertex_with_next_partition)
{
  CSRPartition* csr = (CSRPartition*)void_csr;

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  VertexID vertex = thread_idx + start_vertex;
  if (vertex > end_vertex) {
    return;
  }

  //VertexID* embedding_storage = (VertexID embedding_storage;
  unsigned long long int* embeddings_additions_iter = void_embeddings_additions_iter;
  int* map_orig_embedding_to_additions_next_iter = (int*)void_map_orig_embedding_to_additions_next_iter;
  int* map_orig_embedding_to_additions_prev_iter = (int*)void_map_orig_embedding_to_additions_prev_iter;
  int* map_orig_embedding_to_additions_first_iter = (int*) void_map_orig_embedding_to_additions_first_iter;
  unsigned long long int new_edges = 0;//map_orig_embedding_to_additions_first_iter[2*thread_idx + 1];
  // printf ("thread idx %d array_start_idx %ld\n", thread_idx, input_embedding->get_array_start_idx ());
  /*Perform a single hop for all vertices in the input embedding*/
  
  int start_edge_idx = csr->get_start_edge_idx (vertex);
  const int end_edge_idx = csr->get_end_edge_idx (vertex);
  if (end_edge_idx != -1) {
    while (start_edge_idx <= end_edge_idx) {
      int v = csr->get_edge (start_edge_idx);
      if (csr->is_vertex_in_partition (v) and v != common_vertex_with_previous_partition and v != common_vertex_with_next_partition) {
        assert (v-start_vertex >= 0);
        new_edges += map_orig_embedding_to_additions_prev_iter [2*(v-start_vertex)+1];
      }
      else
        new_edges += edges_to_prev_iter_additions[start_edge_idx - csr->first_edge_idx];

      start_edge_idx++;
    }
  }

  //printf ("new_edges %ld\n", new_edges);
  unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
  map_orig_embedding_to_additions_next_iter[2*thread_idx] = additions_start_iter;
  map_orig_embedding_to_additions_next_iter[2*thread_idx+1] = new_edges;
}

__global__ void run_hop_parallel_single_step (int N_HOPS, int hop, CSR* void_csr,
  void* void_embeddings_additions, 
  size_t num_neighbors,
  void* void_embeddings_additions_prev_hop,
  int* map_orig_embedding_to_additions,
  int* previous_stage_filled_range,
  int* hop_vertex_to_roots,
  int* map_vertex_to_hop_vertex_data,
  int* source_vertex_idx,
  unsigned long long int* profile_branch_1, unsigned long long int* profile_branch_2)
{
  CSR* csr = (CSR*)void_csr;
  __shared__ int vertices[MAX_VERTICES_PER_TB];
  __shared__ int previous_step_end[MAX_VERTICES_PER_TB];
  __shared__ int n_vertex_load;
  __shared__ int thread_idx_to_load[2*MAX_LOAD_PER_TB];
  __shared__ int last_hop_vertex_id;
  __shared__ int last_hop_vertex_roots_remaining;
  __shared__ int last_hop_vertex_roots_done;

#ifdef USE_PARTITION_FOR_SHMEM
  __shared__ VertexID shmem_csr_edges[MAX_EDGES];
  __shared__ int hop_vertex_in_shared_mem[MAX_HOP_VERTICES_IN_SH_MEM];
  __shared__ int hop_vertices_in_shared_mem_start_edge_idx[MAX_HOP_VERTICES_IN_SH_MEM];
  __shared__ int hop_vertices_in_shared_mem_end_edge_idx[MAX_HOP_VERTICES_IN_SH_MEM];
  __shared__ int hop_vertices_in_shared_mem_size;
  __shared__ int shmem_csr_edges_size;
#endif 

  int laneid = threadIdx.x%warpSize;
  int warpid = threadIdx.x/warpSize;

  VertexID* embeddings_additions = (VertexID*)void_embeddings_additions;
  VertexID* embeddings_additions_prev_hop = (VertexID*)void_embeddings_additions_prev_hop;
  int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (hop != 0) {
    thread_idx_to_load [2*threadIdx.x] = -1;
    thread_idx_to_load [2*threadIdx.x + 1] = -1;

    __syncthreads ();

    if (threadIdx.x == 0) {
      last_hop_vertex_id = -1;
      last_hop_vertex_roots_remaining = -1;
      int load = 0;
      n_vertex_load = 0;
      int load_assigned_index = 0;
      int warp_assigned = 0;

#ifdef USE_PARTITION_FOR_SHMEM
      hop_vertices_in_shared_mem_size = 0;
      int edges_in_shared_mem = 0;
      shmem_csr_edges_size = 0;
#endif

      while (n_vertex_load < MAX_VERTICES_PER_TB && load < MAX_LOAD_PER_TB) {
        vertices[n_vertex_load] = atomicAdd(source_vertex_idx, 1);
        if (vertices[n_vertex_load] >= gridDim.x) {
          break;
        }
        
        int start_edge_idx = csr->get_start_edge_idx (vertices[n_vertex_load]);
        const int end_edge_idx = csr->get_end_edge_idx (vertices[n_vertex_load]);
        const int n_edges = (end_edge_idx != -1) ? (end_edge_idx - start_edge_idx + 1) : 0;
#ifdef USE_PARTITION_FOR_SHMEM
        if (hop_vertices_in_shared_mem_size < MAX_HOP_VERTICES_IN_SH_MEM && n_edges != 0 && 
            n_edges + edges_in_shared_mem < MAX_EDGES) {
          int v = vertices[n_vertex_load];
          hop_vertex_in_shared_mem[hop_vertices_in_shared_mem_size] = v;
          edges_in_shared_mem += n_edges;
          hop_vertices_in_shared_mem_size++;
        }
#endif
        int shfl_warp_size = n_edges_to_warp_size(n_edges);
        int root_vertices = map_vertex_to_hop_vertex_data[2*vertices[n_vertex_load] + 1];

        if (root_vertices != 0 and n_edges != 0) {
          int root_vertex_idx;
          for (root_vertex_idx = 0; root_vertex_idx < root_vertices && warp_assigned < MAX_LOAD_PER_TB; root_vertex_idx++) {
            for (int ii = warp_assigned; ii < min (warp_assigned + shfl_warp_size, MAX_LOAD_PER_TB); ii++) {
              thread_idx_to_load[2*ii] = n_vertex_load;
              thread_idx_to_load[2*ii+1] = root_vertex_idx;
            }
            warp_assigned += shfl_warp_size;
            load_assigned_index += 1;
          }

          if (warp_assigned >= MAX_LOAD_PER_TB) {
            last_hop_vertex_roots_remaining = root_vertices - root_vertex_idx;
            last_hop_vertex_roots_done = root_vertex_idx;
            last_hop_vertex_id = n_vertex_load;
          }

          load += root_vertices*shfl_warp_size;
          n_vertex_load++;
        }
      }
    }
    
    __syncthreads ();

#ifdef USE_PARTITION_FOR_SHMEM
    for (int __hop = 0; __hop < hop_vertices_in_shared_mem_size/(blockDim.x/warpSize) + 1; __hop++) {
      int hop = __hop * (blockDim.x/warpSize) + warpid;
      if (hop >= hop_vertices_in_shared_mem_size) {
        continue;
      }

      int start_edge_idx = csr->get_start_edge_idx (hop_vertex_in_shared_mem[hop]);
      const int end_edge_idx = csr->get_end_edge_idx (hop_vertex_in_shared_mem[hop]);
      const int n_edges = (end_edge_idx != -1) ? (end_edge_idx - start_edge_idx + 1) : 0;
      assert (n_edges > 0);
      int _shmem_start = -1;
      if (laneid == 0) {
        _shmem_start = atomicAdd (&shmem_csr_edges_size, n_edges);
      }

      int shmem_start = __shfl_sync (FULL_MASK, _shmem_start, 0, warpSize);
      assert (shmem_start != -1);
      for (int e = 0; e < n_edges/warpSize + 1; e++) {
        int edge_idx = e*warpSize + laneid;
        if (edge_idx < n_edges) {
          shmem_csr_edges[shmem_start + edge_idx] = csr->get_edges ()[start_edge_idx + edge_idx];
        }
      }
      __syncwarp ();
      if (laneid == 0) {
        hop_vertices_in_shared_mem_start_edge_idx[hop] = shmem_start;
        hop_vertices_in_shared_mem_end_edge_idx[hop] = shmem_start + n_edges - 1;
      }

      __syncwarp ();
    }
#endif

    __syncthreads ();

    assert (n_vertex_load <= MAX_VERTICES_PER_TB);
    int _curr_vertex_id = thread_idx_to_load[2*threadIdx.x];
    int root_vertex_idx = thread_idx_to_load[2*threadIdx.x + 1];
    
    int hop_vertex_start_idx = -1;
    int n_root_vertices = -1;
    int root_vertex = -1;
    int hop_idx = -1;
    int first_active_thread = -1;
    int participating_threads = 0;

    if (_curr_vertex_id != -1 && root_vertex_idx != -1 && vertices[_curr_vertex_id] < gridDim.x) {
      hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*vertices[_curr_vertex_id]];
      n_root_vertices = map_vertex_to_hop_vertex_data[2*vertices[_curr_vertex_id] + 1];
      root_vertex = hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx];

      hop_idx = hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx + 1];

      if (root_vertex != -1 && root_vertex < gridDim.x) {
        int vertex = root_vertex;
        int start = map_orig_embedding_to_additions[2*vertex];
      }
    }

    __syncthreads ();

    uint warp_hop_mask = get_warp_mask_and_participating_threads (_curr_vertex_id != -1 && 
      vertices[_curr_vertex_id] < gridDim.x && root_vertex_idx != -1 && root_vertex != -1 && root_vertex < gridDim.x, participating_threads, first_active_thread);
      //__syncthreads ();
    if (_curr_vertex_id != -1 && root_vertex_idx != -1 && vertices[_curr_vertex_id] < gridDim.x) {
      if (root_vertex != -1 && root_vertex < gridDim.x) {
        int vertex = root_vertex;
        int start = map_orig_embedding_to_additions[2*vertex];
        int hop_vertex = embeddings_additions_prev_hop[hop_idx];
       
        assert (hop_vertex == vertices[_curr_vertex_id]);
        int start_edge_idx = csr->get_start_edge_idx (hop_vertex);
        const int end_edge_idx = csr->get_end_edge_idx (hop_vertex);
        __syncwarp (warp_hop_mask);


        int* end = &previous_stage_filled_range[2*vertex + 1];
        if (end_edge_idx != -1) {
          __syncwarp (warp_hop_mask);
          int e = -1;
          const int n_edges = end_edge_idx - start_edge_idx + 1;
          int shfl_warp_size = n_edges_to_warp_size(n_edges);
          if (laneid%shfl_warp_size == 0) {
            e = atomicAdd (end, n_edges);
          }
          //TODO: Add synchronization point
          int _e = __shfl_sync (warp_hop_mask, e, 0, shfl_warp_size);  
          assert (_e != -1);
#ifdef USE_PARTITION_FOR_SHMEM
          if (_curr_vertex_id >= hop_vertices_in_shared_mem_size) {
            int iter = 0;
            while (start_edge_idx + laneid%shfl_warp_size <= end_edge_idx) {
              VertexID edge = csr->get_edges ()[start_edge_idx + laneid%shfl_warp_size];
              embeddings_additions[start + _e + iter*shfl_warp_size + laneid%shfl_warp_size] = edge;
              start_edge_idx += shfl_warp_size;
              iter++;
            }
          } else {
            int iter = 0;
            int _start_edge_idx = hop_vertices_in_shared_mem_start_edge_idx[_curr_vertex_id];
            int _end_edge_idx = hop_vertices_in_shared_mem_end_edge_idx[_curr_vertex_id];
            assert (hop_vertex == hop_vertex_in_shared_mem[_curr_vertex_id]);
            assert (n_edges == (_end_edge_idx - _start_edge_idx) + 1);
            while (_start_edge_idx + laneid%shfl_warp_size <= _end_edge_idx) {
              VertexID edge = shmem_csr_edges[_start_edge_idx + laneid%shfl_warp_size];
              embeddings_additions[start + _e + iter*shfl_warp_size + laneid%shfl_warp_size] = edge;
              _start_edge_idx += shfl_warp_size;
              iter++;
            }
          }
#else
          int iter = 0;
          while (start_edge_idx + laneid%shfl_warp_size <= end_edge_idx) {
            VertexID edge = csr->get_edges ()[start_edge_idx + laneid%shfl_warp_size];
            int addr = start + _e + iter*shfl_warp_size + laneid%shfl_warp_size;
            //assert (addr < ) //TODO: Add asserts.
            embeddings_additions[addr] = edge;
            start_edge_idx += shfl_warp_size;
            iter++;
          }
#endif
        }

        __syncwarp (warp_hop_mask);

      }
    }

    __syncwarp ();
    
    if (last_hop_vertex_id != -1 && last_hop_vertex_roots_remaining != -1) {
      int hop_vertex = vertices[last_hop_vertex_id];
      int hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*hop_vertex];
      int n_root_vertices = map_vertex_to_hop_vertex_data[2*hop_vertex + 1];
      
      __syncthreads ();
      
      for (int i = 0; i < last_hop_vertex_roots_remaining/blockDim.x + 1; i++) {
        int root_idx = i*blockDim.x + threadIdx.x;
        if (root_idx >= last_hop_vertex_roots_remaining) {
          continue;
        }
        
        int root_vertex = hop_vertex_to_roots[hop_vertex_start_idx + 2*(root_idx + last_hop_vertex_roots_done)];
        int hop_idx = hop_vertex_to_roots[hop_vertex_start_idx + 2*(root_idx + last_hop_vertex_roots_done) + 1];
        int start = map_orig_embedding_to_additions[2*root_vertex];
        int* end = &previous_stage_filled_range[2*root_vertex + 1];
        int hop_vertex = embeddings_additions_prev_hop[hop_idx];
        int start_edge_idx = csr->get_start_edge_idx (hop_vertex);
        const int end_edge_idx = csr->get_end_edge_idx (hop_vertex);

        if (end_edge_idx != -1) {
          int e = atomicAdd (end, end_edge_idx - start_edge_idx + 1);
          int iter = 0;
          
          while (start_edge_idx <= end_edge_idx) {
            VertexID edge = csr->get_edges ()[start_edge_idx];
            if (root_vertex == 3030 and edge == 3111) {
              assert (false);
            }
            embeddings_additions[start + e + iter] = edge;
            start_edge_idx++;
            iter++;
          }
        }
      }
      __syncthreads ();
    }
  } else {
    int source_vertex = blockIdx.x;
    assert (source_vertex < csr->get_n_vertices ());
    int start = map_orig_embedding_to_additions[2*source_vertex];
    
    int start_edge_idx = csr->get_start_edge_idx (source_vertex);
    const int end_edge_idx = csr->get_end_edge_idx (source_vertex);
    const int n_edges = end_edge_idx - start_edge_idx + 1;

    if (end_edge_idx != -1) {
      int* end = &previous_stage_filled_range[2*source_vertex + 1];

      for (int i = 0; i < n_edges/blockDim.x + 1; i++) {
        int edge_idx = i*blockDim.x + threadIdx.x;
        if (edge_idx < n_edges) {
          VertexID edge = csr->get_edges()[start_edge_idx + edge_idx];
          int e = atomicAdd (end, 1);
          assert (start + e < num_neighbors);
          embeddings_additions[start + e] = edge;    
        }
      }
    }
  
    __syncthreads ();
    previous_stage_filled_range[2*source_vertex] = start;
  }
}

__global__ void update_filled_ranges (int n_vertices, int* previous_stage_filled_range)
{
  int thread_idx = threadIdx.x + blockDim.x*blockIdx.x;

  if (thread_idx >= n_vertices) 
    return;
  
  previous_stage_filled_range[2*thread_idx] = previous_stage_filled_range[2*thread_idx] + previous_stage_filled_range[2*thread_idx+1];
}

std::vector <std::unordered_set <VertexID>> n_hop_cpu_distinct (CSR* csr, const int N_HOPS)
{
  std::vector <std::unordered_set <VertexID>> hops = std::vector<std::unordered_set<VertexID>> (csr->get_n_vertices ());

  int hop = 0;

  for (int vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    int start_edge_idx = csr->get_start_edge_idx (vertex);
    int end_edge_idx = csr->get_end_edge_idx (vertex);
    if (start_edge_idx != -1) {
      for (int edge = start_edge_idx; edge <= end_edge_idx; edge++) {
        hops[vertex].insert (csr->get_edges()[edge]);
      }
    }
  }

  for (int vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    int hop = 1;
    std::unordered_set <VertexID> vertex_hops[N_HOPS + 1];
    vertex_hops[0].insert (hops[vertex].begin (), hops[vertex].end ());
    while (hop < N_HOPS) {
      for (int hop_vertex : vertex_hops[hop - 1]) {
        int start_edge_idx = csr->get_start_edge_idx (hop_vertex);
        int end_edge_idx = csr->get_end_edge_idx (hop_vertex);
        
        if (start_edge_idx != -1) {
          for (int edge = start_edge_idx; edge <= end_edge_idx; edge++) {
            int v = csr->get_edges()[edge];
            if (hops[vertex].count (v) == 0)
              vertex_hops[hop].insert (v);
          }
        }
      }

      hops[vertex].insert (vertex_hops[hop].begin (), vertex_hops[hop].end ());
      hop++;
    }
  }

  return hops;
}

void copy_partition_to_gpu (CSRPartition& partition, CSRPartition*& device_csr, CSR::Vertex*& device_vertex_array, CSR::Edge*& device_edge_array)
{
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_csr, sizeof(CSRPartition)));
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_vertex_array, sizeof(CSR::Vertex)*partition.get_n_vertices ()));
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_edge_array, sizeof(CSR::Edge)*partition.get_n_edges ()));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_vertex_array, partition.vertices, sizeof (CSR::Vertex)*partition.get_n_vertices (), cudaMemcpyHostToDevice));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_edge_array, partition.edges, sizeof (CSR::Edge)*partition.get_n_edges (), cudaMemcpyHostToDevice));

  CSRPartition device_csr_partition_value = CSRPartition (partition.first_vertex_id, partition.last_vertex_id, 
                                                          partition.first_edge_idx, partition.last_edge_idx, 
                                                          device_vertex_array, device_edge_array);
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_csr, &device_csr_partition_value, sizeof(CSRPartition), cudaMemcpyHostToDevice));
}

int get_common_vertex_with_previous_partition (std::vector<CSRPartition> csr_partitions, int partition_idx)
{
  if (partition_idx <= 0)
    return -1;

  if (csr_partitions[partition_idx].first_vertex_id == csr_partitions[partition_idx - 1].last_vertex_id) {
    return csr_partitions[partition_idx].first_vertex_id;
  }

  return -1;
}


int get_common_vertex_with_next_partition (std::vector<CSRPartition> csr_partitions, int partition_idx)
{
  if (partition_idx >= csr_partitions.size () - 1)
    return -1;

  if (csr_partitions[partition_idx].last_vertex_id == csr_partitions[partition_idx + 1].first_vertex_id) {
    return csr_partitions[partition_idx].last_vertex_id;
  }

  return -1;
}

void create_csr_partitions (CSR* csr, std::vector<CSRPartition>& csr_partitions, const size_t effective_partition_size)
{
  std::vector<std::tuple<VertexID, VertexID, int, int>> vertex_partition_positions_vector;

  //Create Partitions.
  int u = 0;
  int partition_edge_start_idx = 0;

  while (u < csr->get_n_vertices ()) {
    int n_edges = 0;
    int u_start = u;
    int end_edge_idx = 0;
    int u_end = csr->get_n_vertices () - 1;
    int edges = 0;
    int partial_edges = 0;
    for (int v = u; v < csr->get_n_vertices (); v++) {
      int start = csr->get_start_edge_idx (v);
      const int end = csr->get_end_edge_idx (v);
      if (end != -1) {
        if (v == u) {
          //std::cout << "1829: " << " partition_edge_start_idx " << partition_edge_start_idx << " u " << u << " start " << start << " end " << end << std::endl;
        }
        if (v == u && partition_edge_start_idx >= start) {
          start = partition_edge_start_idx;
        }
        edges = end - start + 1;
        assert (edges >= 0);
      } else {
        edges = 0;
      }
      if ((n_edges + edges) * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex) >= effective_partition_size) {
        end_edge_idx = (effective_partition_size - (v-u_start + 1)*sizeof(CSR::Vertex))/sizeof (CSR::Edge) - n_edges;
        //std::cout << " v " << v << " n_edges " << n_edges << " u " << u_start  << "  sizeof (CSR::Edge) " << sizeof (CSR::Edge) <<  " sizeof(CSR::Vertex) " << sizeof(CSR::Vertex) << " end_edge_idx " << end_edge_idx << " effective_partition_size " << effective_partition_size << " start " << start << " end " << end << std::endl;
        if (end_edge_idx <= 0) {
          u = v;
          u_end = v - 1;
          partial_edges = 0;
          end_edge_idx = start - 1;
        } else if (end_edge_idx < edges) {
          u = v;
          u_end = v;
          partial_edges = end_edge_idx;
          end_edge_idx += start - 1; //Including last edge
        } else {
          u_end = v;
          u = v + 1;
          partial_edges = 0;
          end_edge_idx += start - 1; //Including last edge
        }

        if (u_end < u_start) 
        {
          std::cout << "u_end : " << u_end << " u_start: "  << u_start  << std::endl;
          std::cout << "ERROR: Cannot create partition " << std::endl;
          assert (false);
        }

        break;
      }

      n_edges += edges;
    }

    vertex_partition_positions_vector.push_back (std::make_tuple (u_start, u_end, partition_edge_start_idx, (end_edge_idx == 0) ? csr->get_end_edge_idx (u_end) : end_edge_idx));
    //Vertex partition: [u_start, u_end]. Edge partition is all edges from u_start to u_end if end_edge_idx = 0. otherwise all edges of vertices from u_start to u_end - 1 and edges of u_end u_end.start_edge_idx to end_edge_idx.
    
    partition_edge_start_idx = end_edge_idx + 1;

    if (u_end == csr->get_n_vertices () - 1) {
      break;
    }
  }

  std::cout << __LINE__ << ": " << partition_edge_start_idx << " " << csr->get_n_edges () - 1 << std::endl;


  //Create remaining partitions if last vertex's edges are remaining
  if (partition_edge_start_idx != 1 && partition_edge_start_idx < csr->get_n_edges ()) {
    assert ((csr->get_n_edges () - partition_edge_start_idx) * sizeof (CSR::Edge) + (1)*sizeof(CSR::Vertex) <= effective_partition_size);
    vertex_partition_positions_vector.push_back (std::make_tuple (csr->get_n_vertices () - 1, csr->get_n_vertices () - 1, partition_edge_start_idx, csr->get_n_edges ()- 1));
  }

  //Create partitions
  for (auto p : vertex_partition_positions_vector) {
    int u = std::get<0> (p);
    int v = std::get<1> (p);
    int start = std::get<2> (p);
    int end = std::get<3> (p);

    CSR::Vertex* vertex_array = new CSR::Vertex[v - u + 1];
    memcpy (vertex_array, &csr->get_vertices ()[u], (v-u + 1)*sizeof(CSR::Vertex));
    vertex_array[0].set_start_edge_id (start);
    vertex_array[v-u].set_end_edge_id (end);

    CSR::Edge* edge_array = new CSR::Edge[end - start + 1];
    memcpy (edge_array, &csr->get_edges ()[start], (end - start + 1)*sizeof (CSR::Edge));
    CSRPartition part = CSRPartition (u, v, start, end, vertex_array, edge_array);
    csr_partitions.push_back (part);
  }

  /** Check if partitions created are correct**/
  //Sum of edges of all partitions is equal to N_EDGES
  int sum_partition_edges = 0;

  for (int id = 0; id < csr_partitions.size (); id++) {
    auto part = csr_partitions[id];
    std::cout << id << " " << part.last_edge_idx << " " << part.first_edge_idx << " " << part.first_vertex_id << " " << part.last_vertex_id << std::endl;
    if (part.last_edge_idx != -1) {
      sum_partition_edges += part.last_edge_idx - part.first_edge_idx + 1;
    }
  }

  if (!(sum_partition_edges == N_EDGES)) {
    std::cout << __LINE__ <<": "<<sum_partition_edges  << " " << N_EDGES << std::endl;
  }
  assert (sum_partition_edges == N_EDGES);

  int sum_vertices = 0;
  for (int p = 0; p < csr_partitions.size (); p++) {
    if (p > 0 && csr_partitions[p].first_vertex_id == csr_partitions[p-1].last_vertex_id) {
      sum_vertices += csr_partitions[p].last_vertex_id - (csr_partitions[p].first_vertex_id);
    } else {
      sum_vertices += csr_partitions[p].last_vertex_id - csr_partitions[p].first_vertex_id + 1;
    }
  }

  assert (sum_vertices == N);

  int equal_edges = 0;

  /*Check if union of all partitions is equal to the graph*/
  for (int p = 0; p < csr_partitions.size (); p++) {
    int u = csr_partitions[p].first_vertex_id;
    int v = csr_partitions[p].last_vertex_id;
    int end = csr_partitions[p].last_edge_idx;
    int start = csr_partitions[p].first_edge_idx;
    for (int vertex = u; vertex <= v; vertex++) {
      int _start = csr->get_start_edge_idx (vertex);
      if (p > 0 && vertex == csr_partitions[p-1].last_vertex_id) {
        _start = start;
      }
      int _end = csr->get_end_edge_idx (vertex);
      int part_start = csr_partitions[p].get_start_edge_idx (vertex);
      int part_end = csr_partitions[p].get_end_edge_idx (vertex);
      
      if (_end != -1 && part_end != -1) {
        while (_start <= _end && _start <= end && part_start <= part_end) {
          if (!(csr->get_edges ()[_start] == csr_partitions[p].get_edge (part_start))) {
            std::cout << "part_start " << part_start << " part_end " << 
            part_end << " _start " << _start << " _end " << _end << " vertex " 
            << vertex << std::endl;  
            abort ();
          }
          
          equal_edges++;
          part_start++;
          _start++;
        }
      }
    }
  }

  assert (equal_edges == N_EDGES);  
  /********Checking DONE*******/
}


size_t compute_source_to_root_data (std::vector<std::vector<std::pair <VertexID, int>>>& host_src_to_roots,
                                    CSR* csr, int hop, int*** final_map_vertex_to_additions, int** additions_sizes, 
                                    VertexID** neighbors, int* neighbors_sizes,
                                    int*& device_src_to_roots, int*& device_src_to_root_positions)
{
  int* host_src_to_roots_positions = nullptr;
  int *host_src_to_roots_linear = nullptr;

  double t1 = convertTimeValToDouble(getTimeOfDay ());
  host_src_to_roots.clear ();
  //Create per hop vertex data
  for (int v = 0; v < csr->get_n_vertices (); v++) {
    host_src_to_roots.push_back (std::vector<std::pair <VertexID, int> > ());
  }
  for (int v = 0; v < csr->get_n_vertices (); v++) {
    int start = final_map_vertex_to_additions[hop-1][0][2*v];
    int end   = additions_sizes[hop-1][2*v + 1];
    for (int i = 0; i < end; i++) {
      int src = neighbors[hop-1][start + i];
      assert (start + i < neighbors_sizes[hop-1]/sizeof(VertexID));
      assert (src >= 0 && src < N);
      host_src_to_roots[src].push_back (std::make_pair (v, start + i));
    }
  }

  int host_hop_vertex_data_size = 0;

  for (int v = 0; v < csr->get_n_vertices (); v++) {
    host_hop_vertex_data_size += host_src_to_roots[v].size ();
  }

  host_src_to_roots_linear = new int [2*host_hop_vertex_data_size];
  host_src_to_roots_positions = new int[2*csr->get_n_vertices ()];
  int iter = 0;

  for (int v = 0; v < csr->get_n_vertices (); v++) {
    for (int i = 0; i < host_src_to_roots[v].size (); i++) {
      host_src_to_roots_linear[iter + 2*i] = std::get<0> (host_src_to_roots[v][i]);
      host_src_to_roots_linear[iter + 2*i + 1] = std::get<1> (host_src_to_roots[v][i]);
    }

    host_src_to_roots_positions [2*v] = iter;
    host_src_to_roots_positions [2*v + 1] = host_src_to_roots[v].size ();
    iter += 2*host_src_to_roots[v].size ();
  }

  double t2 = convertTimeValToDouble(getTimeOfDay ());
        
  std::cout << "Time taken to create hop vertex data: " << (t2 - t1) << " secs " << std::endl;
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_src_to_roots, 
                                  2*host_hop_vertex_data_size*sizeof (int)));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_src_to_roots, 
                                  host_src_to_roots_linear, 
                                  2*host_hop_vertex_data_size*sizeof (int), 
                                  cudaMemcpyHostToDevice));
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_src_to_root_positions, 
                                  2*csr->get_n_vertices()*sizeof (int)));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_src_to_root_positions,  
                                  host_src_to_roots_positions, 
                                  2*csr->get_n_vertices()*sizeof (int), 
                                  cudaMemcpyHostToDevice));
  delete host_src_to_roots_linear;
  delete host_src_to_roots_positions;

  return host_hop_vertex_data_size;
}

int main (int argc, char* argv[])
{
  std::vector<Vertex> vertices;
  int n_edges = 0;

  if (argc < 2) {
    std::cout << "Arguments: graph-file" << std::endl;
    return -1;
  }

  char* graph_file = argv[1];
  FILE* fp = fopen (graph_file, "r+");
  if (fp == nullptr) {
    std::cout << "File '" << graph_file << "' not found" << std::endl;
    return 1;
  }

  Graph graph (fp);

  fclose (fp);

  std::cout << "n_edges "<<graph.get_n_edges () <<std::endl;
  std::cout << "vertices " << graph.get_vertices ().size () << std::endl; 


  CSR* csr = new CSR(N, N_EDGES);
  std::cout << "sizeof(CSR)"<< sizeof(CSR)<<std::endl;
  csr_from_graph (csr, graph);

  size_t global_mem_size = 15*1024*1024*1024UL;
  #define PINNED_MEMORY
  #ifdef PINNED_MEMORY
    char* global_mem_ptr;
    cudaError_t malloc_error = cudaMallocHost ((void**)&global_mem_ptr, global_mem_size);
    std::cout << "Malloc error: " << cudaGetErrorString (malloc_error) << std::endl;
    assert (malloc_error == cudaSuccess);
  #else
    char* global_mem_ptr = new char[global_mem_size];
  #endif

  std::cout << "Pinned Memory Allocated" << std::endl;
  GlobalMemAllocator::initialize (global_mem_ptr, global_mem_size);

  std::vector<VectorVertexEmbedding> initial_embeddings = get_initial_embedding_vector (csr);
  std::vector<VectorVertexEmbedding> output;
  size_t new_embeddings_size = 0;

  std::vector<VectorVertexEmbedding>& input_embeddings = initial_embeddings;
  std::vector<VectorVertexEmbedding> iter_1_embeddings;
  {
    run_single_step_initial_vector (input_embeddings, csr, output, iter_1_embeddings);
    input_embeddings = iter_1_embeddings;
  }

  double total_stream_time = 0;

  const size_t max_embedding_size_per_iter = (12000000/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
  double_t kernelTotalTime = 0.0;
  std::vector<CSRPartition> csr_partitions;

#ifdef ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM 
  create_csr_partitions (csr, csr_partitions, GRAPH_PARTITION_SIZE - sizeof (CSRPartition));
#else
  CSRPartition full_partition = CSRPartition (0, csr->get_n_vertices () - 1, 0, csr->get_n_edges () - 1, 
                                              csr->get_vertices (), csr->get_edges ());
  csr_partitions.push_back (full_partition);
#endif

  const int N_HOPS = 2;
  
  //Graph on GPU
  CSRPartition* device_csr;
  CSR::Vertex* device_vertex_array;
  CSR::Edge* device_edge_array;
  int* device_vertex_partition_positions;

#if 0
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_vertex_partition_positions, n_partitions*sizeof(int)*2));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_vertex_partition_positions, vertex_partition_positions, n_partitions*sizeof(int)*2, cudaMemcpyHostToDevice));
#endif

  double gpu_time = 0;
  
  std::cout << "Generating additions" << std::endl;
  int* device_additions_sizes;
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_additions_sizes, sizeof(VertexID)*csr->get_n_vertices ()*2));
  EXECUTE_CUDA_FUNC (cudaMemset (device_additions_sizes, 0, sizeof(VertexID)*csr->get_n_vertices ()*2));
  void* device_additions; //Storage to store inputs added to each embedding
  void* device_additions_prev_hop = nullptr;

  int* device_filled_ranges;
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_filled_ranges, sizeof (int)*csr->get_n_vertices ()));
  int* device_prev_thread_idx_to_edge_in_additions = nullptr;

  std::vector<std::vector <std::pair <VertexID, int>>> host_src_to_roots;

  VertexID** neighbors = new VertexID* [N_HOPS];
  int* neighbors_sizes = new int [N_HOPS];
  int** additions_sizes = new int* [N_HOPS];
  int*** final_map_vertex_to_additions = new int**[N_HOPS];
  //Map of idx of embedding to the start of how many inputs are added and number of new embeddings
  int*** device_map_vertex_to_additions = new int**[N_HOPS]; new int*[csr_partitions.size ()]; 

  for (int hop = 0; hop < N_HOPS; hop++) {
    int* source_vertex_idx;
    int* device_src_to_roots;
    int* device_src_to_root_positions;
    unsigned long long* device_max_neighbors_iter;
    unsigned long long int* device_profile_branch_1;
    unsigned long long int* device_profile_branch_2;
    size_t num_neighbors = 0;
    final_map_vertex_to_additions[hop] = new int*[csr_partitions.size ()];
    //size_t map_vertex_to_additions_size;
    const size_t map_vertex_to_additions_size = csr->get_n_vertices () * sizeof (VertexID) * 2;
    final_map_vertex_to_additions[hop][0] = (int*)new char[map_vertex_to_additions_size];
    size_t final_map_vertex_to_additions_iter = 0;
    device_map_vertex_to_additions[hop] = new int*[csr_partitions.size ()];
    
    for (int p = 0; p < csr_partitions.size (); p++) {
      device_map_vertex_to_additions[hop][p] = nullptr;
    }
    /********************Get the output additions lengths*******************/
    for (int partition_idx = 0; partition_idx < csr_partitions.size (); partition_idx++) {
      unsigned long long num_neighbors_iter = 0;
      CSRPartition& partition = csr_partitions[partition_idx];

      copy_partition_to_gpu (partition, device_csr, device_vertex_array, device_edge_array);

      num_neighbors_iter = 0;
      const int partition_map_vertex_to_additions_size = partition.get_n_vertices ()* 2;
      
      EXECUTE_CUDA_FUNC (cudaMalloc (&device_max_neighbors_iter, 
                                     sizeof(unsigned long long)));
      EXECUTE_CUDA_FUNC (cudaMemset (device_max_neighbors_iter, 0, 
                                     sizeof (unsigned long long)));
      EXECUTE_CUDA_FUNC (cudaMalloc (&device_map_vertex_to_additions[hop][partition_idx], 
                                     partition_map_vertex_to_additions_size*sizeof (VertexID)));

      std::cout << "Calling cuda kernel for hop: " << hop << " partition: " << partition_idx << " vertex = [" << csr_partitions[partition_idx].first_vertex_id << ", "<< csr_partitions[partition_idx].last_vertex_id << "]" << std::endl;

      int N_THREADS = 128;
      int N_BLOCKS = (partition.get_n_vertices ()%128 == 0) ? partition.get_n_vertices ()/128 : partition.get_n_vertices ()/128 + 1;
      double t1 = convertTimeValToDouble(getTimeOfDay ());

      int* device_edges_to_prev_iter_additions;
      const int vertex_with_next_partition = get_common_vertex_with_next_partition (csr_partitions, partition_idx);
      const int vertex_with_prev_partition = get_common_vertex_with_previous_partition (csr_partitions, partition_idx);
                                                                            
      if (hop > 0) {
        int* edges_to_prev_iter_additions;
        edges_to_prev_iter_additions = new int[partition.get_n_edges ()];
        for (int e = partition.first_edge_idx; e <= partition.last_edge_idx; e++) {
          VertexID v = partition.get_edge (e);
          if (!partition.is_vertex_in_partition(v) || 
              v == vertex_with_next_partition || v == vertex_with_prev_partition) {
            edges_to_prev_iter_additions[e - partition.first_edge_idx] = final_map_vertex_to_additions[hop-1][0][2*v + 1];
          }
        }

        EXECUTE_CUDA_FUNC (cudaMalloc (&device_edges_to_prev_iter_additions, 
                                       partition.get_n_edges ()*sizeof(VertexID)));
        EXECUTE_CUDA_FUNC (cudaMemcpy (device_edges_to_prev_iter_additions, 
                                       edges_to_prev_iter_additions, 
                                       partition.get_n_edges ()*sizeof(VertexID), 
                                       cudaMemcpyHostToDevice));
      }

      if (hop == 0) {
        get_max_lengths_for_vertices_first_iter <<<N_BLOCKS, N_THREADS>>> (device_csr, partition.first_vertex_id, 
                                                                           partition.last_vertex_id,
                                                                           device_max_neighbors_iter,
                                                                           device_map_vertex_to_additions[hop][partition_idx]);
      } else {
        get_max_lengths_for_vertices_single_step <<<N_BLOCKS, N_THREADS>>> (device_csr, partition.first_vertex_id, 
                                                                            partition.last_vertex_id,
                                                                            device_max_neighbors_iter,
                                                                            device_map_vertex_to_additions[hop-1][partition_idx],
                                                                            device_map_vertex_to_additions[hop][partition_idx],
                                                                            device_map_vertex_to_additions[0][partition_idx],
                                                                            device_edges_to_prev_iter_additions,
                                                                            vertex_with_prev_partition,
                                                                            vertex_with_next_partition);
      }
  
      EXECUTE_CUDA_FUNC (cudaDeviceSynchronize ());
      double t2 = convertTimeValToDouble(getTimeOfDay ());
  
      gpu_time += t2 - t1;
  
      std::cout << "Cuda Kernel Done " << std::endl;
      is_cuda_error (cudaGetLastError ());
      EXECUTE_CUDA_FUNC (cudaMemcpy (&num_neighbors_iter, device_max_neighbors_iter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      std::cout << "New Neighbors " << num_neighbors_iter << std::endl;
      
      int *partition_map_vertex_to_additions = new int[partition_map_vertex_to_additions_size];
      EXECUTE_CUDA_FUNC (cudaMemcpy (partition_map_vertex_to_additions, 
                                     device_map_vertex_to_additions[hop][partition_idx], 
                                     partition_map_vertex_to_additions_size*sizeof (int), 
                                     cudaMemcpyDeviceToHost));
      
      if (partition_idx == 0) {
        memcpy (&final_map_vertex_to_additions[hop][0][final_map_vertex_to_additions_iter],
                partition_map_vertex_to_additions,
                partition_map_vertex_to_additions_size*sizeof (int));
        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size;
      } else if (vertex_with_prev_partition != -1) {
        int common_vertex = vertex_with_prev_partition;
        int common_vertex_new_additions = partition_map_vertex_to_additions[2*(common_vertex - common_vertex) + 1];
        int common_vertex_start_pos = final_map_vertex_to_additions[hop][0][2*common_vertex];

        for (int v = csr_partitions[partition_idx - 1].first_vertex_id; 
             v < csr_partitions[partition_idx - 1].last_vertex_id; v++) {
          if (final_map_vertex_to_additions[hop][0][2*v] > common_vertex_start_pos) {
            final_map_vertex_to_additions[hop][0][2*v] += common_vertex_new_additions;
          }
        }
        final_map_vertex_to_additions[hop][0][2*common_vertex + 1] += common_vertex_new_additions;
        int start_pos = 0;
        //TODO: start_pos is sum of all embedding additions so far
        int max_v = 0;
        for (int v = csr_partitions[partition_idx - 1].first_vertex_id; 
             v <= csr_partitions[partition_idx - 1].last_vertex_id; v++) {
          int p = final_map_vertex_to_additions[hop][0][2*v] + final_map_vertex_to_additions[hop][0][2*v + 1];
          if (p > start_pos) {
            start_pos = p;
            max_v = v;
          }
        }
        
        assert (start_pos <= (num_neighbors + common_vertex_new_additions));
        for (int v = 1; v < csr_partitions[partition_idx].get_n_vertices (); v++) {
          int vertex = csr_partitions[partition_idx].first_vertex_id + v;
          assert (partition_map_vertex_to_additions[2*v] >= 0);
          int vertex_start_pos = partition_map_vertex_to_additions[2*v];
          if (vertex_start_pos > partition_map_vertex_to_additions[0]) {
            vertex_start_pos -= common_vertex_new_additions;
          }

          final_map_vertex_to_additions[hop][0][2*vertex] = start_pos + vertex_start_pos;
          final_map_vertex_to_additions[hop][0][2*vertex + 1] = partition_map_vertex_to_additions[2*v + 1];
        }
        
        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size - 2;
      } else {
        int start_pos = 0;
        for (int v = csr_partitions[partition_idx - 1].first_vertex_id; 
             v <= csr_partitions[partition_idx - 1].last_vertex_id; v++) {
          int pos = final_map_vertex_to_additions[hop][0][2*v] + final_map_vertex_to_additions[hop][0][2*v + 1];
          start_pos = max (start_pos, pos);
        }
        assert (start_pos <= num_neighbors);
        for (int v = 0; v < csr_partitions[partition_idx].get_n_vertices (); v++) {
          int vertex = csr_partitions[partition_idx].first_vertex_id + v;
          final_map_vertex_to_additions[hop][0][2*vertex] = start_pos + partition_map_vertex_to_additions[2*v];
          final_map_vertex_to_additions[hop][0][2*vertex + 1] = partition_map_vertex_to_additions[2*v + 1];
        }

        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size;
      }

      num_neighbors += num_neighbors_iter;
    }

    num_neighbors = num_neighbors * sizeof (VertexID);
    EXECUTE_CUDA_FUNC (cudaFree (device_max_neighbors_iter));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_additions, num_neighbors));
    EXECUTE_CUDA_FUNC (cudaMemset (device_additions, -1, num_neighbors));
    /**************************DONE**********************/

    EXECUTE_CUDA_FUNC (cudaMalloc (&source_vertex_idx, sizeof(int)));
    EXECUTE_CUDA_FUNC (cudaMemset (source_vertex_idx, 0,  sizeof (int)));
    
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_profile_branch_1, sizeof (unsigned long)));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_profile_branch_2, sizeof (unsigned long)));
    EXECUTE_CUDA_FUNC (cudaMemset (device_profile_branch_1, 0, sizeof (unsigned long)));
    EXECUTE_CUDA_FUNC (cudaMemset (device_profile_branch_2, 0, sizeof (unsigned long)));

    int N_BLOCKS = csr->get_n_vertices ();
    
    neighbors[hop] = (VertexID*) new char[num_neighbors];
    neighbors_sizes[hop] = num_neighbors;
    
    if (hop > 0) {
      compute_source_to_root_data (host_src_to_roots, csr, hop, 
                                   final_map_vertex_to_additions, 
                                   additions_sizes, neighbors, 
                                   neighbors_sizes, device_src_to_roots, 
                                   device_src_to_root_positions);
    }

    CSR* device_csr1;
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_csr1, sizeof (CSR)));
    EXECUTE_CUDA_FUNC (cudaMemcpy (device_csr1, csr, sizeof (CSR), cudaMemcpyHostToDevice));

    int* device_final_map_vertex_to_additions;

    EXECUTE_CUDA_FUNC (cudaMemset (device_additions_sizes, 0, sizeof(VertexID)*csr->get_n_vertices ()*2));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_final_map_vertex_to_additions, map_vertex_to_additions_size));
    EXECUTE_CUDA_FUNC (cudaMemcpy (device_final_map_vertex_to_additions, 
                                   &final_map_vertex_to_additions[hop][0][0],
                                   map_vertex_to_additions_size,
                                   cudaMemcpyHostToDevice));

    double t1 = convertTimeValToDouble(getTimeOfDay ());
    run_hop_parallel_single_step <<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_csr1,  
                                                            device_additions,
                                                            num_neighbors,
                                                            device_additions_prev_hop,
                                                            device_final_map_vertex_to_additions,
                                                            device_additions_sizes,
                                                            device_src_to_roots,
                                                            device_src_to_root_positions,
                                                            source_vertex_idx,
                                                            device_profile_branch_1,
                                                            device_profile_branch_2);
    EXECUTE_CUDA_FUNC (cudaDeviceSynchronize ());
    double t2 = convertTimeValToDouble(getTimeOfDay ());
    gpu_time += t2 - t1;
    additions_sizes[hop] = new int[csr->get_n_vertices () * 2];
    EXECUTE_CUDA_FUNC (cudaMemcpy (neighbors[hop], device_additions, neighbors_sizes[hop], cudaMemcpyDeviceToHost));
    EXECUTE_CUDA_FUNC (cudaMemcpy (additions_sizes[hop], device_additions_sizes, csr->get_n_vertices ()*sizeof(VertexID)*2, cudaMemcpyDeviceToHost));
    
#ifdef PROFILE
    unsigned long profile_branch_1, profile_branch_2;
    EXECUTE_CUDA_FUNC (cudaMemcpy (&profile_branch_1, device_profile_branch_1, sizeof(profile_branch_1), cudaMemcpyDeviceToHost));
    EXECUTE_CUDA_FUNC (cudaMemcpy (&profile_branch_2, device_profile_branch_2, sizeof(profile_branch_1), cudaMemcpyDeviceToHost));

    std::cout << "profile_branch_1 " << profile_branch_1 << std::endl;
    std::cout << "profile_branch_2 " << profile_branch_2 << std::endl;
#endif
    device_additions_prev_hop = device_additions;
  }
  
  std::cout << "Getting embeddings from GPU" << std::endl;
  std::vector <VectorVertexEmbedding> produced_embeddings;
  for (int input_embedding_idx = 0; input_embedding_idx < csr->get_n_vertices (); input_embedding_idx++) {
    size_t produced_embedding_size = 0;
    for (int hop = 0; hop < N_HOPS; hop++) {
      VectorVertexEmbedding& input_embedding = input_embeddings[input_embedding_idx];
      int n_additions = additions_sizes[hop][2*input_embedding_idx + 1];
      produced_embedding_size += n_additions;
    }
    //std::cout << " input_embedding_idx " << input_embedding_idx << std::endl;
    int copied = 0;
    size_t global_mem_idx = GlobalMemAllocator::alloc_vertices_array (produced_embedding_size);
    for (int hop = 0; hop < N_HOPS; hop++) {
      VectorVertexEmbedding& input_embedding = input_embeddings[input_embedding_idx];
      int start_idx = final_map_vertex_to_additions[hop][0][2*input_embedding_idx];
      int n_additions = additions_sizes[hop][2*input_embedding_idx + 1];
      //std::cout << "i " << input_embedding_idx << " produced_embedding_size " << produced_embedding_size << " global_mem_idx " << global_mem_idx << std::endl;
      VertexID* ptr = (VertexID*) ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_idx);
      memcpy (ptr + copied, &neighbors[hop][start_idx], sizeof(VertexID)*n_additions);

      if (input_embedding_idx == 3030) {
        for (int ii = start_idx; ii < n_additions + start_idx; ii++) {
          std::cout << neighbors[hop][ii] << std::endl;
        }
      }
      copied += n_additions;
    }

    VectorVertexEmbedding embedding = VectorVertexEmbedding ((uint32_t)produced_embedding_size, global_mem_idx, true);
    produced_embeddings.push_back (embedding);
  }

  cudaFree (device_csr);

  std::cout << "Generating CPU Embeddings:" << std::endl;
  double cpu_t1 = convertTimeValToDouble (getTimeOfDay ());
  std::vector<std::vector<VertexID>> hops = n_hop_cpu (csr, N_HOPS);
  double cpu_t2 = convertTimeValToDouble (getTimeOfDay ());

  std::cout << "CPU Time: " << (cpu_t2 - cpu_t1) << " secs" << std::endl;
  std::cout << "GPU Time: " << gpu_time << " secs" << std::endl;
  assert (produced_embeddings.size () == hops.size ());
  for (int idx = 0; idx < produced_embeddings.size (); idx++) {
    if (idx == 3030) {
      std::cout << "vertices " << hops[idx].size () << std::endl;
    }
    std::unordered_set<VertexID> cpu_set = std::unordered_set<VertexID> (hops[idx].begin (), hops[idx].end ());
    std::vector<VertexID> vector_hops;
    vector_hops.insert (vector_hops.begin (), cpu_set.begin(), cpu_set.end ());
    std::sort (vector_hops.begin (), vector_hops.end ());
    std::vector<VertexID> gpu_vector = produced_embeddings [idx].to_vector ();
    std::unordered_set<VertexID> gpu_vector_set = std::unordered_set<VertexID> (gpu_vector.begin (), gpu_vector.end ());
    gpu_vector = std::vector<VertexID> (gpu_vector_set.begin (), gpu_vector_set.end ());
    std::sort (gpu_vector.begin (), gpu_vector.end ());

    if (vector_hops != gpu_vector) {
      std::cout << "checking for vertex " << idx << " start " << final_map_vertex_to_additions[1][0][2*idx] << " " << additions_sizes[1][2*idx+1] << std::endl;
      std::cout << "size " << vector_hops.size () << " " << gpu_vector.size () << std::endl;
      #if 1
      for (int i = 0; i < max (vector_hops.size (), gpu_vector.size ()); i++) {
        if (i < min (vector_hops.size (), gpu_vector.size ()))
          std::cout << vector_hops[i] << "  " << gpu_vector[i] << std::endl;
        else if (i < vector_hops.size ()) 
          std::cout << vector_hops[i] << std::endl;
        else if (i < gpu_vector.size ()) 
          std::cout << "     " << gpu_vector[i] << std::endl;
      }
      #endif
    }
    assert (vector_hops == gpu_vector);
  }

#ifdef PINNED_MEMORY
  // cudaFree (global_mem_ptr);
#else
  delete[] global_mem_ptr;
#endif
  std::cout << "Number of embeddings found "<< input_embeddings.size () << std::endl;
  std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  std::cout << "Time spent in Streams " << total_stream_time << std::endl;
}