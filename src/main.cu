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
#include <algorithm>
#include <numeric>
#include <string.h>
#include <assert.h>
#include <tuple>
#include <queue>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/cub.cuh>
#include <curand.h>
#include <cuda.h>

//citeseer.graph
// const int N = 3312;
// const int N_EDGES = 9074;
//micro.graph
//const int N = 100000;
//const int N_EDGES = 2160312;
//rmat.graph
// const int N = 1024;
// const int N_EDGES = 29381;
//ego-facebook
// const int N = 4039;
// const int N_EDGES = 88244;
//ego-twitter
//const int N = 81306;
//const int N_EDGES = 2420766;
//ego-gplus
//const int N = 107614;
//const int N_EDGES = 13652253;
//soc-pokec-relationships
//const int N = 1632803;
//const int N_EDGES = 30480021;
//soc-LiveJournal1
const int N = 4847571;
const int N_EDGES = 68556521;

#include "csr.hpp"
#include "utils.hpp"
#include "pinned_memory_alloc.hpp"

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);\
  return EXIT_FAILURE;}} while(0)

using namespace utils;

#define MAX_EDGES (2*MAX_LOAD_PER_TB)
//#define USE_PARTITION_FOR_SHMEM
#define MAX_HOP_VERTICES_IN_SH_MEM (MAX_VERTICES_PER_TB)
//#define ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM
#define RANDOM_WALK

//#define GRAPH_PARTITION_SIZE (50*1024*1024) //24 KB is the size of each partition of graph
//#define REMOVE_DUPLICATES_ON_GPU
//#define CHECK_RESULT

typedef uint8_t SharedMemElem;

const int N_THREADS = 256;

#define MAX_LOAD_PER_TB (N_THREADS)
#define MAX_VERTICES_PER_TB 1
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

enum SourceVertexExec_t
{
  BlockLevel,
  DeviceLevel
};


__device__ int n_edges_to_warp_size (const EdgePos_t n_edges, SourceVertexExec_t src_vertex_exec) 
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
    return warpSize;
  }
#endif
}

__global__ void get_max_lengths_for_vertices_first_iter (CSRPartition* void_csr,
                                                         VertexID start_vertex, 
                                                         VertexID end_vertex,
                                                         unsigned long long int* embeddings_additions_iter,
                                                         EdgePos_t* map_orig_embedding_to_additions)
{
  CSRPartition* csr = (CSRPartition*)void_csr;

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  VertexID vertex = thread_idx + start_vertex;
  if (vertex > end_vertex) {
    return;
  }

  unsigned long long int new_edges = 0;

  /*Perform a single hop for all vertices in the input embedding*/
  const EdgePos_t start_edge_idx = csr->get_start_edge_idx (vertex);
  const EdgePos_t end_edge_idx = csr->get_end_edge_idx (vertex);

#ifdef RANDOM_WALK
    new_edges = 1;
    unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
    map_orig_embedding_to_additions[2*thread_idx] = thread_idx;
    map_orig_embedding_to_additions[2*thread_idx+1] = new_edges;
#else
  if (end_edge_idx != -1) {
    EdgePos_t e = (end_edge_idx - start_edge_idx) + 1;
    if (e < 0) {
      printf ("v %d s %d e %d\n", vertex, start_edge_idx, end_edge_idx);
    }
    assert (e >= 0);
    new_edges += e;  
  }
  unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
  map_orig_embedding_to_additions[2*thread_idx] = additions_start_iter;
  map_orig_embedding_to_additions[2*thread_idx+1] = new_edges;
  #endif
}

__device__ int src_vertex_to_part_vertex_idx (CSRPartition* csr, int src) 
{
  return src - csr->first_vertex_id;
}

typedef uint32_t ShMemEdgePos_t;

__global__ void __launch_bounds__(N_THREADS) run_hop_parallel_single_step_device_level (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,
              VertexID* embeddings_additions_prev_hop,
              EdgePos_t* map_orig_embedding_to_additions,
              EdgePos_t* previous_stage_filled_range,
              VertexID* hop_vertex_to_roots,
              EdgePos_t* map_vertex_to_hop_vertex_data,
              const VertexID source_vertex,
              const VertexID common_vertex_with_previous_partition,
              const VertexID common_vertex_with_previous_partition_additions
#ifndef NDEBUG
              , unsigned long long int* profile_branch_1, unsigned long long int* profile_branch_2
#endif
)
{
#ifdef USE_PARTITION_FOR_SHMEM
  __shared__ VertexID shmem_csr_edges[N_THREADS];
#endif

  int laneid = threadIdx.x%warpSize;
  
  VertexID hop_vertex = source_vertex;
  VertexID part_vertex_id = src_vertex_to_part_vertex_idx (csr, hop_vertex);
  EdgePos_t hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id];
  const EdgePos_t hop_vertex_end_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id + 1];
  int N_WARPS = blockDim.x/warpSize;
  int warpid = threadIdx.x/warpSize;
  const EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
  EdgePos_t ss = csr->get_start_edge_idx (hop_vertex);
  EdgePos_t ee = csr->get_end_edge_idx (hop_vertex);
  const EdgePos_t start_edge_idx = csr->get_start_edge_idx (hop_vertex) + blockIdx.x * blockDim.x;
  const EdgePos_t end_edge_idx = min (csr->get_end_edge_idx (hop_vertex), start_edge_idx +  blockDim.x - 1);
  const EdgePos_t edges_per_tb = end_edge_idx - start_edge_idx + 1;  

  assert (n_edges >= blockDim.x);

#ifdef USE_PARTITION_FOR_SHMEM
  EdgePos_t edge_pos = start_edge_idx + threadIdx.x;
  if (threadIdx.x < edges_per_tb) {
    shmem_csr_edges[threadIdx.x] = csr->get_edge(edge_pos);
  }
#endif

  for (EdgePos_t _root_vertex_idx = 0; _root_vertex_idx < hop_vertex_end_idx; 
       _root_vertex_idx += N_WARPS) {
    EdgePos_t root_vertex_idx = _root_vertex_idx + warpid;
    if (root_vertex_idx >= hop_vertex_end_idx) {
      continue;
    }

    assert (hop_vertex_start_idx + 2*root_vertex_idx < 
            hop_vertex_start_idx + 2*hop_vertex_end_idx);
    const VertexID root_vertex = hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx];
    
    assert (root_vertex >= root_partition->first_vertex_id && root_vertex <= root_partition->last_vertex_id);
    const EdgePos_t hop_idx = hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx + 1];

    uint warp_hop_mask = FULL_MASK;
    const EdgePos_t start = map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id)];
    assert (hop_vertex == embeddings_additions_prev_hop[hop_idx]);

    EdgePos_t* end = &previous_stage_filled_range[2*(root_vertex - root_partition->first_vertex_id) + 1];
    EdgePos_t e = -1;
    int shfl_warp_size = warpSize; //n_edges_to_warp_size(edges_per_tb, SourceVertexExec_t::DeviceLevel);
    if (laneid%shfl_warp_size == 0) {
      e = utils::atomicAdd (end, edges_per_tb);
    }

    EdgePos_t _e;
    
    _e = __shfl_sync (warp_hop_mask, e, 0, shfl_warp_size);  
    assert (_e != -1);

#ifdef USE_PARTITION_FOR_SHMEM
    int iter = 0;
    EdgePos_t _start_edge_idx = 0;
    EdgePos_t _end_edge_idx = edges_per_tb;    
    while (_start_edge_idx + laneid%shfl_warp_size < _end_edge_idx) {
      VertexID edge = shmem_csr_edges[_start_edge_idx + laneid%shfl_warp_size];
      embeddings_additions[start + _e + iter*shfl_warp_size + laneid%shfl_warp_size] = edge;
      _start_edge_idx += shfl_warp_size;
      iter++;
    }
#else
    int iter = 0;
    EdgePos_t _start_edge_idx = start_edge_idx;

    while (_start_edge_idx + laneid%shfl_warp_size <= end_edge_idx) {
      VertexID edge = csr->get_edge (_start_edge_idx + laneid%shfl_warp_size);
      EdgePos_t addr = start + _e + iter*shfl_warp_size + laneid%shfl_warp_size;
      // if (!(addr < num_neighbors)) {
      //   printf ("v %d start %d addr %d num_neighbors %d\n", root_vertex, start, addr, num_neighbors);
      // }
      // assert (addr < num_neighbors);
      // if (!(addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1])) {
      //   printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1], root_vertex);
      // }
      // assert (addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1]);
      // assert (addr >= start);
      // if (embeddings_additions[addr] != -1) {
      //   //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
      //   printf ("not -1 at %ld hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
      // }
      // assert (embeddings_additions[addr] == -1);
      embeddings_additions[addr] = edge;
      _start_edge_idx += shfl_warp_size;
      iter++;
    }
#endif
  }

  __syncthreads ();
  //previous_stage_filled_range[idx] = start;
}

__device__ __host__ inline EdgePos_t vertex_sample_set_start_pos_fixed_size (VertexID vertex) 
{
  return vertex;
}

//#define USE_PARTITION_FOR_SHMEM_1

__global__ void run_hop_parallel_single_step_device_level_fixed_size (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID last_src_vertex,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,
              VertexID* embeddings_additions_prev_hop,
              EdgePos_t* map_orig_embedding_to_additions,
              EdgePos_t* previous_stage_filled_range,
              VertexID* hop_vertex_to_roots,
              EdgePos_t* map_vertex_to_hop_vertex_data,
              VertexID* source_vertex_idx,
              const VertexID common_vertex_with_previous_partition,
              const VertexID common_vertex_with_previous_partition_additions,
              VertexID* thread_to_src,
              VertexID* thread_to_roots,
              EdgePos_t total_roots,
              EdgePos_t start_linear_id,
              EdgePos_t src_num_roots,
              float* rand

#ifndef NDEBUG
              , unsigned long long int* profile_branch_1, unsigned long long int* profile_branch_2
#endif
)
{
#ifdef USE_PARTITION_FOR_SHMEM_1
  __shared__ EdgePos_t src_num_edges[N_THREADS];
#endif


  int laneid = threadIdx.x%warpSize;
  int warpid = threadIdx.x/warpSize;
  int linear_thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  int global_thread_id = start_linear_id + linear_thread_id;
  VertexID hop_vertex;
#ifdef USE_PARTITION_FOR_SHMEM_1
  if (linear_thread_id < total_roots) {
    hop_vertex = thread_to_src[linear_thread_id];
    src_num_edges[threadIdx.x] = csr->get_n_edges_for_vertex(hop_vertex);
  }

  __syncthreads();
#endif

  // if (linear_thread_id >= src_num_roots)
  //   return;
    
  if (global_thread_id >= total_roots) 
    return;

#ifndef USE_PARTITION_FOR_SHMEM_1
  hop_vertex = thread_to_src[global_thread_id];
#endif
  VertexID vertex = thread_to_roots[global_thread_id];
  if (hop_vertex == -1 || vertex == -1)
    return;
  EdgePos_t start = vertex_sample_set_start_pos_fixed_size(vertex);//map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)];  
  //assert (map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)] == vertex);
  EdgePos_t start_edge_idx;
  start_edge_idx = csr->get_start_edge_idx (hop_vertex);

#ifdef USE_PARTITION_FOR_SHMEM_1
  EdgePos_t n_edges = src_num_edges[threadIdx.x];//csr->get_n_edges_for_vertex(hop_vertex);
#else
  EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
#endif

  if (n_edges > 0) {
    previous_stage_filled_range[global_thread_id] = 1;
    EdgePos_t _e = (EdgePos_t)round(0.5 + n_edges * rand[global_thread_id]) - 1;
    
    {
      VertexID edge = csr->get_edge (start_edge_idx + _e);
      EdgePos_t addr = start;

      embeddings_additions[global_thread_id] = edge;
    }
  }
}

__global__ void run_hop_parallel_single_step_block_level_fixed_size (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID last_src_vertex,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,
              VertexID* embeddings_additions_prev_hop,
              EdgePos_t* map_orig_embedding_to_additions,
              EdgePos_t* previous_stage_filled_range,
              VertexID* hop_vertex_to_roots,
              EdgePos_t* map_vertex_to_hop_vertex_data,
              VertexID* source_vertex_idx,
              const VertexID common_vertex_with_previous_partition,
              const VertexID common_vertex_with_previous_partition_additions,
              VertexID* thread_to_src,
              VertexID* thread_to_roots,
              EdgePos_t total_roots,
              float* rand

#ifndef NDEBUG
              , unsigned long long int* profile_branch_1, unsigned long long int* profile_branch_2
#endif
)
{
  int laneid = threadIdx.x%warpSize;
  int warpid = threadIdx.x/warpSize;
  int linear_thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  VertexID hop_vertex;
  if (linear_thread_id >= total_roots) 
    return;

  VertexID vertex = thread_to_roots[linear_thread_id];
  EdgePos_t start = vertex_sample_set_start_pos_fixed_size(vertex);//map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)];  
  //assert (map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)] == vertex);
  EdgePos_t start_edge_idx;
  start_edge_idx = csr->get_start_edge_idx (hop_vertex);
  EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);

  if (n_edges > 0) {
    previous_stage_filled_range[linear_thread_id] = 1;
    EdgePos_t _e = (EdgePos_t)round(0.5 + n_edges * rand[linear_thread_id]) - 1;
    
    {
      VertexID edge = csr->get_edge (start_edge_idx + _e);
      EdgePos_t addr = start;

      embeddings_additions[linear_thread_id] = edge;
    }
  }
}

__global__ void run_hop_parallel_single_step_block_level (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID last_src_vertex,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,
              VertexID* embeddings_additions_prev_hop,
              EdgePos_t* map_orig_embedding_to_additions,
              EdgePos_t* previous_stage_filled_range,
              VertexID* hop_vertex_to_roots,
              EdgePos_t* map_vertex_to_hop_vertex_data,
              VertexID* source_vertex_idx,
              const VertexID common_vertex_with_previous_partition,
              const VertexID common_vertex_with_previous_partition_additions
#ifndef NDEBUG
              , unsigned long long int* profile_branch_1, unsigned long long int* profile_branch_2
#endif
)
{
  __shared__ VertexID vertices[MAX_VERTICES_PER_TB];
  __shared__ VertexID n_vertex_load;
  __shared__ VertexID thread_idx_to_load[2*MAX_LOAD_PER_TB];
  __shared__ VertexID last_hop_vertex_id;
  __shared__ VertexID last_hop_vertex_roots_remaining;
  __shared__ VertexID last_hop_vertex_roots_done;

#ifdef USE_PARTITION_FOR_SHMEM
  __shared__ VertexID shmem_csr_edges[MAX_EDGES];
  __shared__ VertexID hop_vertex_in_shared_mem[MAX_HOP_VERTICES_IN_SH_MEM];
  __shared__ ShMemEdgePos_t hop_vertices_in_shared_mem_start_edge_idx[MAX_HOP_VERTICES_IN_SH_MEM];
  __shared__ ShMemEdgePos_t hop_vertices_in_shared_mem_end_edge_idx[MAX_HOP_VERTICES_IN_SH_MEM];
  __shared__ VertexID hop_vertices_in_shared_mem_size;
  __shared__ ShMemEdgePos_t shmem_csr_edges_size;
#endif 

  int laneid = threadIdx.x%warpSize;
  int warpid = threadIdx.x/warpSize;
  assert (last_src_vertex <= csr->last_vertex_id);
  assert (csr->first_vertex_id <= last_src_vertex);

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
        
        vertices[n_vertex_load] = ::atomicAdd(source_vertex_idx, 1);

        if (vertices[n_vertex_load] > last_src_vertex) {
          break;
        }
        
        assert (vertices[n_vertex_load] >= 0 && vertices[n_vertex_load] <= last_src_vertex);
        EdgePos_t start_edge_idx = csr->get_start_edge_idx (vertices[n_vertex_load]);
        const EdgePos_t end_edge_idx = csr->get_end_edge_idx (vertices[n_vertex_load]);
        const EdgePos_t n_edges = (end_edge_idx != -1) ? (end_edge_idx - start_edge_idx + 1) : 0;
        VertexID root_vertices = map_vertex_to_hop_vertex_data[2*src_vertex_to_part_vertex_idx(csr, vertices[n_vertex_load]) + 1];
#ifdef USE_PARTITION_FOR_SHMEM
        if (hop_vertices_in_shared_mem_size < MAX_HOP_VERTICES_IN_SH_MEM &&
            n_edges != 0 && n_edges + edges_in_shared_mem < MAX_EDGES && 
            root_vertices != 0) {
          VertexID v = vertices[n_vertex_load];
          hop_vertex_in_shared_mem[hop_vertices_in_shared_mem_size] = v;
          edges_in_shared_mem += n_edges;
          hop_vertices_in_shared_mem_size++;
        }
#endif
        int shfl_warp_size = n_edges_to_warp_size(n_edges, SourceVertexExec_t::BlockLevel);

        if (root_vertices != 0 and n_edges != 0) {
          int root_vertex_idx;
          for (root_vertex_idx = 0; root_vertex_idx < root_vertices && warp_assigned < MAX_LOAD_PER_TB; root_vertex_idx++) {
            for (int ii = warp_assigned; ii < min (warp_assigned + shfl_warp_size, MAX_LOAD_PER_TB); ii++) {
              // if (vertices[0] == 3310) {
              //   int hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*src_vertex_to_part_vertex_idx(csr, vertices[n_vertex_load])];
              //   printf ("624: root %d root_idx %d ii %d warp_assigned %d\n", hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx], root_vertex_idx, ii, warp_assigned);
              // }
              thread_idx_to_load[2*ii] = n_vertex_load;
              thread_idx_to_load[2*ii+1] = root_vertex_idx;
            }
            warp_assigned += shfl_warp_size;
            load_assigned_index += 1;
          }

          if (warp_assigned >= MAX_LOAD_PER_TB) {
            // if (vertices[n_vertex_load] == 3310) {
            //   printf ("635: warp_assigned %d\n", warp_assigned);
            // }
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

    int _curr_vertex_id;
    int root_vertex_idx;

    _curr_vertex_id = thread_idx_to_load[2*threadIdx.x];
    root_vertex_idx = thread_idx_to_load[2*threadIdx.x + 1];

#ifdef USE_PARTITION_FOR_SHMEM
    for (int __hop = 0; __hop < hop_vertices_in_shared_mem_size/(blockDim.x/warpSize) + 1; __hop++) {
      int hop = __hop * (blockDim.x/warpSize) + warpid;
      if (hop >= hop_vertices_in_shared_mem_size) {
        continue;
      }

      EdgePos_t start_edge_idx = csr->get_start_edge_idx (hop_vertex_in_shared_mem[hop]);
      const EdgePos_t end_edge_idx = csr->get_end_edge_idx (hop_vertex_in_shared_mem[hop]);
      const EdgePos_t n_edges = (end_edge_idx != -1) ? (end_edge_idx - start_edge_idx + 1) : 0;
      assert (n_edges > 0);
      
      int _shmem_start = -1;
      if (laneid == 0) {
        _shmem_start = atomicAdd (&shmem_csr_edges_size, n_edges);
      }

      int shmem_start = __shfl_sync (FULL_MASK, _shmem_start, 0, warpSize);
      assert (shmem_start != -1);
      for (EdgePos_t e = 0; e < n_edges/warpSize + 1; e++) {
        EdgePos_t edge_idx = e*warpSize + laneid;
        if (edge_idx < n_edges) {
          shmem_csr_edges[shmem_start + edge_idx] = csr->get_edge (start_edge_idx + edge_idx);
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
   
    int hop_vertex_start_idx = -1;
    //int n_root_vertices = -1;
    int root_vertex = -1;
    int hop_idx = -1;
    int first_active_thread = -1;
    int participating_threads = 0;

    if (_curr_vertex_id != -1 && root_vertex_idx != -1 && vertices[_curr_vertex_id] <= last_src_vertex) {
      VertexID part_vertex_id = src_vertex_to_part_vertex_idx (csr, vertices[_curr_vertex_id]);
      hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id];
      EdgePos_t hop_vertex_end_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id + 1];
      assert (hop_vertex_start_idx + 2*root_vertex_idx < hop_vertex_start_idx + 2*hop_vertex_end_idx);
      root_vertex = hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx];
      assert (root_vertex >= root_partition->first_vertex_id && root_vertex <= root_partition->last_vertex_id);
      hop_idx = hop_vertex_to_roots[hop_vertex_start_idx + 2*root_vertex_idx + 1];      
    }

    __syncthreads ();

    const uint warp_hop_mask = get_warp_mask_and_participating_threads (_curr_vertex_id != -1 && root_vertex_idx != -1 && vertices[_curr_vertex_id] <= last_src_vertex, participating_threads, first_active_thread);
    
    if (_curr_vertex_id != -1 && root_vertex_idx != -1 && vertices[_curr_vertex_id] <= last_src_vertex) {
      if (root_vertex != -1) {
        VertexID vertex = root_vertex;
        EdgePos_t start = map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)];
        VertexID hop_vertex = embeddings_additions_prev_hop[hop_idx];
        if (!(hop_vertex == vertices[_curr_vertex_id])) {
          printf ("hop_vertex %d vertices[_curr_vertex_id] %d\n", hop_vertex, vertices[_curr_vertex_id]);
        }
        assert (hop_vertex == vertices[_curr_vertex_id]);
        EdgePos_t start_edge_idx;
        EdgePos_t end_edge_idx;
        start_edge_idx = csr->get_start_edge_idx (hop_vertex);
#ifdef RANDOM_WALK
        end_edge_idx = start_edge_idx;
        EdgePos_t n_edges = 1;
#else
        end_edge_idx = csr->get_end_edge_idx (hop_vertex);
        EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
#endif
        __syncwarp (warp_hop_mask);

        EdgePos_t* end = &previous_stage_filled_range[2*(vertex - root_partition->first_vertex_id) + 1];
        if (end_edge_idx != -1) {
          __syncwarp (warp_hop_mask);
          
          EdgePos_t e = -1;
          int shfl_warp_size = n_edges_to_warp_size(n_edges, SourceVertexExec_t::BlockLevel);
#ifdef RANDOM_WALK
          *end = 1;
          EdgePos_t _e = 0;
#else
          if (laneid%shfl_warp_size == 0) {
            e = utils::atomicAdd (end, n_edges);
          }
          EdgePos_t _e;
          
          _e = __shfl_sync (warp_hop_mask, e, 0, shfl_warp_size);  
#endif
          assert (_e != -1);
#ifdef USE_PARTITION_FOR_SHMEM
          if (_curr_vertex_id >= hop_vertices_in_shared_mem_size) {
            int iter = 0;
            while (start_edge_idx + laneid%shfl_warp_size <= end_edge_idx) {
              VertexID edge = csr->get_edge (start_edge_idx + laneid%shfl_warp_size);
              embeddings_additions[start + _e + iter*shfl_warp_size + laneid%shfl_warp_size] = edge;
              start_edge_idx += shfl_warp_size;
              iter++;
            }
          } else {
            int iter = 0;
            EdgePos_t _start_edge_idx = hop_vertices_in_shared_mem_start_edge_idx[_curr_vertex_id];
            EdgePos_t _end_edge_idx = hop_vertices_in_shared_mem_end_edge_idx[_curr_vertex_id];
            if (!(hop_vertex == hop_vertex_in_shared_mem[_curr_vertex_id]))
              printf ("hop_vertex %d hop_vertex_in_shared_mem[%d] %d sz %d\n", hop_vertex, _curr_vertex_id, hop_vertex_in_shared_mem[_curr_vertex_id], hop_vertices_in_shared_mem_size);
            assert (hop_vertex_in_shared_mem[_curr_vertex_id] != -1);
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
            VertexID edge = csr->get_edge (start_edge_idx + laneid%shfl_warp_size);
            EdgePos_t addr = start + _e + iter*shfl_warp_size + laneid%shfl_warp_size;
            // if (!(addr < num_neighbors)) {
            //   printf ("v %d start %d addr %d num_neighbors %d\n", vertex, start, addr, num_neighbors);
            // }
            // assert (addr < num_neighbors);
            // if (!(addr < start + map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id) + 1])) {
            //   printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id) + 1], vertex);
            // }
            // assert (addr < start + map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id) + 1]);
            // assert (addr >= start);
            // if (embeddings_additions[addr] != -1) {
            //   //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
            //   printf ("not -1 at %d hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
            // }
            // assert (embeddings_additions[addr] == -1);
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
      VertexID part_vertex_id = src_vertex_to_part_vertex_idx(csr, vertices[last_hop_vertex_id]);
      VertexID hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id];
      //VertexID n_root_vertices = map_vertex_to_hop_vertex_data[2*part_vertex_id + 1];
      
      __syncthreads ();
      
      VertexID _hop_vertex = vertices[last_hop_vertex_id];
      const EdgePos_t n_edges = csr->get_n_edges_for_vertex(_hop_vertex);
      int shfl_warp_size = n_edges_to_warp_size(n_edges, SourceVertexExec_t::BlockLevel);
      const int N_WARPS = blockDim.x/shfl_warp_size;
      const int warpid = threadIdx.x/shfl_warp_size;

      const EdgePos_t start_edge_idx = csr->get_start_edge_idx (_hop_vertex);
#ifdef RANDOM_WALK
      const EdgePos_t end_edge_idx = start_edge_idx;
#else
      const EdgePos_t end_edge_idx = csr->get_end_edge_idx (_hop_vertex);
#endif

#ifdef USE_PARTITION_FOR_SHMEM
      assert (n_edges <= sizeof(shmem_csr_edges)/sizeof(shmem_csr_edges[0]));
      for (EdgePos_t i = 0; i < n_edges/blockDim.x + 1; i++) {
        EdgePos_t pos = i*blockDim.x + threadIdx.x;
        EdgePos_t edge_pos = start_edge_idx + pos;
        if (pos < end_edge_idx) {
          shmem_csr_edges[pos] = csr->get_edge(edge_pos);
        }
      }

      __syncthreads();
#endif
      for (int i = 0; i < last_hop_vertex_roots_remaining/N_WARPS + 1; i++) {
        VertexID root_idx = i*N_WARPS + warpid;
        if (root_idx >= last_hop_vertex_roots_remaining) {
          continue;
        }
        
        VertexID root_vertex = hop_vertex_to_roots[hop_vertex_start_idx + 2*(root_idx + last_hop_vertex_roots_done)];
        VertexID hop_idx = hop_vertex_to_roots[hop_vertex_start_idx + 2*(root_idx + last_hop_vertex_roots_done) + 1];
        EdgePos_t start = map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id)];
        EdgePos_t* end = &previous_stage_filled_range[2*(root_vertex - root_partition->first_vertex_id) + 1];
        VertexID hop_vertex = embeddings_additions_prev_hop[hop_idx];
        assert (_hop_vertex == hop_vertex);

        if (end_edge_idx != -1) {
          EdgePos_t e = -1;

          if (laneid%shfl_warp_size == 0) {
            e = utils::atomicAdd (end, n_edges);
            // if (root_vertex == 37692) {
            //   printf ("%d: e %ld end %ld threadIdx.x %d n_edges %ld root_idx %d\n", __LINE__, e, *end, threadIdx.x, n_edges, root_idx);
            // }
          }
          
          EdgePos_t _e;
          
          _e = __shfl_sync (FULL_MASK, e, 0, shfl_warp_size);  
          assert (_e != -1);

#ifdef USE_PARTITION_FOR_SHMEM
          EdgePos_t _start_edge_idx = 0;
          int iter = 0;
          while (_start_edge_idx + laneid%shfl_warp_size < n_edges) {
            VertexID edge = shmem_csr_edges[_start_edge_idx + laneid%shfl_warp_size];
            EdgePos_t addr = start + _e + iter*shfl_warp_size + laneid%shfl_warp_size;
            // if (!(addr < num_neighbors)) {
            //   printf ("v %d start %d addr %d num_neighbors %d\n", root_vertex, start, addr, num_neighbors);
            // }
            // assert (addr < num_neighbors);
            // if (!(addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1])) {
            //   printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1], root_vertex);
            // }
            // assert (addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1]);
            // assert (addr >= start);
            // if (embeddings_additions[addr] != -1) {
            //   //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
            //   printf ("not -1 at %d hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
            // }
            // assert (embeddings_additions[addr] == -1);
            embeddings_additions[addr] = edge;
            _start_edge_idx += shfl_warp_size;
            iter++;
          }
#else
          EdgePos_t _start_edge_idx = start_edge_idx;
          int iter = 0;
          while (_start_edge_idx + laneid%shfl_warp_size <= end_edge_idx) {
            VertexID edge = csr->get_edge(_start_edge_idx + laneid%shfl_warp_size);
            EdgePos_t addr = start + _e + iter*shfl_warp_size + laneid%shfl_warp_size;
            // if (!(addr < num_neighbors)) {
            //   printf ("v %d start %d addr %d num_neighbors %d\n", root_vertex, start, addr, num_neighbors);
            // }
            // assert (addr < num_neighbors);
            // if (!(addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1])) {
            //   printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1], root_vertex);
            // }
            // assert (addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1]);
            // assert (addr >= start);
            // if (embeddings_additions[addr] != -1) {
            //   //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
            //   printf ("not -1 at %d hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
            // }
            // assert (embeddings_additions[addr] == -1);
            embeddings_additions[addr] = edge;
            _start_edge_idx += shfl_warp_size;
            iter++;
          }
#endif
        }
      }
      __syncthreads ();
    }
  } else {
    VertexID source_vertex = blockIdx.x + csr->first_vertex_id;
    assert (source_vertex <= csr->last_vertex_id);
    assert (csr->first_vertex_id >= 0);
    assert (csr->first_vertex_id <= csr->last_vertex_id);

    VertexID idx = 2*(source_vertex - csr->first_vertex_id);
    EdgePos_t start = map_orig_embedding_to_additions[idx];
    // if (source_vertex == common_vertex_with_previous_partition) {
    //   start += common_vertex_with_previous_partition_additions;
    // }
    EdgePos_t start_edge_idx = csr->get_start_edge_idx (source_vertex);
    const EdgePos_t end_edge_idx = csr->get_end_edge_idx (source_vertex);
    
    if (end_edge_idx != -1) {
      assert (idx < csr->get_n_vertices () * 2);
      assert (idx >= 0);
#ifdef RANDOM_WALK
      const EdgePos_t n_edges = 1;
#else
      const EdgePos_t n_edges = end_edge_idx - start_edge_idx + 1;
#endif
      EdgePos_t* end = &previous_stage_filled_range[idx + 1];

      for (EdgePos_t i = 0; i < n_edges/blockDim.x + 1; i++) {
        EdgePos_t edge_idx = i*blockDim.x + threadIdx.x;
        if (edge_idx < n_edges) {
          // if (start_edge_idx + edge_idx < csr->first_edge_idx) {
          //   printf ("previous_stage_filled_range %p end %p idx %d\n", previous_stage_filled_range, end, idx);
          // }
          VertexID edge = csr->get_edge (start_edge_idx + edge_idx);
#ifdef RANDOM_WALK
          EdgePos_t e = *end;
          *end = 1;
#else
          EdgePos_t e = utils::atomicAdd (end, 1);
#endif
          assert (start + e < num_neighbors);
          assert (start + e >= 0);
          //printf ("edge %d for %d\n", edge, source_vertex);
          embeddings_additions[start + e] = edge;    
        }
      }
    }
  
    __syncthreads ();
    previous_stage_filled_range[idx] = start;
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void remove_duplicates_in_hop_per_block (int N_HOPS, int hop, 
                                          CSRPartition* root_partition,
                                          VertexID* embeddings_additions,
                                          EdgePos_t* previous_stage_filled_range,
                                          EdgePos_t* map_orig_embedding_to_additions)
{

  VertexID root_vertex = blockIdx.x; //+ root_partition->first_vertex_id;
  if (root_vertex >= root_partition->get_n_vertices ()) {
    return;
  }

  // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
  typedef cub::BlockScan<VertexID, BLOCK_THREADS> BlockScanT;
  typedef cub::BlockLoad<
      VertexID, BLOCK_THREADS, ITEMS_PER_THREAD> BlockLoadT;
  typedef cub::BlockStore<
      VertexID, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStoreT;
  typedef cub::BlockRadixSort<
      VertexID, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union {
      typename BlockLoadT::TempStorage       load; 
      typename BlockStoreT::TempStorage      store; 
      typename BlockRadixSortT::TempStorage  sort;
      typename BlockScanT::TempStorage scan;
  } temp_storage; 

  __shared__ VertexID thread_boundary_items[BLOCK_THREADS*ITEMS_PER_THREAD];
  __shared__ VertexID is_equal[BLOCK_THREADS*ITEMS_PER_THREAD];

  int start = map_orig_embedding_to_additions[2*root_vertex];
  int end = previous_stage_filled_range[2*root_vertex + 1];
  if (end > 1024)
    return;
  if (end <= 1)
    return;

  VertexID thread_items[ITEMS_PER_THREAD];
  
  const int per_iter_items = blockDim.x * ITEMS_PER_THREAD;
  assert (end <= per_iter_items);

  /*Sort Neighbors*/
  BlockLoadT(temp_storage.load).Load(&embeddings_additions[start], thread_items, end, N+1);
  
  __syncthreads ();
  
  BlockRadixSortT(temp_storage.sort).Sort(thread_items);

  __syncthreads ();
  
  /*Load sorted neighbors in shared memory*/
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_boundary_items[threadIdx.x*ITEMS_PER_THREAD + i] = thread_items[i];
  }

  /*Set each element of is_equal as 0 or 1 based on weather
    two consecutive elements of thread_boundary_items are equal*/
  __syncthreads ();
  is_equal[0] = 1;
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int elem_idx = threadIdx.x*ITEMS_PER_THREAD + i;
    if (elem_idx > 0 and elem_idx < BLOCK_THREADS*ITEMS_PER_THREAD)
      is_equal[elem_idx] = (thread_boundary_items[elem_idx] == thread_boundary_items[elem_idx-1]) ? 0 : 1;
  }

  __syncthreads ();

  /*Do Prefix sum*/
  BlockLoadT(temp_storage.load).Load(is_equal, thread_items, end, N+1);
  __syncthreads ();

  BlockScanT(temp_storage.scan).ExclusiveSum(thread_items, thread_items);

  __syncthreads ();

  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    is_equal[threadIdx.x*ITEMS_PER_THREAD + i] = thread_items[i];
  }
  
  __syncthreads ();

  /*Store elements back*/
  if (threadIdx.x == 0) {
    for (int i = 0; i < end; i++) {
      int is_equal_idx = i;
      if (is_equal_idx < end) {
        int idx = is_equal[is_equal_idx];
        assert (idx < end);
        embeddings_additions[start + idx] = thread_boundary_items[is_equal_idx];
      }
    }
  }

  /*Update the last */
  if (threadIdx.x == 0) {
    if (thread_boundary_items[end-1] == thread_boundary_items[end-2]) {
      end = is_equal[end -1];
    } else {
      end = is_equal[end -1] + 1;
    }
    previous_stage_filled_range[2*root_vertex + 1] = end;
  }
}

__global__ void update_filled_ranges (VertexID n_vertices, EdgePos_t* previous_stage_filled_range)
{
  VertexID thread_idx = threadIdx.x + blockDim.x*blockIdx.x;

  if (thread_idx >= n_vertices) 
    return;
  
  previous_stage_filled_range[2*thread_idx] = previous_stage_filled_range[2*thread_idx] + previous_stage_filled_range[2*thread_idx+1];
}

std::vector <std::unordered_set <VertexID>> n_hop_cpu_distinct (CSR* csr, const int N_HOPS)
{
  std::vector <std::unordered_set <VertexID>> hops = std::vector<std::unordered_set<VertexID>> (csr->get_n_vertices ());

  for (VertexID vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    EdgePos_t start_edge_idx = csr->get_start_edge_idx (vertex);
    EdgePos_t end_edge_idx = csr->get_end_edge_idx (vertex);
    if (start_edge_idx != -1) {
      for (EdgePos_t edge = start_edge_idx; edge <= end_edge_idx; edge++) {
        hops[vertex].insert (csr->get_edges()[edge]);
      }
    }
  }

  for (VertexID vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    int hop = 1;
    std::unordered_set <VertexID> vertex_hops[N_HOPS + 1];
    vertex_hops[0].insert (hops[vertex].begin (), hops[vertex].end ());
    while (hop < N_HOPS) {
      for (VertexID hop_vertex : vertex_hops[hop - 1]) {
        EdgePos_t start_edge_idx = csr->get_start_edge_idx (hop_vertex);
        EdgePos_t end_edge_idx = csr->get_end_edge_idx (hop_vertex);
        
        if (start_edge_idx != -1) {
          for (EdgePos_t edge = start_edge_idx; edge <= end_edge_idx; edge++) {
            VertexID v = csr->get_edges()[edge];
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
  CHK_CU (cudaMalloc (&device_csr, sizeof(CSRPartition)));
  CHK_CU (cudaMalloc (&device_vertex_array, sizeof(CSR::Vertex)*partition.get_n_vertices ()));
  CHK_CU (cudaMalloc (&device_edge_array, sizeof(CSR::Edge)*partition.get_n_edges ()));
  CHK_CU (cudaMemcpy (device_vertex_array, partition.vertices, sizeof (CSR::Vertex)*partition.get_n_vertices (), cudaMemcpyHostToDevice));
  CHK_CU (cudaMemcpy (device_edge_array, partition.edges, sizeof (CSR::Edge)*partition.get_n_edges (), cudaMemcpyHostToDevice));

  CSRPartition device_csr_partition_value = CSRPartition (partition.first_vertex_id, partition.last_vertex_id, 
                                                          partition.first_edge_idx, partition.last_edge_idx, 
                                                          device_vertex_array, device_edge_array);
  CHK_CU (cudaMemcpy (device_csr, &device_csr_partition_value, sizeof(CSRPartition), cudaMemcpyHostToDevice));
}

VertexID get_common_vertex_with_previous_partition (std::vector<CSRPartition> csr_partitions, int partition_idx)
{
  if (partition_idx <= 0)
    return -1;

  if (csr_partitions[partition_idx].first_vertex_id == csr_partitions[partition_idx - 1].last_vertex_id) {
    return csr_partitions[partition_idx].first_vertex_id;
  }

  return -1;
}


VertexID get_common_vertex_with_next_partition (std::vector<CSRPartition> csr_partitions, int partition_idx)
{
  if (partition_idx >= (int)csr_partitions.size () - 1)
    return -1;

  if (csr_partitions[partition_idx].last_vertex_id == csr_partitions[partition_idx + 1].first_vertex_id) {
    return csr_partitions[partition_idx].last_vertex_id;
  }

  return -1;
}

void create_csr_partitions (CSR* csr, std::vector<CSRPartition>& csr_partitions, const size_t effective_partition_size)
{
  std::vector<std::tuple<VertexID, VertexID, EdgePos_t, EdgePos_t>> vertex_partition_positions_vector;

  //Create Partitions.
  VertexID u = 0;
  EdgePos_t partition_edge_start_idx = 0;

  while (u < csr->get_n_vertices ()) {
    EdgePos_t n_edges = 0;
    VertexID u_start = u;
    EdgePos_t end_edge_idx = 0;
    VertexID u_end = csr->get_n_vertices () - 1;
    EdgePos_t edges = 0;

    for (VertexID v = u; v < csr->get_n_vertices (); v++) {
      EdgePos_t start = csr->get_start_edge_idx (v);
      const EdgePos_t end = csr->get_end_edge_idx (v);
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

        if (end_edge_idx < edges) {
          u = v;
          u_end = v - 1;
          end_edge_idx = start - 1;
        } else if (end_edge_idx == edges) {
          u_end = v;
          u = v + 1;
          end_edge_idx += start - 1; //Including last edge
        } else {
          assert (false);
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
    VertexID u = std::get<0> (p);
    VertexID v = std::get<1> (p);
    EdgePos_t start = std::get<2> (p);
    EdgePos_t end = std::get<3> (p);

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
  EdgePos_t sum_partition_edges = 0;

  for (int id = 0; id < (int)csr_partitions.size (); id++) {
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

  VertexID sum_vertices = 0;
  for (int p = 0; p < (int)csr_partitions.size (); p++) {
    if (p > 0 && csr_partitions[p].first_vertex_id == csr_partitions[p-1].last_vertex_id) {
      sum_vertices += csr_partitions[p].last_vertex_id - (csr_partitions[p].first_vertex_id);
    } else {
      sum_vertices += csr_partitions[p].last_vertex_id - csr_partitions[p].first_vertex_id + 1;
    }
  }

  assert (sum_vertices == N);

  EdgePos_t equal_edges = 0;

  /*Check if union of all partitions is equal to the graph*/
  for (int p = 0; p < (int)csr_partitions.size (); p++) {
    VertexID u = csr_partitions[p].first_vertex_id;
    VertexID v = csr_partitions[p].last_vertex_id;
    EdgePos_t end = csr_partitions[p].last_edge_idx;
    EdgePos_t start = csr_partitions[p].first_edge_idx;
    for (VertexID vertex = u; vertex <= v; vertex++) {
      EdgePos_t _start = csr->get_start_edge_idx (vertex);
      if (p > 0 && vertex == csr_partitions[p-1].last_vertex_id) {
        _start = start;
      }
      EdgePos_t _end = csr->get_end_edge_idx (vertex);
      VertexID part_start = csr_partitions[p].get_start_edge_idx (vertex);
      VertexID part_end = csr_partitions[p].get_end_edge_idx (vertex);
      
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

size_t cpu_get_max_lengths_for_vertices_single_step (int hop, CSRPartition& root_partition, CSR* csr,
                                                     EdgePos_t* map_vertex_to_additions_prev_hop, 
                                                     EdgePos_t* addition_sizes_prev_hop, 
                                                     VertexID* additions_prev_hop, 
                                                     EdgePos_t* map_vertex_to_additions_curr_iter)
{
  size_t embeddings_additions_iter = 0;

  for (VertexID v : root_partition.get_vertex_range()) {
    EdgePos_t new_additions = 0;
#ifdef RANDOM_WALK
    new_additions += 1;
    
    map_vertex_to_additions_curr_iter[2*(v - root_partition.first_vertex_id)] = vertex_sample_set_start_pos_fixed_size(v);
    map_vertex_to_additions_curr_iter[2*(v - root_partition.first_vertex_id) + 1] = 1;
#else
    const EdgePos_t start = map_vertex_to_additions_prev_hop[2*v];
    const EdgePos_t end = addition_sizes_prev_hop[2*v+1];
    for (int idx = start; idx < start + end; idx++) {
      VertexID addition = additions_prev_hop[idx];
      new_additions += csr->n_edges_for_vertex (addition);
    }

    map_vertex_to_additions_curr_iter[2*(v - root_partition.first_vertex_id)] = embeddings_additions_iter;
    map_vertex_to_additions_curr_iter[2*(v - root_partition.first_vertex_id) + 1] = new_additions;
  #endif
    embeddings_additions_iter += new_additions;
  }

  return embeddings_additions_iter;
}

size_t partition_map_vertex_to_additions_size (CSRPartition& partition)
{
  return partition.get_n_vertices ()* 2;
}

int get_partition_idx_of_vertex (std::vector<CSRPartition> csr_partitions, VertexID v) 
{
  std::vector<int> parts;
  for (int part_idx = 0; part_idx < (int)csr_partitions.size (); part_idx++) {
    if (csr_partitions[part_idx].has_vertex (v))
      return part_idx;
  }
  assert (false);
  return -1;
}

//#define SRC_TO_ROOT_VERTEX_IN_SORTED_ORDER

void compute_source_to_root_data (std::vector<std::vector<std::pair <VertexID, int>>>& host_src_to_roots,
                                    const CSR* csr,
                                    const VertexID root_part_idx, const int hop, 
                                    const std::vector<CSRPartition>& csr_partitions,
                                    EdgePos_t*** final_map_vertex_to_additions, 
                                    EdgePos_t** additions_sizes, 
                                    VertexID** neighbors, 
                                    EdgePos_t* neighbors_sizes,
                                    std::unordered_set<int>& src_partitions,
                                    std::vector<VertexID*>& per_part_src_to_roots,
                                    std::vector<EdgePos_t>& per_part_src_to_roots_size, 
                                    std::vector<EdgePos_t*>& per_part_src_to_root_positions)
{
  const CSRPartition& root_partition = csr_partitions[root_part_idx];
  double t1 = convertTimeValToDouble(getTimeOfDay ());
  host_src_to_roots.clear ();

  //Create per hop vertex data
  for (VertexID v = 0; v < csr->get_n_vertices (); v++) {
    host_src_to_roots.push_back (std::vector<std::pair <VertexID, int> > ());
  }
  VertexID common_vertex_with_previous_partition = get_common_vertex_with_previous_partition (csr_partitions, root_part_idx);
  VertexID first_vertex_id = root_partition.first_vertex_id;
  //Do not count a common root vertex with prev partition twice.
  //Always assign the common root vertex to the prev partition.
  if (common_vertex_with_previous_partition != -1) {
    first_vertex_id++;
  }
  
#ifdef SRC_TO_ROOT_VERTEX_IN_SORTED_ORDER
  std::vector<std::pair<EdgePos_t, VertexID>> vertices_sorted_by_start(root_partition.get_n_vertices());
  
  //TODO: Add a vertex and edge iterator in each root partition
  for (VertexID v = first_vertex_id; 
       v <= root_partition.last_vertex_id; v++) {
    EdgePos_t start = final_map_vertex_to_additions[hop-1][0][2*v];
    vertices_sorted_by_start.push_back(std::make_pair(start, v));
  }

  //Sort pair based on the first value.
  std::sort(vertices_sorted_by_start.begin(), vertices_sorted_by_start.end());

  for (auto pair: vertices_sorted_by_start) {
    VertexID v = std::get<1>(pair);
#else
  for (VertexID v = first_vertex_id; v <= root_partition.last_vertex_id; v++) {
#endif

    EdgePos_t start = final_map_vertex_to_additions[hop-1][0][2*v];
    EdgePos_t end   = additions_sizes[hop-1][2*v + 1];

    for (EdgePos_t i = 0; i < end; i++) {
      EdgePos_t src = neighbors[hop-1][start + i];
      int part = get_partition_idx_of_vertex (csr_partitions, src);
      src_partitions.insert (part);
      assert (start + i < neighbors_sizes[hop-1]/sizeof(VertexID));
      assert (src >= 0 && src < N);
      assert (csr_partitions[root_part_idx].get_n_edges_for_vertex(v) > 0);
      host_src_to_roots[src].push_back (std::make_pair (v, start + i));
    }
  }

  per_part_src_to_roots = std::vector<VertexID*> (csr_partitions.size (), nullptr);
  per_part_src_to_root_positions = std::vector<EdgePos_t*> (csr_partitions.size (), nullptr);
  per_part_src_to_roots_size = std::vector<EdgePos_t> (csr_partitions.size(), 0);

  for (VertexID v = 0; v < csr->get_n_vertices (); v++) {
    VertexID part = get_partition_idx_of_vertex (csr_partitions, v);
    per_part_src_to_roots_size[part] += host_src_to_roots[v].size ();
  }

  for (int part = 0; part < (int)csr_partitions.size (); part++) {    
    per_part_src_to_root_positions[part] = new EdgePos_t[2*csr_partitions[part].get_n_vertices ()];
    per_part_src_to_roots[part] = new VertexID[2*per_part_src_to_roots_size[part]];
  }

  for (int part = 0; part < (int)csr_partitions.size (); part++) {
    int iter = 0;

    for (VertexID v = csr_partitions[part].first_vertex_id; 
         v <= csr_partitions[part].last_vertex_id; v++) {
      int part_v = v - csr_partitions[part].first_vertex_id;
      for (VertexID i = 0; i < (VertexID)host_src_to_roots[v].size (); i++) {
        per_part_src_to_roots[part][iter + 2*i] = std::get<0> (host_src_to_roots[v][i]);
        per_part_src_to_roots[part][iter + 2*i + 1] = std::get<1> (host_src_to_roots[v][i]);
      }

      per_part_src_to_root_positions[part][2*part_v] = iter;
      per_part_src_to_root_positions[part][2*part_v + 1] = host_src_to_roots[v].size ();
      iter += 2*host_src_to_roots[v].size ();

      //if (hop == 8 and per_part_src_to_root_positions[part][2*part_v + 1] >= 32)
         //std::cout <<"Src: " << v << " # of roots " << per_part_src_to_root_positions[part][2*part_v + 1] << std::endl;
    }
  }

  double t2 = convertTimeValToDouble(getTimeOfDay ());

  std::cout << "Time taken to create hop vertex data: " << (t2 - t1) << " secs " << std::endl;
}

void remove_duplicates_in_hop_on_cpu(CSRPartition& root_partition, 
                                     EdgePos_t* partition_map_vertex_to_additions, 
                                     EdgePos_t* part_additions_sizes, VertexID* part_additions)
{
  for (VertexID v = root_partition.first_vertex_id; 
       v < root_partition.last_vertex_id; v++) {
    const EdgePos_t start = partition_map_vertex_to_additions[2*v];
    const EdgePos_t end = part_additions_sizes[2*v + 1];

    std::unordered_set<VertexID> distinct;
    for (EdgePos_t idx = start; idx < start + end; idx++) {
      distinct.insert(part_additions[idx]);
    }

    part_additions_sizes[2*v + 1] = distinct.size ();

    EdgePos_t idx = start;
    for (auto elem : distinct) {
      part_additions[idx++] = elem;
    }
  }
}

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

#define PINNED_MEMORY

int main (int argc, char* argv[])
{
  std::vector<Vertex> vertices;

  if (argc < 2) {
    std::cout << "Arguments: graph-file" << std::endl;
    return -1;
  }

  size_t global_mem_size = 1024UL*1024*1024;

#ifdef PINNED_MEMORY
  char* global_mem_ptr;
  //PinnedMemory::allocate();
  void* _ptr = PinnedMemory::pinned_memory_heap.malloc(global_mem_size);
  global_mem_ptr = (char*)_ptr;
#endif


  char* graph_file = argv[1];
  FILE* fp = fopen (graph_file, "r+");
  if (fp == nullptr) {
    std::cout << "File '" << graph_file << "' not found" << std::endl;
    return 1;
  }

  Graph graph (fp);

  fclose (fp);
  //graph.print (std::cout);
  std::cout << "n_edges "<<graph.get_n_edges () <<std::endl;
  std::cout << "vertices " << graph.get_vertices ().size () << std::endl; 

  CSR* csr = new CSR(N, N_EDGES);
  std::cout << "sizeof(CSR)"<< sizeof(CSR)<<std::endl;
  csr_from_graph (csr, graph);
  std::cout << "csr.n_vertices " << csr->get_n_vertices () << " N " << N << std::endl;
  {
    double_t t1 = convertTimeValToDouble(getTimeOfDay ());
    bfs (csr);
    double_t t2 = convertTimeValToDouble(getTimeOfDay ());

    std::cout << "Time spent in BFS " << (t2- t1) << " secs"<< std::endl;
  }

  std::cout << "Pinned Memory Allocated" << std::endl;

  double total_stream_time = 0;

  double_t kernelTotalTime = 0.0;
  std::vector<CSRPartition> csr_partitions;

#ifdef ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM 
  create_csr_partitions (csr, csr_partitions, GRAPH_PARTITION_SIZE - sizeof (CSRPartition));
#else
  CSRPartition full_partition = CSRPartition (0, csr->get_n_vertices () - 1, 0, csr->get_n_edges () - 1, 
                                              csr->get_vertices (), csr->get_edges ());
  csr_partitions.push_back (full_partition);
#endif

  const int N_HOPS = 10;
  
  //Graph on GPU
  CSRPartition* device_csr;
  CSR::Vertex* device_vertex_array;
  CSR::Edge* device_edge_array;

  double gpu_time = 0;
#ifdef RANDOM_WALK
  double rand_walk_time = 0.0;
#endif
  std::cout << "Generating additions" << std::endl;
  EdgePos_t* device_additions_sizes;
  VertexID* device_additions; //Storage to store inputs added to each embedding
  VertexID* device_additions_prev_hop = nullptr;

  std::vector<std::vector <std::pair <VertexID, int>>> host_src_to_roots;

  VertexID** neighbors = new VertexID* [N_HOPS];
  EdgePos_t* neighbors_sizes = new EdgePos_t [N_HOPS];
  EdgePos_t** additions_sizes = new EdgePos_t* [N_HOPS];
  EdgePos_t*** final_map_vertex_to_additions = new EdgePos_t**[N_HOPS];
  //Map of idx of embedding to the start of how many inputs are added and number of new embeddings
  EdgePos_t*** device_map_vertex_to_additions = new EdgePos_t**[N_HOPS];

  VertexID* source_vertex_idx;
  CHK_CU (cudaMalloc (&source_vertex_idx, sizeof(int)));
  unsigned long long* device_max_neighbors_iter;
  CHK_CU (cudaMalloc (&device_max_neighbors_iter, 
                      sizeof(unsigned long long)));

  double end_to_end_t1 = convertTimeValToDouble(getTimeOfDay ());
  for (int hop = 0; hop < N_HOPS; hop++) {
    unsigned long long int* device_profile_branch_1;
    unsigned long long int* device_profile_branch_2;
    EdgePos_t* partition_map_vertex_to_additions[csr_partitions.size ()] = {nullptr};
    std::vector<EdgePos_t> per_part_num_neighbors = std::vector<EdgePos_t> (csr_partitions.size (), 0);
    EdgePos_t num_neighbors = 0;
    final_map_vertex_to_additions[hop] = new EdgePos_t*[csr_partitions.size ()];
    //size_t map_vertex_to_additions_size;
    const EdgePos_t map_vertex_to_additions_size = csr->get_n_vertices () * sizeof (EdgePos_t) * 2;
    final_map_vertex_to_additions[hop][0] = (EdgePos_t*)new char[map_vertex_to_additions_size];
    EdgePos_t final_map_vertex_to_additions_iter = 0;
    device_map_vertex_to_additions[hop] = new EdgePos_t*[csr_partitions.size ()];
    additions_sizes[hop] = new EdgePos_t[csr->get_n_vertices () * 2];
    VertexID** part_neighbors = new VertexID*[csr_partitions.size ()];
    EdgePos_t** part_additions_sizes = new EdgePos_t*[csr_partitions.size ()];

    for (int p = 0; p < (int)csr_partitions.size (); p++) {
      device_map_vertex_to_additions[hop][p] = nullptr;
    }


#ifdef RANDOM_WALK
    EdgePos_t* root_to_linear_thread = new EdgePos_t[csr->get_n_vertices()]; //Make it per part
    for (int i = 0; i < csr->get_n_vertices(); i++) {
      root_to_linear_thread[i] = -1;
    }
#endif
    /********************Get the output additions lengths*******************/
    for (int root_part_idx = 0; root_part_idx < (int)csr_partitions.size (); root_part_idx++) {
      unsigned long long num_neighbors_iter = 0;
      CSRPartition& root_partition = csr_partitions[root_part_idx];
      CSRPartition* device_root_partition;

      copy_partition_to_gpu (root_partition, device_root_partition, device_vertex_array, device_edge_array);

      num_neighbors_iter = 0;
      
      CHK_CU (cudaMemset (device_max_neighbors_iter, 0, 
                          sizeof (unsigned long long)));
      CHK_CU (cudaMalloc (&device_map_vertex_to_additions[hop][root_part_idx], 
                          partition_map_vertex_to_additions_size (root_partition)*sizeof (EdgePos_t)));

      std::cout << "Calling cuda kernel for hop: " << hop << " partition: " << root_part_idx << " vertex = [" << csr_partitions[root_part_idx].first_vertex_id << ", "<< csr_partitions[root_part_idx].last_vertex_id << "]" << std::endl;

      int MAX_LENGTHS_N_THREADS = 128;
      int N_BLOCKS = (root_partition.get_n_vertices ()%MAX_LENGTHS_N_THREADS == 0) ? root_partition.get_n_vertices ()/128 : root_partition.get_n_vertices ()/MAX_LENGTHS_N_THREADS + 1;

      VertexID* device_edges_to_prev_iter_additions;
      EdgePos_t* device_prev_hop_addition_sizes;
      const VertexID vertex_with_prev_partition = get_common_vertex_with_previous_partition (csr_partitions, root_part_idx);
      assert (vertex_with_prev_partition == -1);

      //partition_map_vertex_to_additions[root_part_idx] = new EdgePos_t[partition_map_vertex_to_additions_size (root_partition)];
      partition_map_vertex_to_additions[root_part_idx] = (EdgePos_t*)PinnedMemory::pinned_memory_heap.malloc(partition_map_vertex_to_additions_size (root_partition)*sizeof(EdgePos_t));
      double t1 = convertTimeValToDouble(getTimeOfDay ());

      if (hop == 0) {
        //TODO: No need to invoke on GPU.
        get_max_lengths_for_vertices_first_iter <<<N_BLOCKS, MAX_LENGTHS_N_THREADS>>> (device_root_partition, 
                                                                           root_partition.first_vertex_id, 
                                                                           root_partition.last_vertex_id,
                                                                           device_max_neighbors_iter,
                                                                           device_map_vertex_to_additions[hop][root_part_idx]);
      } else {
        num_neighbors_iter = cpu_get_max_lengths_for_vertices_single_step (hop, root_partition, csr,final_map_vertex_to_additions[hop-1][0], additions_sizes[hop-1], neighbors[hop-1], partition_map_vertex_to_additions[root_part_idx]);        
      }

      CHK_CU (cudaDeviceSynchronize ());
      double t2 = convertTimeValToDouble(getTimeOfDay ());

      if (hop > 0) {
        //CHK_CU (cudaFree (device_edges_to_prev_iter_additions));
        //CHK_CU (cudaFree (device_prev_hop_addition_sizes));
      }

      if (hop > 0) {
        CHK_CU (cudaMemcpy (device_map_vertex_to_additions[hop][root_part_idx],
          partition_map_vertex_to_additions[root_part_idx], 
          partition_map_vertex_to_additions_size (root_partition)*sizeof (EdgePos_t), 
          cudaMemcpyHostToDevice));
      }
  
      //gpu_time += t2 - t1;
  
      std::cout << "Cuda Kernel Done " << std::endl;
      is_cuda_error (cudaGetLastError ());
      if (hop == 0) {
        CHK_CU (cudaMemcpy (&num_neighbors_iter, device_max_neighbors_iter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      } else {
       // num_neighbors_iter = ;
      }
      std::cout << "New " << hop << "-Hop Neighbors " << num_neighbors_iter << std::endl;

      num_neighbors += num_neighbors_iter;
      per_part_num_neighbors [root_part_idx] = num_neighbors_iter;

      CHK_CU (cudaFree ((void*)device_vertex_array));
      CHK_CU (cudaFree ((void*)device_edge_array));
      CHK_CU (cudaFree ((void*)device_root_partition));

      //TODO: Don't use indices of src_partitions, instead use references or pointers
      std::unordered_set<int> src_partitions;
      std::vector<VertexID*> per_part_src_to_roots;
      std::vector<EdgePos_t*> per_part_src_to_roots_positions;
      std::vector<EdgePos_t> per_part_src_to_roots_size;

      CHK_CU (cudaMalloc (&device_root_partition, sizeof (CSRPartition)));
      CHK_CU (cudaMemcpy (device_root_partition, &root_partition, 
                          sizeof (CSRPartition), cudaMemcpyHostToDevice));

      const EdgePos_t partition_additions_sizes_size = partition_map_vertex_to_additions_size (root_partition)*sizeof(EdgePos_t);
#ifdef PROFILE
      CHK_CU (cudaMalloc (&device_profile_branch_1, sizeof (unsigned long)));
      CHK_CU (cudaMalloc (&device_profile_branch_2, sizeof (unsigned long)));
      CHK_CU (cudaMemset (device_profile_branch_1, 0, sizeof (unsigned long)));
      CHK_CU (cudaMemset (device_profile_branch_2, 0, sizeof (unsigned long)));
#endif 
      CHK_CU (cudaMalloc (&device_additions_sizes, 
                          partition_additions_sizes_size));
      CHK_CU (cudaMemset (device_additions_sizes, 0, 
                          partition_additions_sizes_size));
      CHK_CU (cudaMalloc (&device_additions, per_part_num_neighbors[root_part_idx]*sizeof (VertexID)));
#ifndef NDEBUG
      VertexID *temp_array = new VertexID[per_part_num_neighbors[root_part_idx]];
      for (VertexID i = 0; i < per_part_num_neighbors[root_part_idx]; i++)
        temp_array[i] = -1;

      CHK_CU (cudaMemcpy (device_additions, temp_array, per_part_num_neighbors[root_part_idx]*sizeof (VertexID), cudaMemcpyHostToDevice));
      delete temp_array;
#endif

      if (hop > 0) {
        compute_source_to_root_data (host_src_to_roots, csr, root_part_idx, hop,
                                     csr_partitions, final_map_vertex_to_additions,
                                     additions_sizes, neighbors, neighbors_sizes,
                                     src_partitions, per_part_src_to_roots, 
                                     per_part_src_to_roots_size,
                                     per_part_src_to_roots_positions);
      } else {
        src_partitions.insert (root_part_idx);
      }
      
      const EdgePos_t vertex_with_prev_partition_adds = additions_sizes[hop][2*vertex_with_prev_partition + 1];

      for (auto part_idx : src_partitions) {
        VertexID* device_src_to_roots;
        EdgePos_t* device_src_to_root_positions;
        CSRPartition& src_partition = csr_partitions[part_idx];
        std::cout << "Find " << hop << "-hops for root partition " << root_part_idx << " using src partition " << part_idx << std::endl;
        copy_partition_to_gpu (src_partition, device_csr, device_vertex_array, device_edge_array);
        CHK_CU (cudaMemcpy (source_vertex_idx, &src_partition.first_vertex_id,  sizeof (VertexID), cudaMemcpyHostToDevice));
        //TODO: Free source_vertex_idx

        if (hop > 0) {
          CHK_CU (cudaMalloc (&device_src_to_roots, 
                              2*per_part_src_to_roots_size[part_idx]*sizeof (per_part_src_to_roots_size[0])));
          CHK_CU (cudaMemcpy (device_src_to_roots, 
                              per_part_src_to_roots[part_idx], 
                              2*per_part_src_to_roots_size[part_idx]*sizeof (per_part_src_to_roots[0][0]), 
                              cudaMemcpyHostToDevice));
          CHK_CU (cudaMalloc (&device_src_to_root_positions,
                              2*src_partition.get_n_vertices()*sizeof (EdgePos_t)));
          CHK_CU (cudaMemcpy (device_src_to_root_positions,  
                              per_part_src_to_roots_positions[part_idx],
                              2*src_partition.get_n_vertices()*sizeof (EdgePos_t), 
                              cudaMemcpyHostToDevice));
        }

        double t1 = convertTimeValToDouble(getTimeOfDay ());
        std::cout << "hop " << hop << std::endl;

        if (hop >= 1) {
          VertexID vertex_for_block_level_exec = -1;
#if 0
          //for (VertexID src_vertex_id = src_partition.first_vertex_id; 
              //src_vertex_id <= src_partition.last_vertex_id; src_vertex_id++) {
          for (VertexID src_vertex_id : src_partition.get_vertex_range()) {
            if (src_partition.get_n_edges_for_vertex(src_vertex_id) >= N_THREADS) {
              if (vertex_for_block_level_exec == -1) {
                vertex_for_block_level_exec = src_vertex_id - 1;
              }
              int N_BLOCKS = src_partition.get_n_edges_for_vertex(src_vertex_id)/N_THREADS + 1;

              run_hop_parallel_single_step_device_level <<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_csr,  
                                                                    device_root_partition,
                                                                    device_additions,
                                                                    per_part_num_neighbors[root_part_idx],
                                                                    device_additions_prev_hop,
                                                                    device_map_vertex_to_additions[hop][root_part_idx],
                                                                    device_additions_sizes,
                                                                    device_src_to_roots,
                                                                    device_src_to_root_positions,
                                                                    src_vertex_id,
                                                                    vertex_with_prev_partition,
                                                                    vertex_with_prev_partition_adds,
    #ifndef NDEBUG
                                                                    device_profile_branch_1,
                                                                    device_profile_branch_2
    #endif
                                                                  );

              CHK_CU(cudaGetLastError());
            }
          //CHK_CU (cudaDeviceSynchronize ());
          }
#endif
          vertex_for_block_level_exec = src_partition.last_vertex_id;
          std::cout << "vertex_for_block_level_exec " << vertex_for_block_level_exec << std::endl;
          if (vertex_for_block_level_exec != -1) {
            EdgePos_t num_block_level_threads = 0;
            EdgePos_t num_device_level_threads = 0;
            int non_zero_src = 0;
            const EdgePos_t THRESHOLD = 1024;
            for (int src = 0; src < src_partition.get_n_vertices(); src++) {
              EdgePos_t num_roots = per_part_src_to_roots_positions[part_idx][2*src + 1];
              if (num_roots < THRESHOLD)
                num_block_level_threads += num_roots;
              else
                num_device_level_threads += (num_roots % THRESHOLD == 0)?num_roots:(num_roots/THRESHOLD+1)*THRESHOLD;
            }
            std::vector<std::pair<EdgePos_t, VertexID>> src_and_num_roots;
            for (int src = 0; src < src_partition.get_n_vertices(); src++) {
              if (per_part_src_to_roots_positions[part_idx][2*src + 1] != 0) 
                src_and_num_roots.push_back(std::make_pair(per_part_src_to_roots_positions[part_idx][2*src + 1], src));
            }

            std::sort(src_and_num_roots.begin(), src_and_num_roots.end());
            VertexID* block_level_thread_to_src = new VertexID[num_block_level_threads];
            VertexID* block_level_thread_to_roots = new VertexID[num_block_level_threads];
            VertexID* device_level_thread_to_src = new VertexID[num_device_level_threads];
            VertexID* device_level_thread_to_roots = new VertexID[num_device_level_threads];
            bool enable_load_balancing = true;
            int shfl_warp_size = 1;
            int block_level_linear_thread = 0;
            int grid_level_linear_thread = 0;
            VertexID sorted_src_idx_for_grid_level_exec = -1;
            EdgePos_t* src_to_last_linear_thread = new EdgePos_t[src_partition.get_n_vertices()];
            EdgePos_t* src_to_first_linear_thread = new EdgePos_t[src_partition.get_n_vertices()];
            EdgePos_t src_and_num_roots_idx = 0;

            for (src_and_num_roots_idx = 0; src_and_num_roots_idx < src_and_num_roots.size(); src_and_num_roots_idx++) {
              std::pair<EdgePos_t, VertexID> src_num_root = src_and_num_roots[src_and_num_roots_idx];
              VertexID src = std::get<1>(src_num_root);
              EdgePos_t num_roots = std::get<0> (src_num_root);
              
              if (sorted_src_idx_for_grid_level_exec == -1 and num_roots >= THRESHOLD) {
                sorted_src_idx_for_grid_level_exec = src_and_num_roots_idx;
                break;
              }
              
              src_to_first_linear_thread[src + src_partition.first_vertex_id] = block_level_linear_thread;

              for (int _idx = 0; _idx < num_roots; _idx++) {
                EdgePos_t root_idx = per_part_src_to_roots_positions[part_idx][2*src] + 2*_idx;
                block_level_thread_to_src[block_level_linear_thread] = src + src_partition.first_vertex_id;
                VertexID root = per_part_src_to_roots[part_idx][root_idx];
                block_level_thread_to_roots[block_level_linear_thread] = root;
                root_to_linear_thread[root] = block_level_linear_thread;
                
                block_level_linear_thread += 1;
              }

              assert (sorted_src_idx_for_grid_level_exec == -1 || (sorted_src_idx_for_grid_level_exec != -1 and num_roots >= THRESHOLD));
            }
            std::cout << "sorted_src_idx_for_grid_level_exec " << std::get<1>(src_and_num_roots[sorted_src_idx_for_grid_level_exec]) << " " <<  std::get<1>(src_and_num_roots[sorted_src_idx_for_grid_level_exec-1]) << std::endl;
            //For device level mapped, map each source vertex to multiples of thread block
            // if (linear_thread%THRESHOLD != 0)
            //   linear_thread = (linear_thread/THRESHOLD+1)*THRESHOLD;
            for (; src_and_num_roots_idx < src_and_num_roots.size(); src_and_num_roots_idx++) {
              std::pair<EdgePos_t, VertexID> src_num_root = src_and_num_roots[src_and_num_roots_idx];
              VertexID src = std::get<1>(src_num_root);
              EdgePos_t num_roots = std::get<0> (src_num_root);

              if (num_roots >= THRESHOLD and grid_level_linear_thread%THRESHOLD != 0) {
                while (grid_level_linear_thread%THRESHOLD != 0){
                  device_level_thread_to_src[grid_level_linear_thread] = -1;
                  device_level_thread_to_roots[grid_level_linear_thread] = -1;
                  grid_level_linear_thread++;
                }
              }

              src_to_first_linear_thread[src + src_partition.first_vertex_id] = grid_level_linear_thread;

              for (int _idx = 0; _idx < num_roots; _idx++) {
                EdgePos_t root_idx = per_part_src_to_roots_positions[part_idx][2*src] + 2*_idx;
                device_level_thread_to_src[grid_level_linear_thread] = src + src_partition.first_vertex_id;
                VertexID root = per_part_src_to_roots[part_idx][root_idx];
                device_level_thread_to_roots[grid_level_linear_thread] = root;
                root_to_linear_thread[root] = block_level_linear_thread + grid_level_linear_thread;
                
                grid_level_linear_thread += 1;
              }

              src_to_last_linear_thread[src + src_partition.first_vertex_id] = grid_level_linear_thread-1;
            }

            assert (num_device_level_threads == grid_level_linear_thread);

            VertexID* device_block_level_thread_to_src;
            VertexID* device_block_level_thread_to_roots;
            std::cout << "sum_all_roots " << num_block_level_threads +  num_device_level_threads << "grid_level_linear_thread " << grid_level_linear_thread << std::endl;
            CHK_CU(cudaMalloc(&device_block_level_thread_to_src, num_block_level_threads*sizeof(VertexID)));
            CHK_CU(cudaMalloc(&device_block_level_thread_to_roots, num_block_level_threads*sizeof(VertexID)));
            CHK_CU(cudaMemcpy(device_block_level_thread_to_src, block_level_thread_to_src, num_block_level_threads*sizeof(VertexID), cudaMemcpyHostToDevice));
            CHK_CU(cudaMemcpy(device_block_level_thread_to_roots, block_level_thread_to_roots, num_block_level_threads*sizeof(VertexID), cudaMemcpyHostToDevice));
            
            VertexID* device_device_level_thread_to_src;
            VertexID* device_device_level_thread_to_roots;
            CHK_CU(cudaMalloc(&device_device_level_thread_to_src, num_device_level_threads*sizeof(VertexID)));
            CHK_CU(cudaMalloc(&device_device_level_thread_to_roots, num_device_level_threads*sizeof(VertexID)));
            CHK_CU(cudaMemcpy(device_device_level_thread_to_src, device_level_thread_to_src, num_device_level_threads*sizeof(VertexID), cudaMemcpyHostToDevice));
            CHK_CU(cudaMemcpy(device_device_level_thread_to_roots, device_level_thread_to_roots, num_device_level_threads*sizeof(VertexID), cudaMemcpyHostToDevice));

            float* device_block_level_rand;
            cudaMalloc(&device_block_level_rand, num_device_level_threads*sizeof(float));
            float* device_device_level_rand;
            cudaMalloc(&device_device_level_rand, num_device_level_threads*sizeof(float));
            curandGenerator_t gen;
            CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, clock()));
            CURAND_CALL(curandGenerateUniform(gen, device_block_level_rand, num_block_level_threads));
            CURAND_CALL(curandGenerateUniform(gen, device_device_level_rand, num_device_level_threads));

            double t1 = convertTimeValToDouble(getTimeOfDay ());
            
#ifdef RANDOM_WALK
            if (enable_load_balancing) {
              VertexID src_idx = sorted_src_idx_for_grid_level_exec; 
              {
                VertexID _src = std::get<1> (src_and_num_roots[src_idx]);
                EdgePos_t num_roots = num_device_level_threads;
                int num_threads = min((num_roots%32 == 0)?num_roots : (num_roots/32+1)*32, 1024);
                int N_BLOCKS = (num_roots%num_threads == 0) ? num_roots/num_threads:num_roots/num_threads+1;
                run_hop_parallel_single_step_device_level_fixed_size<<<N_BLOCKS,num_threads>>> (N_HOPS, hop, device_csr,
                                                                      device_root_partition,
                                                                      vertex_for_block_level_exec,
                                                                      device_additions,
                                                                      per_part_num_neighbors[root_part_idx],
                                                                      device_additions_prev_hop,
                                                                      device_map_vertex_to_additions[hop][root_part_idx],
                                                                      device_additions_sizes,
                                                                      device_src_to_roots,
                                                                      device_src_to_root_positions,
                                                                      source_vertex_idx,
                                                                      vertex_with_prev_partition,
                                                                      vertex_with_prev_partition_adds,
                                                                      device_device_level_thread_to_src,
                                                                      device_device_level_thread_to_roots,
                                                                      grid_level_linear_thread,
                                                                      block_level_linear_thread,
                  0,
                  num_roots,
                  device_device_level_rand
      #ifndef NDEBUG
                , device_profile_branch_1,
                device_profile_branch_2
      #endif
              );
              }
            }
            
            EdgePos_t block_level_roots = block_level_linear_thread;
            int N_BLOCKS = (block_level_roots%N_THREADS == 0) ? block_level_roots/N_THREADS:block_level_roots/N_THREADS+1;
            run_hop_parallel_single_step_block_level_fixed_size<<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_csr,  
              device_root_partition,
              vertex_for_block_level_exec,
              device_additions,
              per_part_num_neighbors[root_part_idx],
              device_additions_prev_hop,
              device_map_vertex_to_additions[hop][root_part_idx],
              device_additions_sizes,
              device_src_to_roots,
              device_src_to_root_positions,
              source_vertex_idx,
              vertex_with_prev_partition,
              vertex_with_prev_partition_adds,
              device_block_level_thread_to_src,
              device_block_level_thread_to_roots,
              block_level_roots,
              device_block_level_rand
    #ifndef NDEBUG
              , device_profile_branch_1,
              device_profile_branch_2
    #endif
            );
#else
            int N_BLOCKS = vertex_for_block_level_exec - src_partition.first_vertex_id + 1;
            run_hop_parallel_single_step_block_level <<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_csr,  
              device_root_partition,
              vertex_for_block_level_exec,
              device_additions,
              per_part_num_neighbors[root_part_idx],
              device_additions_prev_hop,
              device_map_vertex_to_additions[hop][root_part_idx],
              device_additions_sizes,
              device_src_to_roots,
              device_src_to_root_positions,
              source_vertex_idx,
              vertex_with_prev_partition,
              vertex_with_prev_partition_adds
    #ifndef NDEBUG
              , device_profile_branch_1,
              device_profile_branch_2
    #endif
            );
#endif
            CHK_CU (cudaDeviceSynchronize ());
            double t2 = convertTimeValToDouble(getTimeOfDay ());
#ifdef RANDOM_WALK
            rand_walk_time += (t2-t1);
#endif
            std::cout<<"Block Level time " << (t2-t1) <<" secs" << std::endl;
          }
        }

        if (hop == 0) {
          int N_BLOCKS = src_partition.get_n_vertices ();
          run_hop_parallel_single_step_block_level <<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_csr,  
            device_root_partition,
            src_partition.last_vertex_id,
            device_additions,
            per_part_num_neighbors[root_part_idx],
            device_additions_prev_hop,
            device_map_vertex_to_additions[hop][root_part_idx],
            device_additions_sizes,
            device_src_to_roots,
            device_src_to_root_positions,
            source_vertex_idx,
            vertex_with_prev_partition,
            vertex_with_prev_partition_adds
  #ifndef NDEBUG
            , device_profile_branch_1,
            device_profile_branch_2
  #endif
          );

          CHK_CU(cudaGetLastError());
        }

        CHK_CU (cudaDeviceSynchronize ());
        double t2 = convertTimeValToDouble(getTimeOfDay ());
        gpu_time += t2 - t1;

        if (hop > 0) {
          CHK_CU (cudaFree (device_src_to_roots))
          CHK_CU (cudaFree (device_src_to_root_positions));


          delete per_part_src_to_roots[part_idx];
          delete per_part_src_to_roots_positions[part_idx];
        }

        CHK_CU (cudaFree ((void*)device_vertex_array));
        CHK_CU (cudaFree ((void*)device_edge_array));
        CHK_CU (cudaFree ((void*)device_csr));
      }

  #ifdef PROFILE
      unsigned long profile_branch_1, profile_branch_2;
      CHK_CU (cudaMemcpy (&profile_branch_1, device_profile_branch_1, sizeof(profile_branch_1), cudaMemcpyDeviceToHost));
      CHK_CU (cudaMemcpy (&profile_branch_2, device_profile_branch_2, sizeof(profile_branch_1), cudaMemcpyDeviceToHost));

      std::cout << "profile_branch_1 " << profile_branch_1 << std::endl;
      std::cout << "profile_branch_2 " << profile_branch_2 << std::endl;
  #endif

      part_additions_sizes[root_part_idx] = new EdgePos_t[partition_additions_sizes_size];
      CHK_CU (cudaMemcpy (part_additions_sizes[root_part_idx], device_additions_sizes, 
        partition_additions_sizes_size, 
        cudaMemcpyDeviceToHost));
      
      if (hop == 0) {
        partition_map_vertex_to_additions[root_part_idx] = (EdgePos_t*)PinnedMemory::pinned_memory_heap.malloc(
        sizeof(EdgePos_t)*partition_map_vertex_to_additions_size (root_partition));
        CHK_CU (cudaMemcpy (partition_map_vertex_to_additions[root_part_idx], 
                          device_map_vertex_to_additions[hop][root_part_idx], 
                          partition_map_vertex_to_additions_size (root_partition)*sizeof (EdgePos_t), 
                          cudaMemcpyDeviceToHost));
      }

      //Remove Duplicates
#ifdef REMOVE_DUPLICATES_ON_GPU
      const VertexID block_level_duplicate_find_max_val = 1024;
      EdgePos_t max_end = 0;
      VertexID* d_max_temp_storage = nullptr;
      EdgePos_t* d_selected = nullptr;

      for (VertexID v : root_partition.get_vertex_range()) {
        EdgePos_t end = part_additions_sizes[root_part_idx][2*(v-root_partition.first_vertex_id) + 1];
        max_end = max(end, max_end);
      }
      max_end = max_end*5;
      
      CHK_CU(cudaMalloc(&d_max_temp_storage, max_end));
      CHK_CU(cudaMalloc(&d_selected, sizeof(EdgePos_t)*root_partition.get_n_vertices ()));
      VertexID* device_intermediate_storage = nullptr;
      CHK_CU(cudaMalloc (&device_intermediate_storage, per_part_num_neighbors[root_part_idx]*sizeof (EdgePos_t)));

      std::cout << "Remove Duplicates" << std::endl;
      double duplicate_t1 = convertTimeValToDouble(getTimeOfDay ());

      //Use BlockLevel primitives of CUB to remove duplicates
      remove_duplicates_in_hop_per_block<256, 4> <<<root_partition.get_n_vertices(), 256>>> (
        N_HOPS, hop, device_root_partition,
        device_additions, 
        device_additions_sizes, 
        device_map_vertex_to_additions[hop][root_part_idx]);
      CHK_CU(cudaDeviceSynchronize ());

      //TODO: We can use CUDA streams to speed this up, but it will lead to high
      //memory usage.
      for (VertexID v = root_partition.get_vertex_range()) {
        EdgePos_t start = partition_map_vertex_to_additions[root_part_idx][2*(v-root_partition.first_vertex_id)];
        EdgePos_t end = part_additions_sizes[root_part_idx][2*(v-root_partition.first_vertex_id) + 1];
        if (end < block_level_duplicate_find_max_val) {
          //Used BlockLevel primitives above
        } else {
          VertexID* d_in = (VertexID*)device_additions + start;
          VertexID* d_out = (VertexID*)device_intermediate_storage + start;
          VertexID* d_temp_storage = nullptr;
          size_t temp_storage_bytes = 0;
          
          //Check if the space runs out.
          //TODO: Use DoubleBuffer version that requires O(P) space.
          cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, 
                                         d_in, d_out, end);
          
          if (d_temp_storage != nullptr and temp_storage_bytes <= (size_t)max_end) {
            d_temp_storage = d_max_temp_storage;
          } else {
            // std::cout << "temp_storage_bytes " << temp_storage_bytes << " end " << end << " temp/end " <<  ((float)temp_storage_bytes)/end << std::endl;
            CHK_CU (cudaMalloc(&d_temp_storage, temp_storage_bytes));
          }
          cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, 
                                         d_in, d_out, end);

          //Swap two spaces
          d_in = (VertexID*)device_intermediate_storage + start;
          d_out = (VertexID*)device_additions + start;
          temp_storage_bytes = 0;

          d_temp_storage = nullptr;
          
          cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, 
                                    d_in, d_out, d_selected + (v-root_partition.first_vertex_id), end);
          
          if (temp_storage_bytes <= (EdgePos_t)max_end) {
            d_temp_storage = d_max_temp_storage;
          } else {
            CHK_CU (cudaMalloc(&d_temp_storage, temp_storage_bytes));
          }

          cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, 
                                    d_in, d_out, d_selected + (v-root_partition.first_vertex_id), end);

          CHK_CU(cudaDeviceSynchronize ());
        }
      }
      
      double duplicate_t2 = convertTimeValToDouble(getTimeOfDay ());

      CHK_CU (cudaMemcpy (part_additions_sizes[root_part_idx], device_additions_sizes, 
        partition_additions_sizes_size, 
        cudaMemcpyDeviceToHost));
      
      CHK_CU (cudaFree (device_intermediate_storage));
      for (VertexID v : root_partition.get_vertex_range()) {
        EdgePos_t end = part_additions_sizes[root_part_idx][2*(v-root_partition.first_vertex_id) + 1];
        if (end >= block_level_duplicate_find_max_val)
          CHK_CU(cudaMemcpy (&part_additions_sizes[root_part_idx][2*(v-root_partition.first_vertex_id) + 1], d_selected + (v-root_partition.first_vertex_id), sizeof (EdgePos_t), cudaMemcpyDeviceToHost));
      }

      std::cout << "Time in removing duplicate: " << duplicate_t2 - duplicate_t1 << " secs" << std::endl;
      std::cout << "Duplicates removed" << std::endl;
#endif
      
      part_neighbors[root_part_idx] = new VertexID[per_part_num_neighbors [root_part_idx]];
      CHK_CU (cudaMemcpy (part_neighbors[root_part_idx], device_additions, 
                          per_part_num_neighbors [root_part_idx]*sizeof (VertexID), 
                          cudaMemcpyDeviceToHost));
      
      CHK_CU (cudaFree (device_additions));
      CHK_CU (cudaFree (device_additions_sizes));

      /***DONE***/
      //TODO: Free device_map_vertex_to_additions
      double cpu_part_t1 = convertTimeValToDouble(getTimeOfDay());
      if (root_part_idx == 0) {
        memcpy (&final_map_vertex_to_additions[hop][0][final_map_vertex_to_additions_iter],
                partition_map_vertex_to_additions[root_part_idx],
                partition_map_vertex_to_additions_size(root_partition)*sizeof (EdgePos_t));
        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size(root_partition);
        per_part_num_neighbors [root_part_idx] = num_neighbors_iter;
      } else if (vertex_with_prev_partition != -1) {
        //TODO remove the cases of vertex_with_prev_partition
        VertexID common_vertex = vertex_with_prev_partition;
        EdgePos_t common_vertex_new_additions = partition_map_vertex_to_additions[root_part_idx][2*(common_vertex - common_vertex) + 1];
        EdgePos_t common_vertex_start_pos = final_map_vertex_to_additions[hop][0][2*common_vertex];
        
        for (VertexID v : csr_partitions[root_part_idx - 1].get_vertex_range()) {
          if (final_map_vertex_to_additions[hop][0][2*v] > common_vertex_start_pos) {
            final_map_vertex_to_additions[hop][0][2*v] += common_vertex_new_additions;
          }
        }
        final_map_vertex_to_additions[hop][0][2*common_vertex + 1] += common_vertex_new_additions;
        EdgePos_t start_pos = 0;
        //TODO: start_pos is sum of all embedding additions so far
        for (VertexID v : csr_partitions[root_part_idx - 1].get_vertex_range()) {
          EdgePos_t p = final_map_vertex_to_additions[hop][0][2*v] + final_map_vertex_to_additions[hop][0][2*v + 1];
          if (p > start_pos) {
            start_pos = p;
          }
        }
        
        assert (start_pos <= (num_neighbors + common_vertex_new_additions));
        for (VertexID v = 1; v < csr_partitions[root_part_idx].get_n_vertices (); v++) {
          VertexID vertex = csr_partitions[root_part_idx].first_vertex_id + v;
          assert (partition_map_vertex_to_additions[root_part_idx][2*v] >= 0);
          EdgePos_t vertex_start_pos = partition_map_vertex_to_additions[root_part_idx][2*v];
          if (vertex_start_pos > partition_map_vertex_to_additions[root_part_idx][0]) {
            vertex_start_pos -= common_vertex_new_additions;
          }

          final_map_vertex_to_additions[hop][0][2*vertex] = start_pos + vertex_start_pos;
          final_map_vertex_to_additions[hop][0][2*vertex + 1] = partition_map_vertex_to_additions[root_part_idx][2*v + 1];
        }
        
        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size(root_partition) - 2;

        per_part_num_neighbors [root_part_idx] = num_neighbors_iter;
      } else {
        EdgePos_t start_pos = 0;
        for (VertexID v : csr_partitions[root_part_idx - 1].get_vertex_range()) {
          EdgePos_t pos = final_map_vertex_to_additions[hop][0][2*v] + final_map_vertex_to_additions[hop][0][2*v + 1];
          start_pos = max (start_pos, pos);
        }
        assert (start_pos <= num_neighbors);
        for (VertexID v = 0; v < csr_partitions[root_part_idx].get_n_vertices (); v++) {
          VertexID vertex = csr_partitions[root_part_idx].first_vertex_id + v;
          final_map_vertex_to_additions[hop][0][2*vertex] = start_pos + partition_map_vertex_to_additions[root_part_idx][2*v];
          final_map_vertex_to_additions[hop][0][2*vertex + 1] = partition_map_vertex_to_additions[root_part_idx][2*v + 1];
        }

        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size(root_partition);
      }

      VertexID first_vertex_id;
      if (root_part_idx == 0) {
#ifdef RANDOM_WALK
        for (VertexID v : root_partition.get_vertex_range()) {
          EdgePos_t idx = (hop > 0) ? root_to_linear_thread[v] : 2*v+1;
          additions_sizes[hop][2*v+1] = part_additions_sizes[root_part_idx][idx];
        }
#else
        memcpy (&additions_sizes[hop][0], part_additions_sizes[root_part_idx], 
                partition_additions_sizes_size);
#endif
        first_vertex_id = root_partition.first_vertex_id;
      } else {
        first_vertex_id = root_partition.first_vertex_id;
#ifdef RANDOM_WALK
        for (VertexID v : root_partition.get_vertex_range()) {
          EdgePos_t idx = (hop > 0) ? root_to_linear_thread[v - root_partition.first_vertex_id] : 2*(v - root_partition.first_vertex_id) +1;
          additions_sizes[hop][2*v+1] = part_additions_sizes[root_part_idx][idx];
        }
#else
        memcpy (&additions_sizes[hop][2*root_partition.first_vertex_id],
                part_additions_sizes[root_part_idx], partition_additions_sizes_size);
#endif
      }

      double cpu_part_t2 = convertTimeValToDouble(getTimeOfDay());

      std::cout << "CPU Part " << (cpu_part_t2 - cpu_part_t1) << std::endl;
    }

    num_neighbors = num_neighbors * sizeof (VertexID);
    /**************************DONE**********************/
    double cpu_part_t1 = convertTimeValToDouble(getTimeOfDay());
    neighbors[hop] = (VertexID*) new char[num_neighbors];
    neighbors_sizes[hop] = num_neighbors;

    for (int part = 0; part < (int)csr_partitions.size (); part++) {
      //Copy back the neighbors at the correct place
      //TODO: copy directly to neigbhors array instead of part_neighbors
      for (VertexID v : csr_partitions[part].get_vertex_range()) {
        VertexID part_v = v - csr_partitions[part].first_vertex_id;
        EdgePos_t part_start = partition_map_vertex_to_additions[part][2*part_v];
#ifdef RANDOM_WALK
        const EdgePos_t part_end = part_start + ((hop == 0) ? part_additions_sizes[part][2*part_v + 1] : part_additions_sizes[part][root_to_linear_thread[part_v]]);
#else
        const EdgePos_t part_end = part_start + part_additions_sizes[part][2*part_v + 1];
#endif
        EdgePos_t final_idx = final_map_vertex_to_additions[hop][0][2*v];
        const EdgePos_t final_end = final_idx + final_map_vertex_to_additions[hop][0][2*v + 1];
        std::unordered_set<VertexID> set_neighbors;
#ifdef RANDOM_WALK
        {
          EdgePos_t idx = ((hop == 0) ? 2*part_v + 1 : root_to_linear_thread[part_v]);
          if (!(part_additions_sizes[part][idx] <= final_map_vertex_to_additions[hop][0][2*part_v + 1]))
            std::cout << "v " << v << " p " << part_additions_sizes[part][idx] << " f " << final_map_vertex_to_additions[hop][0][2*part_v + 1] << std::endl;
          assert (part_additions_sizes[part][idx] <= final_map_vertex_to_additions[hop][0][2*part_v + 1]);
        }
#else
        if (!(part_additions_sizes[part][2*part_v + 1] <= final_map_vertex_to_additions[hop][0][2*v + 1]))
          std::cout << "v " << v << " p " << part_additions_sizes[part][2*part_v + 1] << " f " << final_map_vertex_to_additions[hop][0][2*v + 1] << std::endl;
        assert (part_additions_sizes[part][2*part_v + 1] <= final_map_vertex_to_additions[hop][0][2*v + 1]);
#endif
#ifdef RANDOM_WALK
        if (hop == 0) {
          memcpy (&neighbors[hop][final_idx], &part_neighbors[part][part_start], (part_end-part_start)*sizeof(VertexID));
        } else {
          if (part_end > part_start) {
            EdgePos_t idx = root_to_linear_thread[part_start];
            if (idx != -1) {
              if (part_neighbors[part][idx] == -1) {
                std::cout << "v " << v << " part_end " << part_end << " part_start " << part_start << std::endl;
              }
              assert (part_neighbors[part][idx] != -1);
              neighbors[hop][final_idx] = part_neighbors[part][idx];
            }
          }
        }
#else
        memcpy (&neighbors[hop][final_idx], &part_neighbors[part][part_start], (part_end-part_start)*sizeof(VertexID));
#endif
#if 0
        for (EdgePos_t idx = part_start; idx < part_end; idx++) {
          if (!(final_idx < final_end)) {
            printf ("final_idx %d final_end %d part_start %d part_end %d v %d\n", final_idx, final_end, v, part_start, part_end);
          }
          assert (final_idx < final_end);
          neighbors[hop][final_idx++] = part_neighbors[part][idx];
// #ifdef REMOVE_DUPLICATES_ON_GPU
//          set_neighbors.insert (part_neighbors[part][idx]);
// #endif
        }
#endif
// #ifdef REMOVE_DUPLICATES_ON_GPU
        // if (set_neighbors.size () != (part_end - part_start)) {
        //   printf ("v %d set_neighbors.size () %d (part_end - part_start) %d\n",
        //           v, set_neighbors.size (), (part_end - part_start));
        //   printf ("set_neighbors is:\n");
        //   print_container(set_neighbors);
        //   printf ("part_neighbors is:\n");
        //   for (int idx = part_start; idx < part_end; idx++) {
        //     printf ("%d, ", part_neighbors[part][idx]);
        //   }
        //   printf ("\n");
        // }
        // assert (set_neighbors.size () == (part_end - part_start));
// #endif
      }

      delete part_neighbors[part];
      delete part_additions_sizes[part];
      PinnedMemory::pinned_memory_heap.free(partition_map_vertex_to_additions[part]);
    }

    double cpu_part_t2 = convertTimeValToDouble(getTimeOfDay());

    std::cout << "CPU Part " << (cpu_part_t2 - cpu_part_t1) << std::endl;

    if (device_additions_prev_hop != nullptr) {
      CHK_CU (cudaFree (device_additions_prev_hop));
    }
    CHK_CU (cudaMalloc (&device_additions_prev_hop, neighbors_sizes[hop]));
    CHK_CU (cudaMemcpy (device_additions_prev_hop, neighbors[hop], 
                        neighbors_sizes[hop], cudaMemcpyHostToDevice));
      
  }
  double end_to_end_t2 = convertTimeValToDouble(getTimeOfDay ());

  if (device_additions_prev_hop != nullptr) {
    CHK_CU (cudaFree (device_additions_prev_hop));
  }

  for (int i = 0; i < N_HOPS; i++) {
    for (int part = 0; part < (int)csr_partitions.size (); part++) {
      cudaFree (device_map_vertex_to_additions[i][part]);
    }
  }

  std::cout << "Getting embeddings from GPU" << std::endl;
  EdgePos_t total_neighbors[N_HOPS] = {0};
  std::vector <std::vector<VertexID>> produced_embeddings(csr->get_n_vertices (), std::vector<VertexID>());
  for (VertexID vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    EdgePos_t produced_embedding_size = 0;
    for (int hop = 0; hop < N_HOPS; hop++) {
      int n_additions = additions_sizes[hop][2*vertex + 1];
      produced_embedding_size += n_additions;
      total_neighbors[hop] += n_additions;
    }
    //std::cout << " input_embedding_idx " << input_embedding_idx << std::endl;
    EdgePos_t copied = 0;
    produced_embeddings[vertex] = std::vector<VertexID>(produced_embedding_size);
    
    for (int hop = 0; hop < N_HOPS; hop++) {
      EdgePos_t start_idx = final_map_vertex_to_additions[hop][0][2*vertex];
      EdgePos_t n_additions = additions_sizes[hop][2*vertex + 1];
      //std::cout << "i " << input_embedding_idx << " produced_embedding_size " << produced_embedding_size << " global_mem_idx " << global_mem_idx << std::endl;
      VertexID* ptr = &produced_embeddings[vertex][0];
      memcpy (ptr + copied, &neighbors[hop][start_idx], sizeof(VertexID)*n_additions);

      copied += n_additions;
    }

    //VectorVertexEmbedding embedding = VectorVertexEmbedding ((uint32_t)produced_embedding_size, global_mem_idx, true);
    //produced_embeddings.push_back (embedding);
  }

  #ifdef RANDOM_WALK
    std::cout<<"rand_walk_time " << rand_walk_time << " secs "<<std::endl;
  #endif
  std::cout << "GPU Time: " << gpu_time << " secs" << std::endl;
  std::cout << "End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  std::cout << "Total 2-hop neighbors " << total_neighbors[0]+total_neighbors[1]+total_neighbors[2]+total_neighbors[3]+total_neighbors[4]+total_neighbors[5]+total_neighbors[6]+total_neighbors[7]+total_neighbors[8]+total_neighbors[9] << std::endl;

#ifdef CHECK_RESULT
  std::cout << "Generating CPU Embeddings:" << std::endl;
  double cpu_t1 = convertTimeValToDouble (getTimeOfDay ());
  std::vector<std::unordered_set<VertexID>> hops = n_hop_cpu_distinct (csr, N_HOPS);
  double cpu_t2 = convertTimeValToDouble (getTimeOfDay ());
  std::cout << "CPU Time: " << (cpu_t2 - cpu_t1) << " secs" << std::endl;
  
  assert (produced_embeddings.size () == hops.size ());
  for (size_t idx = 0; idx < produced_embeddings.size (); idx++) {
    std::unordered_set<VertexID> cpu_set = std::unordered_set<VertexID> (hops[idx].begin (), hops[idx].end ());
    std::vector<VertexID> vector_hops;
    vector_hops.insert (vector_hops.begin (), cpu_set.begin(), cpu_set.end ());
    std::sort (vector_hops.begin (), vector_hops.end ());
    std::vector<VertexID> gpu_vector = produced_embeddings [idx];
    std::unordered_set<VertexID> gpu_vector_set = std::unordered_set<VertexID> (gpu_vector.begin (), gpu_vector.end ());
    gpu_vector = std::vector<VertexID> (gpu_vector_set.begin (), gpu_vector_set.end ());
    std::sort (gpu_vector.begin (), gpu_vector.end ());

    if (vector_hops != gpu_vector) {
      std::cout << "checking for vertex " << idx << " start " << final_map_vertex_to_additions[0][0][2*idx] << " " << additions_sizes[0][2*idx+1] << std::endl;
      std::cout << "size " << vector_hops.size () << " " << gpu_vector.size () << std::endl;
      #if 1
      for (size_t i = 0; i < max (vector_hops.size (), gpu_vector.size ()); i++) {
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
#endif 
#ifdef PINNED_MEMORY
  // cudaFree (global_mem_ptr);
#else
#endif
  std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  std::cout << "Time spent in Streams " << total_stream_time << std::endl;
}