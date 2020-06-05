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
#include <curand_kernel.h>
#include <cuda.h>

//TODO-List:
//[] Divide main() function in several small functions.
//[] Divide the code in several include files that can be included in the API.
//[] In GPU Kernels, do refactoring and move them to other places.
//[] Use vectors instead of dynamic arrays and new.
//[] Convert these vectors to a new array type that does not do initialization of data.
//[] Use MKL or cuSPARSE to do the matrix transpose or sorting
//[] A configuration that specifies all the parameters.
//[] Use Templates for cleaner code of Sampler

//Supported:
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
//const int N = 4847571;
//const int N_EDGES = 68556521;

//Not supportred:
//com-orkut.ungraph
// const int N = 3072441;
// const int N_EDGES = 117185083;

#include "csr.hpp"
#include "utils.hpp"
#include "pinned_memory_alloc.hpp"
#include "sampler.cuh"
#include "rand_num_gen.cuh"

using namespace utils;
using namespace GPUUtils;

#define MAX_EDGES (2*MAX_LOAD_PER_TB)
//#define USE_PARTITION_FOR_SHMEM
#define MAX_HOP_VERTICES_IN_SH_MEM (MAX_VERTICES_PER_TB)
#define ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM

#define GRAPH_PARTITION_SIZE (2UL*1024*1024UL*1024UL) //24 KB is the size of each partition of graph
//#define REMOVE_DUPLICATES_ON_GPU
//#define CHECK_RESULT

//For mico, 512 works best
const int N_THREADS = 256;

//TODO try for larger random walks to improve results

#define MAX_LOAD_PER_TB (N_THREADS)
#define MAX_VERTICES_PER_TB 1
#if MAX_VERTICES_PER_TB < 1
  #error "MAX_VERTICES_PER_TB should be greater than or equal to 1"
#endif

#define WARP_HOP

const int ALL_NEIGHBORS = -1;

//GraphSage 2-hop sampling
const bool has_random = true;
__host__ __device__ int size() {return 2;}

__host__ __device__ 
int sampleSize(int k) {return 1;
  return ((k == 0) ? 10 : 5);}

__device__ inline
VertexID next(int k, const VertexID src, const VertexID root, 
              const CSR::Edge* src_edges, const EdgePos_t num_edges,
              const EdgePos_t neighbrId, 
              const RandNumGen* rand_num_gen,
              Sampler& sampler,
              curandState* state)
{
  //EdgePos_t id = RandNumGen::rand_int(state,num_edges);
  return src_edges[neighbrId%num_edges];
}

// #define Node2Vec
//#define UniformRandWalk
//Node2Vec
#ifdef Node2Vec
  #define RANDOM_WALK

  const bool has_random = true;

  __host__ __device__ int size() {return 10;}

  __host__ __device__ 
  int sampleSize(int k) {return 1;}

  #define node2vec_p 2.0f
  #define node2vec_q 0.5f

  __device__ inline
  VertexID next_random_walk(int k, const VertexID src, const VertexID root, 
                const CSR::Edge* src_edges, const EdgePos_t num_edges,
                const EdgePos_t neighbrId,
                const RandNumGen* rand_num_gen,
                Sampler* sampler,
                CSRPartition* graph,
                curandState* state)
  {
    Node2VecSampler& node2vec_sampler = (Node2VecSampler&)sampler;
    VertexID last_stop;
    if (k == 1) {
      last_stop = root;
    } else {
      last_stop = node2vec_sampler.get_last_stop();
    }

  #ifndef NDEBUG
    if (!(last_stop >= 0 && last_stop < graph->get_n_vertices()))
      printf ("last_stop %d graph->get_n_vertices() %d for root %d sampler 0x%p\n", last_stop, graph->get_n_vertices(), root, sampler);
    assert (last_stop >= 0 && last_stop < graph->get_n_vertices());
  #endif
    int thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
    // if (graph->get_n_edges_for_vertex(src) <= 2) 
    // {
    //   node2vec_sampler.set_last_stop(src);
    //   EdgePos_t id = rand_num_gen->rand_neighbor(thread_idx, 0, num_edges);
    //   return src_edges[id];
    // }
    
    const EdgePos_t width = num_edges;
    const float height = max(1.0/node2vec_p, max(1.0f, 1.0/node2vec_q));
    int ii = 0;
    while (true) {
      const EdgePos_t x = RandNumGen::rand_int(state,width);//rand_num_gen->rand_neighbor(thread_idx, ii++, width);
      const float y = curand_uniform(state)*height;//rand_num_gen->rand_float(thread_idx, ii++)*height;
      const VertexID neighbr = src_edges[x];
      const float neighbrH = (last_stop == neighbr) ? 1/node2vec_p : 
                        (graph->has_edge_logn(last_stop, neighbr) ? 1 :
                        1/node2vec_q);
      if (round(y) <= round(neighbrH)) {
        node2vec_sampler.set_last_stop(src);
        return neighbr;
      }
    }
  }

#else 
  #if 0
  // const bool has_random = true;

  // __host__ __device__ int size() {return 10;}

  // __host__ __device__ 
  // int sampleSize(int k) {return 1;}

  // #define node2vec_p 2.0f
  // #define node2vec_q 0.5f

  // __device__ inline
  // VertexID next_random_walk(int k, const VertexID src, const VertexID root, 
  //               const CSR::Edge* src_edges, const EdgePos_t num_edges,
  //               const EdgePos_t neighbrId,
  //               const RandNumGen* rand_num_gen,
  //               Sampler* sampler,
  //               CSRPartition* graph,
  //               curandState* state)
  // {
  //   // if (graph->get_n_edges_for_vertex(src) <= 2) 
  //   // {
  //   //   node2vec_sampler.set_last_stop(src);
  //   //   EdgePos_t id = rand_num_gen->rand_neighbor(thread_idx, 0, num_edges);
  //   //   return src_edges[id];
  //   // }
    
  //   const EdgePos_t width = num_edges;
  //   const float height = max();
  //   int ii = 0;
  //   while (true) {
  //     const EdgePos_t x = RandNumGen::rand_int(state,width);
  //     const float y = curand_uniform(state)*height;
  //     const VertexID neighbr = src_edges[x];
  //     const float neighbrH = (last_stop == neighbr) ? 1/node2vec_p : 
  //                       (graph->has_edge_logn(last_stop, neighbr) ? 1 :
  //                       1/node2vec_q);
  //     if (round(y) <= round(neighbrH)) {
  //       node2vec_sampler.set_last_stop(src);
  //       return neighbr;
  //     }
  //   }
  // }
  #endif
  #ifdef UniformRandWalk
    #define RANDOM_WALK

    //Uniform Random Walk
    const bool has_random = true;
    __host__ __device__ int size() {return 10;}
    __host__ __device__ 
    int sampleSize(int k) {return 1;}

    __device__ inline
    VertexID next_random_walk(int k, const VertexID src, const VertexID root, 
                  const CSR::Edge* src_edges, const EdgePos_t num_edges,
                  const EdgePos_t neighbrId, const RandNumGen* rand_num_gen,
                  Sampler* sampler, CSRPartition* graph, curandState* state)
    {
      EdgePos_t id = RandNumGen::rand_int(state, num_edges);

      return src_edges[id];
    }
  #else
    __device__ inline
    VertexID next_random_walk(int k, const VertexID src, const VertexID root, 
                  const CSR::Edge* src_edges, const EdgePos_t num_edges,
                  const EdgePos_t neighbrId, const RandNumGen* rand_num_gen,
                  Sampler* sampler, CSRPartition* graph, curandState* state)
    {
      assert (false);
     return -1;
    }
  #endif
#endif

__host__ __device__ 
bool distinct(VertexID root) {return false;}

#include "check_results.cu"

__host__ __device__
EdgePos_t new_neighbors_size(int hop, EdgePos_t num_edges)
{
  return (sampleSize(hop) == ALL_NEIGHBORS) ? num_edges : min((EdgePos_t)sampleSize(hop),num_edges);
}

__global__ void __launch_bounds__(N_THREADS) 
run_hop_parallel_single_step_device_level (int N_HOPS, int hop, 
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
              const RandNumGen* rand_num_gen,
              Sampler* samplers,
              curandState* curand_states
#ifndef NDEBUG
              , unsigned long long int* profile_branch_1, unsigned long long int* profile_branch_2
#endif
)
{
#ifdef USE_PARTITION_FOR_SHMEM
  extern __shared__ VertexID shmem_csr_edges[];
  __shared__ EdgePos_t shmem_n_edges;
  __shared__ EdgePos_t shmem_start_edge_idx;
  __shared__ EdgePos_t shmem_end_edge_idx;
#endif
  
  VertexID hop_vertex = source_vertex;
  VertexID part_vertex_id = csr->get_vertex_idx(hop_vertex);

#ifdef USE_PARTITION_FOR_SHMEM
  if (threadIdx.x == 0) {
    shmem_n_edges = csr->get_n_edges_for_vertex(hop_vertex);
    shmem_start_edge_idx = csr->get_start_edge_idx (hop_vertex);
    shmem_end_edge_idx = csr->get_end_edge_idx (hop_vertex);
  }
  __syncthreads();
  const EdgePos_t n_edges = shmem_n_edges;
  const EdgePos_t start_edge_idx = shmem_start_edge_idx;
  const EdgePos_t end_edge_idx = shmem_end_edge_idx;
  for (EdgePos_t e = 0; e < n_edges; e += blockDim.x) {
    EdgePos_t edge_pos = e + threadIdx.x;
    if (edge_pos < n_edges) {
      shmem_csr_edges[edge_pos] = csr->get_edge(shmem_start_edge_idx + edge_pos);
    }
  }

  __syncthreads();
#else
  const EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
  const EdgePos_t start_edge_idx = csr->get_start_edge_idx (hop_vertex);
  const EdgePos_t end_edge_idx = csr->get_end_edge_idx (hop_vertex);
#endif
  const EdgePos_t num_new_neighbors = new_neighbors_size(hop, n_edges);
  const int shfl_warp_size = num_edges_to_warp_size(num_new_neighbors, 
                                                    SourceVertexExec_t::DeviceLevel);  
  const int N_WARPS = blockDim.x/shfl_warp_size;
  const EdgePos_t num_roots = map_vertex_to_hop_vertex_data[2*part_vertex_id + 1];
  const EdgePos_t num_roots_start = blockIdx.x * N_WARPS;
  EdgePos_t hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id];
  const EdgePos_t hop_vertex_end_idx = min (num_roots, num_roots_start +  blockDim.x - 1);
  const EdgePos_t num_roots_per_tb = hop_vertex_end_idx - hop_vertex_start_idx + 1;  

  const int warpid = threadIdx.x/shfl_warp_size;
  const int laneid = threadIdx.x%shfl_warp_size;

  EdgePos_t root_vertex_idx = num_roots_start + warpid;
  if (root_vertex_idx >= hop_vertex_end_idx) {
    return;
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
  
  if (laneid%shfl_warp_size == 0) {
    e = utils::atomicAdd (end, num_new_neighbors);
  }

  EdgePos_t _e;
  
  _e = __shfl_sync (warp_hop_mask, e, 0, shfl_warp_size);  
  assert (_e != -1);

  for (EdgePos_t neighbor_idx = 0; neighbor_idx < num_new_neighbors; neighbor_idx += shfl_warp_size) {
    if (neighbor_idx + laneid%shfl_warp_size >= num_new_neighbors)
      break;

#ifdef USE_PARTITION_FOR_SHMEM
    VertexID edge = next(hop, hop_vertex, root_vertex, &shmem_csr_edges[0], 
                         n_edges, (EdgePos_t)(neighbor_idx + laneid%shfl_warp_size),
                         rand_num_gen, samplers[root_partition->get_vertex_idx(root_vertex)],
                         &curand_states[root_vertex]);
#else
    VertexID edge = next(hop, hop_vertex, root_vertex, csr->get_edges(source_vertex), 
                         n_edges, (EdgePos_t)(neighbor_idx + laneid%shfl_warp_size),
                         rand_num_gen, samplers[root_partition->get_vertex_idx(root_vertex)],
                         &curand_states[root_vertex]);
#endif
    EdgePos_t addr = start + _e + neighbor_idx + laneid%shfl_warp_size;
  #ifndef NDEBUG
    if (!(addr < num_neighbors)) {
      printf ("v %d start %d addr %d num_neighbors %d\n", root_vertex, start, addr, num_neighbors);
    }
    assert (addr < num_neighbors);
    if (!(addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1])) {
      printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1], root_vertex);
    }
    assert (addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1]);
    assert (addr >= start);
    if (embeddings_additions[addr] != -1) {
      //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
      printf ("not -1 at %ld hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
    }
    assert (embeddings_additions[addr] == -1);
  #endif
    embeddings_additions[addr] = edge;
  }
// #endif
}

#include "fixed_size_kernels.cuh"

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
              char* device_src_vertices_for_device_level,
              RandNumGen* rand_num_gen,
              Sampler* samplers,
              curandState* curand_states
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
  assert (last_src_vertex <= csr->last_vertex_id);
  assert (csr->first_vertex_id <= last_src_vertex);
  int thread_idx = threadIdx.x + blockDim.x*blockIdx.x;
  
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
        do {
          vertices[n_vertex_load] = ::atomicAdd(source_vertex_idx, 1);
        } while(vertices[n_vertex_load] <= csr->last_vertex_id and 
                device_src_vertices_for_device_level[vertices[n_vertex_load] - csr->first_vertex_id] == 1);

        if (vertices[n_vertex_load] > csr->last_vertex_id || 
            device_src_vertices_for_device_level[vertices[n_vertex_load] - csr->first_vertex_id] == 1) {
          break;
        }
        
        assert (vertices[n_vertex_load] >= 0 && vertices[n_vertex_load] <= csr->last_vertex_id && 
                device_src_vertices_for_device_level[vertices[n_vertex_load] - csr->first_vertex_id] == 0);
        EdgePos_t start_edge_idx = csr->get_start_edge_idx (vertices[n_vertex_load]);
        const EdgePos_t end_edge_idx = csr->get_end_edge_idx (vertices[n_vertex_load]);
        const EdgePos_t n_edges = (end_edge_idx != -1) ? (end_edge_idx - start_edge_idx + 1) : 0;
        VertexID root_vertices = map_vertex_to_hop_vertex_data[2*csr->get_vertex_idx(vertices[n_vertex_load]) + 1];
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
        int shfl_warp_size = num_edges_to_warp_size(new_neighbors_size(hop, n_edges), SourceVertexExec_t::BlockLevel);

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
    int warpid = threadIdx.x/32;
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
      VertexID part_vertex_id = csr->get_vertex_idx(vertices[_curr_vertex_id]);
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
        end_edge_idx = csr->get_end_edge_idx (hop_vertex);
        EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
        __syncwarp (warp_hop_mask);

        EdgePos_t* end = &previous_stage_filled_range[2*(vertex - root_partition->first_vertex_id) + 1];
        if (end_edge_idx != -1) {
          __syncwarp (warp_hop_mask);
          
          EdgePos_t e = -1;
          EdgePos_t num_new_neighbors = new_neighbors_size(hop, n_edges);
          int shfl_warp_size = num_edges_to_warp_size(num_new_neighbors, SourceVertexExec_t::BlockLevel);
          if (laneid%shfl_warp_size == 0) {
            e = utils::atomicAdd (end, num_new_neighbors);
          }
          EdgePos_t _e;
          
          _e = __shfl_sync (warp_hop_mask, e, 0, shfl_warp_size);  

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
          for (EdgePos_t neighbor_idx = 0; neighbor_idx < num_new_neighbors; 
               neighbor_idx += shfl_warp_size) {
            if (neighbor_idx + laneid%shfl_warp_size >= num_new_neighbors)
              break;
            EdgePos_t new_neighbor_idx = (EdgePos_t)(neighbor_idx + laneid%shfl_warp_size);
            VertexID edge = next(hop, hop_vertex, root_vertex, 
                                 csr->get_edges(hop_vertex), 
                                 n_edges, new_neighbor_idx,
                                 rand_num_gen, 
                                 samplers[root_partition->get_vertex_idx(root_vertex)],
                                 &curand_states[thread_idx]);
            EdgePos_t addr = start + _e + neighbor_idx + laneid%shfl_warp_size;
  #ifndef NDEBUG
            if (!(addr < num_neighbors)) {
              printf ("v %d start %d addr %d num_neighbors %d\n", vertex, start, addr, num_neighbors);
            }
            assert (addr < num_neighbors);
            if (!(addr < start + map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id) + 1])) {
              printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id) + 1], vertex);
            }
            assert (addr < start + map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id) + 1]);
            assert (addr >= start);
            if (embeddings_additions[addr] != -1) {
              //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
              printf ("not -1 at %d hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
            }
            assert (embeddings_additions[addr] == -1);
  #endif
            embeddings_additions[addr] = edge;
          }
#endif
        }

        __syncwarp (warp_hop_mask);
      }
    }

    __syncwarp ();
    
    if (last_hop_vertex_id != -1 && last_hop_vertex_roots_remaining != -1) {
      VertexID part_vertex_id = csr->get_vertex_idx(vertices[last_hop_vertex_id]);
      VertexID hop_vertex_start_idx = map_vertex_to_hop_vertex_data[2*part_vertex_id];
      //VertexID n_root_vertices = map_vertex_to_hop_vertex_data[2*part_vertex_id + 1];
      
      __syncthreads ();
      
      VertexID _hop_vertex = vertices[last_hop_vertex_id];
      const EdgePos_t n_edges = csr->get_n_edges_for_vertex(_hop_vertex);
      EdgePos_t num_new_neighbors = new_neighbors_size(hop, n_edges);
      int shfl_warp_size = num_edges_to_warp_size(num_new_neighbors, SourceVertexExec_t::BlockLevel);
      const int N_WARPS = blockDim.x/shfl_warp_size;
      const int warpid = threadIdx.x/shfl_warp_size;

      const EdgePos_t start_edge_idx = csr->get_start_edge_idx (_hop_vertex);
      const EdgePos_t end_edge_idx = csr->get_end_edge_idx (_hop_vertex);

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
            e = utils::atomicAdd (end, num_new_neighbors);
            // if (root_vertex == 37692) {
            //   printf ("%d: e %ld end %ld threadIdx.x %d n_edges %ld root_idx %d\n", __LINE__, e, *end, threadIdx.x, n_edges, root_idx);
            // }
          }
          
          EdgePos_t _e;
          
          _e = __shfl_sync (FULL_MASK, e, 0, shfl_warp_size);  
          assert (_e != -1);

          for (EdgePos_t neighbor_idx = 0; neighbor_idx < num_new_neighbors; 
            neighbor_idx += shfl_warp_size) {
            if (neighbor_idx + laneid%shfl_warp_size >= num_new_neighbors)
              break;
            EdgePos_t new_neighbor_idx = (EdgePos_t)(neighbor_idx + laneid%shfl_warp_size);
#ifdef USE_PARTITION_FOR_SHMEM
            VertexID edge = next(hop, hop_vertex, root_vertex, 
                                 shmem_csr_edges, 
                                 n_edges, new_neighbor_idx,
                                 rand_num_gen,
                                 samplers[root_partition->get_vertex_idx(root_vertex)],
                                 &curand_states[thread_idx]);
#else
            VertexID edge = next(hop, hop_vertex, root_vertex, 
                                 csr->get_edges(hop_vertex), 
                                 n_edges, new_neighbor_idx,
                                 rand_num_gen,
                                 samplers[root_partition->get_vertex_idx(root_vertex)],
                                 &curand_states[thread_idx]);
#endif
            EdgePos_t addr = start + _e + neighbor_idx + laneid%shfl_warp_size;
#ifndef NDEBUG
            if (!(addr < num_neighbors)) {
              printf ("v %d start %d addr %d num_neighbors %d\n", root_vertex, start, addr, num_neighbors);
            }
            assert (addr < num_neighbors);
            if (!(addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1])) {
              printf ("addr %d max-end %d v %d\n", addr, start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1], root_vertex);
            }
            assert (addr < start + map_orig_embedding_to_additions[2*(root_vertex - root_partition->first_vertex_id) + 1]);
            assert (addr >= start);
            if (embeddings_additions[addr] != -1) {
              //printf ("embeddings_additions addrs %x\n", &embeddings_additions[0]);
              printf ("not -1 at %d hop %d root %d src %d start %d, value %d\n", addr, hop, root_vertex, hop_vertex, start, embeddings_additions[addr]);
            }
            assert (embeddings_additions[addr] == -1);
#endif
            embeddings_additions[addr] = edge;
          }
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
    
    if (csr->get_n_edges_for_vertex(source_vertex) != 0) {
      assert (idx < csr->get_n_vertices () * 2);
      assert (idx >= 0);
#ifdef RANDOM_WALK
      const EdgePos_t n_edges = 1;
#else
      const EdgePos_t n_edges = new_neighbors_size(hop, end_edge_idx - start_edge_idx + 1);
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

#include "distinct.cuh"

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

#include "graph_partitioner.cuh"

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
    assert (sampleSize(hop) == 1);
    new_additions += 1;
    
    map_vertex_to_additions_curr_iter[2*(v - root_partition.first_vertex_id)] = vertex_sample_set_start_pos_fixed_size(&root_partition, v);
    map_vertex_to_additions_curr_iter[2*(v - root_partition.first_vertex_id) + 1] = 1;
#else

    if (hop == 0) {
      EdgePos_t _new_additions;
      _new_additions = new_neighbors_size(hop, csr->n_edges_for_vertex(v));
      new_additions += _new_additions;
    } else {
      const EdgePos_t start = map_vertex_to_additions_prev_hop[2*v];
      const EdgePos_t end = addition_sizes_prev_hop[2*v+1];
      for (int idx = start; idx < start + end; idx++) {
        VertexID addition = additions_prev_hop[idx];
        EdgePos_t _new_additions;
        _new_additions = new_neighbors_size(hop, csr->n_edges_for_vertex(addition)); 
        new_additions += _new_additions;
      }
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
  if (false) {
    int32_t* d_neighbors, *d_neighbors_2, *d_values_in, *d_values_out;
    CHK_CU(cudaMalloc(&d_neighbors, neighbors_sizes[hop-1]));
    CHK_CU(cudaMemcpy(d_neighbors, neighbors[hop-1], neighbors_sizes[hop-1], cudaMemcpyHostToDevice));
    CHK_CU(cudaMalloc(&d_values_in, neighbors_sizes[hop-1]));
    CHK_CU(cudaMemset(d_values_in, 0, neighbors_sizes[hop-1]));
    CHK_CU(cudaMalloc(&d_values_out, neighbors_sizes[hop-1]));
    CHK_CU(cudaMalloc(&d_neighbors_2, neighbors_sizes[hop-1]));
    // CHK_CU(cudaMalloc(&d_values_out, neighbors_sizes[hop-1]));
    //You can save the cost of storing new pair of (root,transit). just store the transit at grid_threadIdx and the existing root vertex is already at that grid_threadIdx. Now the map still exists. Hence, we do not need to store root vertex to new position.
    //We can decrease this overhead of sorting by avoiding those transit vertices in the data that are invalid by may be setting the new transit vertex to 1 << (msb + 1) (or to (1 << (msb+1)) - 1)but use msb in the sort only? Something like  this might work.
//Use this paper to decrease this overhead: https://dl.acm.org/doi/10.1145/3035918.3064043
//Or in CUB you can have (source, root) 64-bit pairs and treat these lowest bytes as 32-bit integer?
    int msb = 0;
    int r = csr->get_n_vertices();
    while (msb <= sizeof(VertexID)*8) {
      if (r == 0) 
        break;
      r = r >> 1;
      msb++;
    }

    std::cout << "msb is " << msb << std::endl;
    int32_t* d_in = d_neighbors;
    int32_t* d_out = d_neighbors_2;
    int32_t* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    //Check if the space runs out.
    //TODO: Use DoubleBuffer version that requires O(P) space.
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                    d_in, d_out, d_values_in, d_values_out, neighbors_sizes[hop-1]/4,0,msb);
    
    CHK_CU (cudaMalloc(&d_temp_storage, temp_storage_bytes));

    double t1 = convertTimeValToDouble(getTimeOfDay());
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                    d_in, d_out,  d_values_in, d_values_out, neighbors_sizes[hop-1]/4,0,msb);
    //--------------------------------------------------------------------------
    // device result check
    CHK_CU(cudaDeviceSynchronize());
    double t2 = convertTimeValToDouble(getTimeOfDay());
    std::cout << "Time " << (t2-t1) << " secs " << " num_neighbors = " << (neighbors_sizes[hop-1]/4) << std::endl;

    CHK_CU(cudaFree(d_in));
    CHK_CU(cudaFree(d_out));
    CHK_CU(cudaFree(d_values_out));
    CHK_CU(cudaFree(d_values_in));
    CHK_CU(cudaFree(d_temp_storage));
  }

  const CSRPartition& root_partition = csr_partitions[root_part_idx];
  double t1 = convertTimeValToDouble(getTimeOfDay ());
  host_src_to_roots.clear ();

  host_src_to_roots = std::vector<std::vector<std::pair <VertexID, int>>>(csr->get_n_vertices ());
  //Create per hop vertex data
  // for (VertexID v = 0; v < csr->get_n_vertices (); v++) {
  //   host_src_to_roots.push_back (std::vector<std::pair <VertexID, int> > ());
  // }
  VertexID first_vertex_id = root_partition.first_vertex_id;
  std::vector<std::list<VertexID>>  host_src_to_roots_2 = std::vector<std::list<VertexID>>
  (csr->get_n_vertices ());
  double _t1 = convertTimeValToDouble(getTimeOfDay());
// #ifdef SRC_TO_ROOT_VERTEX_IN_SORTED_ORDER
//   std::vector<std::pair<EdgePos_t, VertexID>> vertices_sorted_by_start(root_partition.get_n_vertices());
  
//   //TODO: Add a vertex and edge iterator in each root partition
//   for (VertexID v = first_vertex_id; 
//        v <= root_partition.last_vertex_id; v++) {
//     EdgePos_t start = final_map_vertex_to_additions[hop-1][0][2*v];
//     vertices_sorted_by_start.push_back(std::make_pair(start, v));
//   }

//   //Sort pair based on the first value.
//   std::sort(vertices_sorted_by_start.begin(), vertices_sorted_by_start.end());

//   for (auto pair: vertices_sorted_by_start) {
//     VertexID v = std::get<1>(pair);
// #else
  std::vector<VertexID> __new_neighbors;
  for (VertexID v = first_vertex_id; v <= root_partition.last_vertex_id; v++) {
// #endif

    EdgePos_t start = final_map_vertex_to_additions[hop-1][0][2*v];
    EdgePos_t end   = additions_sizes[hop-1][2*v + 1];

    for (EdgePos_t i = 0; i < end; i++) {
      EdgePos_t src = neighbors[hop-1][start + i];
      __new_neighbors.push_back(src);
      int part = get_partition_idx_of_vertex (csr_partitions, src);
      //TODO: Create a bitvector instead a set of source partitions?
      src_partitions.insert (part);
      assert (start + i < neighbors_sizes[hop-1]/sizeof(VertexID));
      assert (src >= 0 && src < csr->get_n_vertices());
      assert (csr_partitions[root_part_idx].get_n_edges_for_vertex(v) > 0);
      host_src_to_roots[src].push_back (std::make_pair (v, start + i));
    }
  }

  double _t2 = convertTimeValToDouble(getTimeOfDay());
  std::cout << "Core Time " << (_t2 - _t1) << " secs __new_neighbors = " << __new_neighbors.size() << std::endl;
  if (true) {
    int32_t* d_neighbors, *d_neighbors_2, *d_values_in, *d_values_out;
    CHK_CU(cudaMalloc(&d_neighbors, __new_neighbors.size()*sizeof(VertexID)));
    CHK_CU(cudaMemcpy(d_neighbors,  __new_neighbors.data(),  sizeof_vector( __new_neighbors), cudaMemcpyHostToDevice));
    CHK_CU(cudaMalloc(&d_values_in, sizeof_vector( __new_neighbors)));
    CHK_CU(cudaMemset(d_values_in, 0, sizeof_vector( __new_neighbors)));
    CHK_CU(cudaMalloc(&d_values_out, sizeof_vector( __new_neighbors)));
    CHK_CU(cudaMalloc(&d_neighbors_2, sizeof_vector( __new_neighbors)));
    // CHK_CU(cudaMalloc(&d_values_out, neighbors_sizes[hop-1]));
    //You can save the cost of storing new pair of (root,transit). just store the transit at grid_threadIdx and the existing root vertex is already at that grid_threadIdx. Now the map still exists. Hence, we do not need to store root vertex to new position.
    //We can decrease this overhead of sorting by avoiding those transit vertices in the data that are invalid by may be setting the new transit vertex to 1 << (msb + 1) (or to (1 << (msb+1)) - 1)but use msb in the sort only? Something like  this might work.
//Use this paper to decrease this overhead: https://dl.acm.org/doi/10.1145/3035918.3064043
//Or in CUB you can have (source, root) 64-bit pairs and treat these lowest bytes as 32-bit integer?
    int msb = 0;
    int r = csr->get_n_vertices();
    while (msb <= sizeof(VertexID)*8) {
      if (r == 0) 
        break;
      r = r >> 1;
      msb++;
    }

    std::cout << "msb is " << msb << std::endl;
    int32_t* d_in = d_neighbors;
    int32_t* d_out = d_neighbors_2;
    int32_t* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    //Check if the space runs out.
    //TODO: Use DoubleBuffer version that requires O(P) space.
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                    d_in, d_out, d_values_in, d_values_out, __new_neighbors.size(),0,msb);
    
    CHK_CU (cudaMalloc(&d_temp_storage, temp_storage_bytes));

    double t1 = convertTimeValToDouble(getTimeOfDay());
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, 
                                    d_in, d_out,  d_values_in, d_values_out, __new_neighbors.size(),0,msb);
    //--------------------------------------------------------------------------
    // device result check
    CHK_CU(cudaDeviceSynchronize());
    double t2 = convertTimeValToDouble(getTimeOfDay());
    std::cout << "Time " << (t2-t1) << " secs " << " num_neighbors = " << (__new_neighbors.size()) << std::endl;

    CHK_CU(cudaFree(d_in));
    CHK_CU(cudaFree(d_out));
    CHK_CU(cudaFree(d_values_out));
    CHK_CU(cudaFree(d_values_in));
    CHK_CU(cudaFree(d_temp_storage));
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

static curandState* device_curand_states = nullptr;

double random_walk(int N_HOPS, int hop, int part_idx, int root_part_idx,
                 CSRPartition& src_partition, CSRPartition& root_partition,
                 EdgePos_t& linear_threads_executed,
                 std::vector<std::vector<EdgePos_t>>& root_to_linear_thread,
                 std::vector<EdgePos_t*>& per_part_src_to_roots_positions, 
                 std::vector<VertexID*>& per_part_src_to_roots,
                 std::vector<EdgePos_t>& per_part_num_neighbors,
                 CSRPartition* device_src_csr, CSRPartition* device_root_partition,
                 VertexID* device_additions, EdgePos_t* device_additions_sizes,
                 const RandNumGen* device_rand_num_gen,
                 Sampler* device_samplers)
{
  EdgePos_t sum_all_roots = 0;
  for (VertexID src = 0; src < src_partition.get_n_vertices(); src++) {
    EdgePos_t num_roots = per_part_src_to_roots_positions[part_idx][2*src + 1];
    sum_all_roots += num_roots;
  }
  
  EdgePos_t num_grid_threads, num_block_threads, num_subwarp_threads;
  LoadBalancing::num_gpu_threads(src_partition.iterate_num_vertices(),
                                 per_part_src_to_roots_positions[part_idx],
                                 num_grid_threads, num_block_threads, 
                                 num_subwarp_threads);

  VertexID* device_thread_to_src_map;
  VertexID* device_thread_to_roots_map;
  VertexID* device_grid_level_thread_to_linear_thread;

  std::vector<std::pair<EdgePos_t, VertexID>> src_and_num_roots;
  for (VertexID src = 0; src < src_partition.get_n_vertices(); src++) {
   
    if (per_part_src_to_roots_positions[part_idx][2*src + 1] != 0) 
      src_and_num_roots.push_back(std::make_pair(per_part_src_to_roots_positions[part_idx][2*src + 1], src));
  }

  std::vector<EdgePos_t> grid_level_thread_to_linear_thread_map = std::vector<EdgePos_t>(num_grid_threads);

  std::sort(src_and_num_roots.begin(), src_and_num_roots.end());
  std::vector<VertexID> thread_to_src = std::vector<VertexID>(sum_all_roots);
  std::vector<VertexID> thread_to_roots = std::vector<VertexID>(sum_all_roots);
  
  EdgePos_t linear_thread = 0;
  VertexID pair_src_num_roots_idx;
  EdgePos_t grid_level_thread_iter = 0;
  EdgePos_t grid_level_start_linear_th_id = linear_thread;
  
  //TODO: After sorting you can find grid_level_assignment using binary search
  for (pair_src_num_roots_idx = 0; 
       pair_src_num_roots_idx < src_and_num_roots.size(); 
       pair_src_num_roots_idx++) {
    std::pair<EdgePos_t, VertexID> src_num_root = src_and_num_roots[pair_src_num_roots_idx];
    VertexID src = std::get<1>(src_num_root);
    EdgePos_t num_roots = std::get<0> (src_num_root);
    if (LoadBalancing::EnableLoadBalancing &&
        LoadBalancing::is_grid_level_assignment(num_roots)) {
      break;
    }

    if (src_partition.get_n_edges_for_vertex(src+ src_partition.first_vertex_id) == 0)
      continue;

    for (int _idx = 0; _idx < num_roots; _idx++) {
      EdgePos_t root_idx = per_part_src_to_roots_positions[part_idx][2*src] + 2*_idx;
      thread_to_src[linear_thread] = src + src_partition.first_vertex_id;
      VertexID root = per_part_src_to_roots[part_idx][root_idx];
      thread_to_roots[linear_thread] = root;
      root_to_linear_thread[root_part_idx][root_partition.get_vertex_idx(root)] = linear_thread+linear_threads_executed;
      linear_thread += 1;
    }
  }

  if (LoadBalancing::EnableLoadBalancing) {
    for (;pair_src_num_roots_idx < src_and_num_roots.size(); 
        pair_src_num_roots_idx++) {
      std::pair<EdgePos_t, VertexID> src_num_root = src_and_num_roots[pair_src_num_roots_idx];
      VertexID src = std::get<1>(src_num_root);
      EdgePos_t num_roots = std::get<0> (src_num_root);
      
      if (src_partition.get_n_edges_for_vertex(src) == 0)
        continue;

      for (int _idx = 0; _idx < num_roots; _idx++,grid_level_thread_iter++,linear_thread++) {
        EdgePos_t root_idx = per_part_src_to_roots_positions[part_idx][2*src] + 2*_idx;
        thread_to_src[linear_thread] = src + src_partition.first_vertex_id;
        VertexID root = per_part_src_to_roots[part_idx][root_idx];
        thread_to_roots[linear_thread] = root;
        root_to_linear_thread[root_part_idx][root_partition.get_vertex_idx(root)] = linear_thread+linear_threads_executed;
        grid_level_thread_to_linear_thread_map[grid_level_thread_iter] = linear_thread;
      }

      set_till_next_multiple(grid_level_thread_iter, 
                            LoadBalancing::LoadBalancingTBSizes::GridLevelTBSize, 
                            grid_level_thread_to_linear_thread_map.data(), (EdgePos_t)-1);
    }
  }

  CHK_CU(cudaMalloc(&device_thread_to_src_map, 
                    sizeof_vector(thread_to_src)));
  CHK_CU(cudaMalloc(&device_thread_to_roots_map, 
                    sizeof_vector(thread_to_roots)));
  CHK_CU(cudaMemcpy(device_thread_to_src_map, thread_to_src.data(), 
                    sizeof_vector(thread_to_src), cudaMemcpyHostToDevice));
  CHK_CU(cudaMemcpy(device_thread_to_roots_map, thread_to_roots.data(), 
                    sizeof_vector(thread_to_roots), cudaMemcpyHostToDevice));
  CHK_CU(cudaMalloc(&device_grid_level_thread_to_linear_thread,
                    sizeof_vector(grid_level_thread_to_linear_thread_map)));
  CHK_CU(cudaMemcpy(device_grid_level_thread_to_linear_thread, 
                    grid_level_thread_to_linear_thread_map.data(),
                    sizeof_vector(grid_level_thread_to_linear_thread_map),
                    cudaMemcpyHostToDevice));
  
  double t1 = convertTimeValToDouble(getTimeOfDay ());
  
  //for (VertexID src_idx = src_idx_for_grid_level_exec; 
       //src_idx < src_and_num_roots.size(); src_idx++) 
  if (grid_level_thread_iter > 0) {
    std::cout << "grid_level_thread_iter " << grid_level_thread_iter << std::endl;
    EdgePos_t num_roots = grid_level_thread_iter;
    int num_threads = min(LoadBalancing::LoadBalancingTBSizes::GridLevelTBSize, N_THREADS);
    assert(LoadBalancing::LoadBalancingTBSizes::GridLevelTBSize%num_threads == 0);
    int N_BLOCKS = thread_block_size(num_roots, (EdgePos_t)num_threads);
    run_hop_parallel_single_step_device_level_fixed_size<<<N_BLOCKS,num_threads>>> (N_HOPS, hop, device_src_csr,
                                                          device_root_partition,
                                                          device_additions,
                                                          per_part_num_neighbors[root_part_idx],
                                                          device_additions_sizes,
                                                          device_thread_to_src_map,
                                                          device_thread_to_roots_map,
                                                          device_grid_level_thread_to_linear_thread,
      linear_thread,
      grid_level_start_linear_th_id,
      num_roots,
      linear_threads_executed,
      device_rand_num_gen,
      device_samplers
  );
  }
  if (num_block_threads + num_subwarp_threads > 0) {
    EdgePos_t block_level_roots = (LoadBalancing::EnableLoadBalancing) ? num_block_threads + num_subwarp_threads : linear_thread;
    std::cout << "block_level_roots " << block_level_roots << std::endl;
    int N_BLOCKS = thread_block_size(block_level_roots, (EdgePos_t)N_THREADS);
    run_hop_parallel_single_step_block_level_fixed_size<<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_src_csr,  
      device_root_partition,
      device_additions,
      per_part_num_neighbors[root_part_idx],
      device_additions_sizes,
      device_thread_to_src_map,
      device_thread_to_roots_map,
      block_level_roots,
      linear_threads_executed,
      device_rand_num_gen,
      device_samplers,
      device_curand_states
    );
  }

  CHK_CU(cudaDeviceSynchronize());

  double t2 = convertTimeValToDouble(getTimeOfDay ());

  CHK_CU(cudaFree(device_thread_to_src_map));
  CHK_CU(cudaFree(device_thread_to_roots_map));
  CHK_CU(cudaFree(device_grid_level_thread_to_linear_thread));
  linear_threads_executed += linear_thread;

  return (t2-t1);
}

#define PINNED_MEMORY

int main (int argc, char* argv[])
{
  std::vector<Vertex> vertices;

  if (argc < 3) {
    std::cout << "Arguments: graph-file, adj-list/edge-list, binary" << std::endl;
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

  Graph graph;
  //TODO: instead of FILE pointer, file path should be passed
  if (strcmp(argv[2], "adj-list") == 0) {
    FILE* fp = fopen (graph_file, "r");
    if (fp == nullptr) {
      std::cout << "File '" << graph_file << "' not found" << std::endl;
      return 1;
    }
    graph.load_from_adjacency_list(fp);
    fclose (fp);
  } else if (strcmp(argv[2], "edge-list") == 0) {
    if (strcmp(argv[3], "binary") == 0) {
      graph.load_from_edge_list_binary(graph_file, true);
    } else {
      FILE* fp = fopen (graph_file, "r");
      if (fp == nullptr) {
        std::cout << "File '" << graph_file << "' not found" << std::endl;
        return 1;
      }
      graph.load_from_edge_list_txt(fp, true);
      fclose (fp);
    }
  } else {
    printf("Incorrect graph file type '%s'\n", argv[2]);
    abort();
  }

  //graph.print (std::cout);
  std::cout << "n_edges "<<graph.get_n_edges () <<std::endl;
  std::cout << "vertices " << graph.get_vertices ().size () << std::endl; 

  CSR* csr = new CSR(graph.get_vertices().size(), graph.get_n_edges());
  std::cout << "sizeof(CSR)"<< sizeof(CSR)<<std::endl;
  csr_from_graph (csr, graph);
  std::cout << "csr.n_vertices " << csr->get_n_vertices () << std::endl;

  std::cout << "Pinned Memory Allocated" << std::endl;

  double total_stream_time = 0;

  double_t kernelTotalTime = 0.0;
  std::vector<CSRPartition> csr_partitions;
  Sampler* samplers = new Node2VecSampler[csr->get_n_vertices()]; //TODO: Allocate for correct sampler

#ifdef ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM 
  create_csr_partitions (csr, csr_partitions, GRAPH_PARTITION_SIZE - sizeof (CSRPartition));
#else
  CSRPartition full_partition = CSRPartition (0, csr->get_n_vertices () - 1, 0, csr->get_n_edges () - 1, 
                                              csr->get_vertices (), csr->get_edges ());
  csr_partitions.push_back (full_partition);
#endif

  const int N_HOPS = size();
  
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
    EdgePos_t** partition_map_vertex_to_additions = new EdgePos_t*[csr_partitions.size ()];
    memset(partition_map_vertex_to_additions, 0, csr_partitions.size()*sizeof(EdgePos_t));
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
    std::vector<std::vector<EdgePos_t>> root_to_linear_thread = std::vector<std::vector<EdgePos_t>>(csr_partitions.size());
#endif
    /********************Get the output additions lengths*******************/
    for (int root_part_idx = 0; root_part_idx < (int)csr_partitions.size (); root_part_idx++) {
      unsigned long long num_neighbors_iter = 0;
      CSRPartition& root_partition = csr_partitions[root_part_idx];
      CSRPartition* device_root_partition;
      Sampler* device_samplers;

      copy_partition_to_gpu (root_partition, device_root_partition, device_vertex_array, device_edge_array);
      Sampler* root_samplers = (Sampler*)(((char*)samplers) + samplers[0].size()*root_partition.first_vertex_id);
#ifdef RANDOM_WALK
      copy_sampler_to_gpu (root_samplers, 
                           root_partition.get_n_vertices(), device_samplers);
#endif

      num_neighbors_iter = 0;
      
      CHK_CU (cudaMemset (device_max_neighbors_iter, 0, 
                          sizeof (unsigned long long)));
      CHK_CU (cudaMalloc (&device_map_vertex_to_additions[hop][root_part_idx], 
                          partition_map_vertex_to_additions_size (root_partition)*sizeof (EdgePos_t)));

      std::cout << "Calling cuda kernel for hop: " << hop << " partition: " << root_part_idx << " vertex = [" << csr_partitions[root_part_idx].first_vertex_id << ", "<< csr_partitions[root_part_idx].last_vertex_id << "]" << std::endl;

      int MAX_LENGTHS_N_THREADS = 128;
      int N_BLOCKS = (root_partition.get_n_vertices ()%MAX_LENGTHS_N_THREADS == 0) ? root_partition.get_n_vertices ()/128 : root_partition.get_n_vertices ()/MAX_LENGTHS_N_THREADS + 1;

      EdgePos_t* device_prev_hop_addition_sizes;

      //partition_map_vertex_to_additions[root_part_idx] = new EdgePos_t[partition_map_vertex_to_additions_size (root_partition)];
      partition_map_vertex_to_additions[root_part_idx] = (EdgePos_t*)PinnedMemory::pinned_memory_heap.malloc(partition_map_vertex_to_additions_size (root_partition)*sizeof(EdgePos_t));
      double t1 = convertTimeValToDouble(getTimeOfDay ());

      if (hop == 0) {
        num_neighbors_iter = cpu_get_max_lengths_for_vertices_single_step (hop, root_partition, csr,
          nullptr, nullptr, nullptr,
          partition_map_vertex_to_additions[root_part_idx]);
      } else {
        num_neighbors_iter = cpu_get_max_lengths_for_vertices_single_step (hop, root_partition, csr,
        final_map_vertex_to_additions[hop-1][0], 
        additions_sizes[hop-1], 
        neighbors[hop-1], 
        partition_map_vertex_to_additions[root_part_idx]);   
      }

      double t2 = convertTimeValToDouble(getTimeOfDay ());

      if (hop > 0) {
        //CHK_CU (cudaFree (device_edges_to_prev_iter_additions));
        //CHK_CU (cudaFree (device_prev_hop_addition_sizes));
      }
      
      CHK_CU (cudaMemcpy (device_map_vertex_to_additions[hop][root_part_idx],
        partition_map_vertex_to_additions[root_part_idx], 
        partition_map_vertex_to_additions_size (root_partition)*sizeof (EdgePos_t), 
        cudaMemcpyHostToDevice));
      
  
      //gpu_time += t2 - t1;
  
      std::cout << "Cuda Kernel Done " << std::endl;
      is_cuda_error (cudaGetLastError ());
      std::cout << "New " << hop << "-Hop Neighbors " << num_neighbors_iter << std::endl;

      num_neighbors += num_neighbors_iter;
      per_part_num_neighbors [root_part_idx] = num_neighbors_iter;
      std::cout << __LINE__ <<": per_part_num_neighbors [root_part_idx] = " << per_part_num_neighbors [root_part_idx] << " " << num_neighbors_iter << " " << root_part_idx << std::endl;
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

      CHK_CU (cudaMemcpy (device_additions, temp_array, per_part_num_neighbors[root_part_idx]*sizeof (VertexID), cudaMemcpyHostToDevice)); //Create a device kernel to memset.
      delete[] temp_array;
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
      VertexID vertex_with_prev_partition = -1;
      const EdgePos_t vertex_with_prev_partition_adds = -1;
#ifdef RANDOM_WALK
      EdgePos_t linear_threads_executed = 0;
      root_to_linear_thread[root_part_idx] = std::vector<EdgePos_t>(root_partition.get_n_vertices());
      for (VertexID v : root_partition.iterate_num_vertices()) {
        root_to_linear_thread[root_part_idx][v] = -1;
      }
#endif

      RandNumGen* rand_num_gen = nullptr;
      RandNumGen* device_rand_num_gen = nullptr;

      for (auto part_idx : src_partitions) {
        VertexID* device_src_to_roots;
        EdgePos_t* device_src_to_root_positions;
        CSRPartition& src_partition = csr_partitions[part_idx];
        std::cout << "Find " << hop << "-hops for root partition " << root_part_idx << " using src partition " << part_idx << std::endl;
        copy_partition_to_gpu (src_partition, device_csr, device_vertex_array, device_edge_array);
        CHK_CU (cudaMemcpy (source_vertex_idx, &src_partition.first_vertex_id,  sizeof (VertexID), cudaMemcpyHostToDevice));
        //TODO: Free source_vertex_idx
#ifdef RANDOM_WALK
        size_t num_curands;
  #ifdef ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM
        if (hop > 0) {
          num_curands = src_partition.get_n_vertices()*((sampleSize(hop) == 1) ? 1 : N_THREADS);
          CHK_CU(cudaMalloc(&device_curand_states, num_curands*sizeof(curandState)));
        } else {
          num_curands = root_partition.get_n_vertices()*((sampleSize(hop) == 1) ? 1 : N_THREADS);
          CHK_CU(cudaMalloc(&device_curand_states, num_curands*sizeof(curandState)));
        }
  

        //TODO: Amortize this cost
        //TODO: support partitions here
        //TODO: Use curandMakeMTGP32KernelState to decrease memory usage. (https://docs.nvidia.com/cuda/curand/device-api-overview.html#bit-generation-1)
        // std::cout << "sizeof(curandMakeMTGP32KernelState) = "<<sizeof(curandMakeMTGP32KernelState) << std::endl;
        size_t num_blocks = thread_block_size(num_curands, 256UL);
        init_curand_states<<<num_blocks, 256>>> (device_curand_states, num_curands);
        CHK_CU(cudaDeviceSynchronize());
  #else
        if (hop == 0) {
          num_curands = root_partition.get_n_vertices()*((sampleSize(hop) == 1) ? 1 : N_THREADS);
          CHK_CU(cudaMalloc(&device_curand_states, num_curands*sizeof(curandState)));
          size_t num_blocks = thread_block_size(num_curands, 256UL);
          init_curand_states<<<num_blocks, 256>>> (device_curand_states, num_curands);
          CHK_CU(cudaDeviceSynchronize());
        }
  #endif
#endif

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

        std::cout << "src_partition.get_n_vertices() = " << src_partition.get_n_vertices() << std::endl;
        char* src_vertices_for_device_level = new char[src_partition.get_n_vertices()]; //TODO: Allocate in PageLocked Memory
        //TODO: Make this a bitset
        char* device_src_vertices_for_device_level;
        cudaMalloc(&device_src_vertices_for_device_level, src_partition.get_n_vertices()*sizeof(char));
        memset(src_vertices_for_device_level, 0, src_partition.get_n_vertices()*sizeof(char));

        double t1 = convertTimeValToDouble(getTimeOfDay ());
        std::cout << "hop " << hop << std::endl;
        
        if (hop >= 1) {
          cudaStream_t* streams = new cudaStream_t[src_partition.get_n_vertices()];
          VertexID vertices_for_grid_level = 0;

          if (LoadBalancing::EnableLoadBalancing) {
#ifndef RANDOM_WALK
            for (VertexID src_vertex_id : src_partition.get_vertex_range()) {
              // if (src_partition.get_n_edges_for_vertex(src_vertex_id) >= N_THREADS) {
              EdgePos_t num_roots = per_part_src_to_roots_positions[part_idx][2*src_vertex_id+1];
              if (num_roots >= N_THREADS) {
                vertices_for_grid_level++;
                
                src_vertices_for_device_level[src_vertex_id] = 1;
                cudaStreamCreate(&streams[src_partition.get_vertex_idx(src_vertex_id)]);
                //int N_BLOCKS = src_partition.get_n_edges_for_vertex(src_vertex_id)/N_THREADS + 1;
                const EdgePos_t num_edges = src_partition.get_n_edges_for_vertex(src_vertex_id);
                int N_BLOCKS = thread_block_size(num_roots*num_edges_to_warp_size(new_neighbors_size(hop, num_edges),SourceVertexExec_t::DeviceLevel), (EdgePos_t)N_THREADS);
#ifdef USE_PARTITION_FOR_SHMEM
                size_t sh_mem = sizeof(VertexID)*num_edges;
                assert (sh_mem < 48*1024-sizeof(EdgePos_t));
#else
                size_t sh_mem =  0;
#endif
                std::cout << "Curand is not supported in case of device level execution" << std::endl;
                assert (false);
                run_hop_parallel_single_step_device_level <<<N_BLOCKS, N_THREADS,sh_mem,streams[src_vertex_id]>>> (N_HOPS, hop, device_csr,  
                                                                      device_root_partition,
                                                                      device_additions,
                                                                      per_part_num_neighbors[root_part_idx],
                                                                      device_additions_prev_hop,
                                                                      device_map_vertex_to_additions[hop][root_part_idx],
                                                                      device_additions_sizes,
                                                                      device_src_to_roots,
                                                                      device_src_to_root_positions,
                                                                      src_vertex_id,
                                                                      device_rand_num_gen,
                                                                      device_samplers,
                                                                      device_curand_states
      #ifndef NDEBUG
                                                                      , device_profile_branch_1,
                                                                      device_profile_branch_2
      #endif
                                                                    );

                CHK_CU(cudaGetLastError());
              }
            //CHK_CU (cudaDeviceSynchronize ());
            }
#endif
          }

          std::cout << "vertices for block level exec " << (src_partition.get_n_vertices() - vertices_for_grid_level) << std::endl;
          if (src_partition.get_n_vertices() - vertices_for_grid_level > 0) {
#ifdef RANDOM_WALK
          double t = random_walk(N_HOPS, hop, part_idx, 
                      root_part_idx, src_partition, 
                      root_partition, 
                      linear_threads_executed,
                      root_to_linear_thread,
                      per_part_src_to_roots_positions, 
                      per_part_src_to_roots,
                      per_part_num_neighbors,
                      device_csr, device_root_partition,
                      device_additions, device_additions_sizes,
                      device_rand_num_gen,
                      device_samplers);
        std::cout<<"Block Level time " << t <<" secs" << std::endl;
        rand_walk_time += t;

#else
            cudaStreamCreate(&streams[0]);
            CHK_CU(cudaMemcpyAsync(device_src_vertices_for_device_level, src_vertices_for_device_level, src_partition.get_n_vertices()*sizeof(char), cudaMemcpyHostToDevice, streams[0]));
            int N_BLOCKS = src_partition.get_n_vertices() - vertices_for_grid_level;
            run_hop_parallel_single_step_block_level <<<N_BLOCKS, N_THREADS, 0, streams[0]>>> (N_HOPS, hop, device_csr,  
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
              device_src_vertices_for_device_level,
              device_rand_num_gen,
              device_samplers,
              device_curand_states
    #ifndef NDEBUG
              , device_profile_branch_1,
              device_profile_branch_2
    #endif
            );
            CHK_CU (cudaDeviceSynchronize ());
#endif
          }

          if (LoadBalancing::EnableLoadBalancing) {
            //TODO: Destroy the streams
            // for (VertexID v = vertex_for_block_level_exec; v <= src_partition.last_vertex_id; v++) {
            //   CHK_CU(cudaStreamDestroy(streams[src_partition.get_vertex_idx(v)]));
            // }

            // CHK_CU(cudaStreamDestroy(streams[0]));

            delete[] streams;
          }

          delete[] src_vertices_for_device_level;
          CHK_CU(cudaFree(device_src_vertices_for_device_level));
        }

        if (hop == 0) {
#if 0
          int N_BLOCKS = thread_block_size(src_partition.get_n_vertices(), N_THREADS);
          float* rand = GPUUtils::gen_rand_on_gpu(src_partition.get_n_vertices());
          run_hop_parallel_single_step_block_level_fixed_size_first_step<<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, 
            device_csr, device_root_partition, device_additions, per_part_num_neighbors[root_part_idx],
            device_map_vertex_to_additions[hop][root_part_idx], device_src_to_roots,
            device_src_to_root_positions, src_partition.get_n_vertices(),
            rand);
          );
          cudaFree(rand);
#else
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
            nullptr,
            device_rand_num_gen,
            device_samplers,
            device_curand_states
  #ifndef NDEBUG
            , device_profile_branch_1,
            device_profile_branch_2
  #endif
          );

          CHK_CU(cudaGetLastError());
#endif
        }

        CHK_CU (cudaDeviceSynchronize ());
        double t2 = convertTimeValToDouble(getTimeOfDay ());
        gpu_time += t2 - t1;

        if (hop > 0) {
          CHK_CU (cudaFree (device_src_to_roots))
          CHK_CU (cudaFree (device_src_to_root_positions));


          delete[] per_part_src_to_roots[part_idx];
          delete[] per_part_src_to_roots_positions[part_idx];
        }

        CHK_CU (cudaFree ((void*)device_vertex_array));
        CHK_CU (cudaFree ((void*)device_edge_array));
        CHK_CU (cudaFree ((void*)device_csr));
#ifdef ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM
        CHK_CU (cudaFree(device_curand_states));
#endif
      }

  #ifdef PROFILE
      unsigned long profile_branch_1, profile_branch_2;
      CHK_CU (cudaMemcpy (&profile_branch_1, device_profile_branch_1, sizeof(profile_branch_1), cudaMemcpyDeviceToHost));
      CHK_CU (cudaMemcpy (&profile_branch_2, device_profile_branch_2, sizeof(profile_branch_1), cudaMemcpyDeviceToHost));

      std::cout << "profile_branch_1 " << profile_branch_1 << std::endl;
      std::cout << "profile_branch_2 " << profile_branch_2 << std::endl;
  #endif
      
      if (rand_num_gen != nullptr) {
        rand_num_gen = nullptr;
        delete rand_num_gen;
      }
      part_additions_sizes[root_part_idx] = new EdgePos_t[partition_additions_sizes_size];
      CHK_CU (cudaMemcpy (part_additions_sizes[root_part_idx], device_additions_sizes, 
        partition_additions_sizes_size, 
        cudaMemcpyDeviceToHost));
      
      if (false && hop == 0) {
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
      for (VertexID v : root_partition.get_vertex_range()) {
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
      
      std::cout << "per_part_num_neighbors [root_part_idx] = " << per_part_num_neighbors [root_part_idx] << " " << num_neighbors_iter << std::endl;
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
      } else if (vertex_with_prev_partition != -1) {
        printf ("should not be called\n");
        assert (false);
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
      } else {
        EdgePos_t start_pos = 0;
        for (VertexID v : csr_partitions[root_part_idx - 1].get_vertex_range()) {
          EdgePos_t pos = final_map_vertex_to_additions[hop][0][2*v] + final_map_vertex_to_additions[hop][0][2*v + 1];
          start_pos = max (start_pos, pos);
        }
        assert (start_pos <= num_neighbors && start_pos > 0);
        for (VertexID v = 0; v < csr_partitions[root_part_idx].get_n_vertices (); v++) {
          VertexID vertex = csr_partitions[root_part_idx].first_vertex_id + v;
          final_map_vertex_to_additions[hop][0][2*vertex] = start_pos + partition_map_vertex_to_additions[root_part_idx][2*v];
          final_map_vertex_to_additions[hop][0][2*vertex + 1] = partition_map_vertex_to_additions[root_part_idx][2*v + 1];
        }

        final_map_vertex_to_additions_iter += partition_map_vertex_to_additions_size(root_partition);
      }

      VertexID first_vertex_id = root_partition.first_vertex_id;
      if (root_part_idx == 0) {
#ifdef RANDOM_WALK
        for (VertexID v : root_partition.get_vertex_range()) {
          EdgePos_t idx = (hop > 0) ? root_to_linear_thread[0][v] : 2*v+1;
          if (idx == -1)
            additions_sizes[hop][2*v+1] = 0;
          else {
            additions_sizes[hop][2*v+1] = part_additions_sizes[root_part_idx][idx];
          }
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
          EdgePos_t idx = (hop > 0) ? root_to_linear_thread[root_part_idx][v - root_partition.first_vertex_id] : 2*(v - root_partition.first_vertex_id) +1;
          if (idx == -1)
            additions_sizes[hop][2*v+1] = 0;
          else {
            additions_sizes[hop][2*v+1] = part_additions_sizes[root_part_idx][idx];
          }
        }
#else
        memcpy (&additions_sizes[hop][2*root_partition.first_vertex_id],
                part_additions_sizes[root_part_idx], partition_additions_sizes_size);
#endif
      }

      double cpu_part_t2 = convertTimeValToDouble(getTimeOfDay());

      std::cout << "CPU Part " << (cpu_part_t2 - cpu_part_t1) << std::endl;
#ifdef RANDOM_WALK
      copy_back_sampler_from_gpu (root_samplers, root_partition.get_n_vertices(), device_samplers);
#endif
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
        const EdgePos_t part_end = part_start + ((hop == 0) ? part_additions_sizes[part][2*part_v + 1] : part_additions_sizes[part][root_to_linear_thread[part][part_v]]);
#else
        const EdgePos_t part_end = part_start + part_additions_sizes[part][2*part_v + 1];
#endif
        EdgePos_t final_idx = final_map_vertex_to_additions[hop][0][2*v];
        const EdgePos_t final_end = final_idx + final_map_vertex_to_additions[hop][0][2*v + 1];
        std::unordered_set<VertexID> set_neighbors;
#ifdef RANDOM_WALK
        EdgePos_t idx = ((hop == 0) ? 2*part_v + 1 : root_to_linear_thread[part][part_v]);
        assert (idx >= -1);
        if (idx != -1) {
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
          assert (part_end >= part_start);
          memcpy (&neighbors[hop][final_idx], &part_neighbors[part][part_start], (part_end-part_start)*sizeof(VertexID));
        } else {
          if (part_end > part_start) {
            EdgePos_t idx = root_to_linear_thread[part][part_start];
            if (idx != -1) {
              if (part_neighbors[part][idx] == -1) {
                std::cout << "v " << v << " part_end " << part_end << " part_start " << part_start << " idx " << idx << std::endl;
              }
              assert (part_neighbors[part][idx] != -1);
              neighbors[hop][final_idx] = part_neighbors[part][idx];
            }
          }
        }
#else
        memcpy (&neighbors[hop][final_idx], &part_neighbors[part][part_start], (part_end-part_start)*sizeof(VertexID));
#endif
      }

      delete[] part_neighbors[part];
      delete[] part_additions_sizes[part];
      PinnedMemory::pinned_memory_heap.free(partition_map_vertex_to_additions[part]);
    }

    double cpu_part_t2 = convertTimeValToDouble(getTimeOfDay());

    std::cout << "CPU Part " << (cpu_part_t2 - cpu_part_t1) << std::endl;

    if (device_additions_prev_hop != nullptr) {
      CHK_CU (cudaFree (device_additions_prev_hop));
    }
    //TODO: Remove this 
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
  EdgePos_t per_hop_total_neighbors[N_HOPS] = {0};
  EdgePos_t total_neighbors = 0;
  std::vector <std::vector<VertexID>> produced_embeddings(csr->get_n_vertices (), std::vector<VertexID>());
  for (VertexID vertex = 0; vertex < csr->get_n_vertices (); vertex++) {
    EdgePos_t produced_embedding_size = 0;
    for (int hop = 0; hop < N_HOPS; hop++) {
      int n_additions = additions_sizes[hop][2*vertex + 1];
      produced_embedding_size += n_additions;
      per_hop_total_neighbors[hop] += n_additions;
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

  for (int hop = 0; hop < N_HOPS; hop++) {
    total_neighbors += per_hop_total_neighbors[hop];
  }

  #ifdef RANDOM_WALK
    std::cout<<"rand_walk_time " << rand_walk_time << " secs "<<std::endl;
  #endif
  std::cout << "GPU Time: " << gpu_time << " secs" << std::endl;
  std::cout << "End to end time " << (end_to_end_t2 - end_to_end_t1) << " secs" << std::endl;
  std::cout << "Total " << N_HOPS << "-hop neighbors " << total_neighbors << std::endl;

#ifdef CHECK_RESULT
  if (sampleSize(1) != ALL_NEIGHBORS) {
    std::cout << "Results are correct? " <<check_result(csr, final_map_vertex_to_additions, additions_sizes, neighbors) << std::endl;
  } else {  
    std::cout << "Generating CPU Embeddings:" << std::endl;
    double cpu_t1 = convertTimeValToDouble (getTimeOfDay ());
    std::vector<std::unordered_set<VertexID>> hops = n_hop_cpu_distinct (csr, N_HOPS);
    double cpu_t2 = convertTimeValToDouble (getTimeOfDay ());
    std::cout << "CPU Time: " << (cpu_t2 - cpu_t1) << " secs" << std::endl;
    //Check result for all k-hop neighbors below:
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
  }
#endif 
#ifdef PINNED_MEMORY
  // cudaFree (global_mem_ptr);
#else
#endif
  std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  std::cout << "Time spent in Streams " << total_stream_time << std::endl;
}
