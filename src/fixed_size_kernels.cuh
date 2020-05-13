__device__ __host__ inline EdgePos_t vertex_sample_set_start_pos_fixed_size (const CSRPartition* root_partition, VertexID vertex) 
{
  assert (root_partition->has_vertex(vertex));
  return vertex-root_partition->first_vertex_id;
}

//#define USE_PARTITION_FOR_SHMEM_1

__global__ void run_hop_parallel_single_step_device_level_fixed_size (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,              
              EdgePos_t* previous_stage_filled_range,
              VertexID* thread_to_src,
              VertexID* thread_to_roots,
              VertexID* grid_level_thread_to_linear_thread_map,
              EdgePos_t total_roots,
              EdgePos_t start_linear_id,
              EdgePos_t src_num_roots,
              float* rand,
              EdgePos_t linear_threads_executed)
{
#ifdef USE_PARTITION_FOR_SHMEM_1
  __shared__ EdgePos_t src_num_edges;
  __shared__ VertexID hop_vertex;
  __shared__ EdgePos_t start_edge_idx;;
#else
  VertexID hop_vertex;
  EdgePos_t start_edge_idx;
#endif

  int device_level_thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  int linear_thread_id = grid_level_thread_to_linear_thread_map[device_level_thread_id];
  int global_thread_id = linear_thread_id;
#ifdef USE_PARTITION_FOR_SHMEM_1
  if (threadIdx.x == 0 && linear_thread_id < src_num_roots) {
    hop_vertex = thread_to_src[global_thread_id];
    src_num_edges = csr->get_n_edges_for_vertex(hop_vertex);
    start_edge_idx = csr->get_start_edge_idx (hop_vertex);
  }

  __syncthreads();
#endif
  if (linear_thread_id == -1)
    return;

#ifndef USE_PARTITION_FOR_SHMEM_1
  hop_vertex = thread_to_src[global_thread_id];
  start_edge_idx = csr->get_start_edge_idx (hop_vertex);
#endif
  VertexID vertex = thread_to_roots[global_thread_id];
  EdgePos_t start = vertex_sample_set_start_pos_fixed_size(root_partition, vertex);

#ifdef USE_PARTITION_FOR_SHMEM_1
  EdgePos_t n_edges = src_num_edges;//csr->get_n_edges_for_vertex(hop_vertex);
#else
  EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
#endif

  if (n_edges > 0) {
    previous_stage_filled_range[linear_threads_executed+global_thread_id] = 1;
    EdgePos_t _e = (EdgePos_t)round(0.5 + n_edges * rand[global_thread_id]) - 1;
    VertexID edge = csr->get_edge (start_edge_idx + _e);
    embeddings_additions[linear_threads_executed+global_thread_id] = edge;
  }
}

__global__ void run_hop_parallel_single_step_block_level_fixed_size_first_step (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,
              EdgePos_t* previous_stage_filled_range,
              VertexID* thread_to_src,
              VertexID* thread_to_roots,
              EdgePos_t total_roots,
              float* rand)
{
  VertexID vertex = blockIdx.x*blockDim.x + threadIdx.x;
  if (vertex >= total_roots) 
    return;

  EdgePos_t start = vertex_sample_set_start_pos_fixed_size(root_partition, vertex);
  EdgePos_t start_edge_idx;
  start_edge_idx = csr->get_start_edge_idx (vertex);
  EdgePos_t n_edges = csr->get_n_edges_for_vertex(vertex);
  
  if (n_edges > 0) {
    previous_stage_filled_range[vertex] = 1;
    EdgePos_t _e = (EdgePos_t)round(0.5 + n_edges * rand[vertex]) - 1;    
    VertexID edge = csr->get_edge (start_edge_idx + _e);
    embeddings_additions[vertex] = edge;
  }
}

__global__ void run_hop_parallel_single_step_block_level_fixed_size (int N_HOPS, int hop, 
              CSRPartition* csr,
              CSRPartition* root_partition,
              VertexID* embeddings_additions, 
              EdgePos_t num_neighbors,
              EdgePos_t* previous_stage_filled_range,
              VertexID* thread_to_src,
              VertexID* thread_to_roots,
              EdgePos_t total_roots,
              float* rand,
              EdgePos_t linear_threads_executed)
{
  int linear_thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  VertexID hop_vertex;
  if (linear_thread_id >= total_roots) 
    return;

  hop_vertex = thread_to_src[linear_thread_id];
  VertexID vertex = thread_to_roots[linear_thread_id];
  EdgePos_t start = vertex_sample_set_start_pos_fixed_size(root_partition, vertex);//map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)];  
  //assert (map_orig_embedding_to_additions[2*(vertex - root_partition->first_vertex_id)] == vertex);
  EdgePos_t start_edge_idx;
  start_edge_idx = csr->get_start_edge_idx (hop_vertex);
  EdgePos_t n_edges = csr->get_n_edges_for_vertex(hop_vertex);
  
  if (n_edges > 0) {
    previous_stage_filled_range[linear_thread_id+linear_threads_executed] = 1;
    EdgePos_t _e = (EdgePos_t)round(0.5 + n_edges * rand[linear_thread_id]) - 1;    
    VertexID edge = csr->get_edge (start_edge_idx + _e);
    embeddings_additions[linear_thread_id+linear_threads_executed] = edge;
  }
}