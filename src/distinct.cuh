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
  
  //TODO: Why is N here? Can we set it to 
  auto invalid_val = root_partition->get_n_vertices()+1;
  BlockLoadT(temp_storage.load).Load(&embeddings_additions[start], thread_items, end, invalid_val);
  
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
  BlockLoadT(temp_storage.load).Load(is_equal, thread_items, 
                                     end, invalid_val);
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