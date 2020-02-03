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

#define LINE_SIZE 1024*1024
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

#define GRAPH_PARTITION_SIZE (48 * 1024) //24 KB is the size of each partition of graph

//#define USE_CONSTANT_MEM

typedef uint8_t SharedMemElem;
typedef uint32_t VertexID;
//citeseer.graph
const int N = 3312;
const int N_EDGES = 9074;
#define ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING false

//micro.graph
//const int N = 100000;
//const int N_EDGES = 2160312;
//#define ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING true

double_t convertTimeValToDouble (struct timeval _time)
{
  return ((double_t)_time.tv_sec) + ((double_t)_time.tv_usec)/1000000.0f;
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

enum BUFFER_STATUS {
  GPU_USING,
  CPU_COPYING,
  READY_CPU_COPYING,
  FREE,
};

class Vertex
{
private:
  int id;
  int label;
  std::vector <int> edges;

public:
  Vertex (int _id, int _label) : label(_label), id (_id)
  {
  }

  void set_id (int _id) {id = _id;}
  int get_id () {return id;}
  int get_label () {return label;}
  void add_edge (int vertexID) {edges.push_back (vertexID);}
  void sort_edges () {std::sort (edges.begin(), edges.end ());}
  std::vector <int>& get_edges () {return edges;}
  void print (std::ostream& os)
  {
    os << id << " " << label << " ";
    for (auto edge : edges) {
      os << edge << " ";
    }

    os << std::endl;
  }

  static bool compare_vertex (Vertex& v1, Vertex& v2) 
  {
    return v1.edges.size () < v2.edges.size ();
  }
};

int chars_in_int (int num)
{
  if (num == 0) return sizeof(char);
  return (int)((ceil(log10(num))+1)*sizeof(char));
}

class Graph
{
private:
  std::vector<Vertex> vertices;
  int n_edges;

public:
  Graph (std::vector<Vertex> _vertices, int _n_edges) :
    vertices (_vertices), n_edges(_n_edges)
  {}

  const std::vector<Vertex>& get_vertices () {return vertices;}
  int get_n_edges () {return n_edges;}
};



class CSR
{
public:
  struct Vertex
{
  int id;
  int label;
  int start_edge_id;
  int end_edge_id;
  __host__ __device__
  Vertex ()
  {
    id = -1;
    label = -1;
    start_edge_id = -1;
    end_edge_id = -1;
  }

  void set_from_graph_vertex (::Vertex& vertex)
  {
    id = vertex.get_id ();
    label = vertex.get_label ();
  }

  void set_start_edge_id (int start) {start_edge_id = start;}
  void set_end_edge_id (int end) {end_edge_id = end;}
};

typedef int Edge;

  CSR::Vertex vertices[N];
  CSR::Edge edges[N_EDGES];
  int n_vertices;
  int n_edges;

public:
  CSR (int _n_vertices, int _n_edges)
  {
    n_vertices = _n_vertices;
    n_edges = _n_edges;
  }

  __host__ __device__
  CSR ()
  {
    n_vertices = N;
    n_edges = N_EDGES;
  }

  void print (std::ostream& os)
  {
    for (int i = 0; i < n_vertices; i++) {
      os << vertices[i].id << " " << vertices[i].label << " ";
      for (int edge_iter = vertices[i].start_edge_id;
           edge_iter <= vertices[i].end_edge_id; edge_iter++) {
        os << edges[edge_iter] << " ";
      }
      os << std::endl;
    }
  }

  __host__ __device__
  int get_start_edge_idx (int vertex_id)
  {
    if (!(vertex_id < n_vertices && 0 <= vertex_id)) {
      printf ("vertex_id %d, n_vertices %d\n", vertex_id, n_vertices);
      assert (false);
    }
    return vertices[vertex_id].start_edge_id;
  }

  __host__ __device__
  int get_end_edge_idx (int vertex_id)
  {
    assert (vertex_id < n_vertices && 0 <= vertex_id);
    return vertices[vertex_id].end_edge_id;
  }

  __host__ __device__
  bool has_edge (int u, int v)
  {
    //TODO: Since graph is sorted, do this using binary search
    for (int e = get_start_edge_idx (u); e <= get_end_edge_idx (u); e++) {
      if (edges[e] == v) {
        return true;
      }
    }

    return false;
  }

  __host__ __device__
  const CSR::Edge* get_edges () {return &edges[0];}

  __host__ __device__
  const CSR::Vertex* get_vertices () {return &vertices[0];}

  __host__ __device__
  int get_n_vertices () {return n_vertices;}

  __host__ __device__
  void copy_vertices (CSR* src, int start, int end)
  {
    for (int i = start; i < end; i++) {
      vertices[i] = src->get_vertices()[i];
    }
  }

  __host__ __device__
  void copy_edges (CSR* src, int start, int end)
  {
    for (int i = start; i < end; i++) {
      edges[i] = src->get_edges ()[i];
    }
  }

  __host__ __device__
  int get_n_edges () {return n_edges;}
};

class CSRPartition
{
public:
  int start_vertex_id;
  int end_vertex_id;
  int edge_start_idx;
  int edge_end_idx;
  CSR::Vertex *vertices;
  CSR::Edge *edges;

  __device__
  CSRPartition () 
  {

  }

  __device__
  void initialize (int _start, int _end, int _edge_start_idx, int _edge_end_idx, CSR::Vertex* _vertices, CSR::Edge* _edges)
  {
    start_vertex_id = _start;
    end_vertex_id = _end;
    vertices = _vertices;
    edges = _edges;
    edge_start_idx = _edge_start_idx;
    edge_end_idx = _edge_end_idx;
  }

  CSRPartition (int _start, int _end, int _edge_start_idx, int _edge_end_idx, CSR::Vertex* _vertices, CSR::Edge* _edges)
  {
    start_vertex_id = _start;
    end_vertex_id = _end;
    vertices = _vertices;
    edges = _edges;
    edge_start_idx = _edge_start_idx;
    edge_end_idx = _edge_end_idx;
  }

  __host__ __device__
  int get_start_edge_idx (int vertex_id) {
    if (!(vertex_id <= end_vertex_id && start_vertex_id <= vertex_id)) {
      printf ("vertex_id %d, end_vertex %d, start_vertex %d\n", vertex_id, end_vertex_id, start_vertex_id);
      assert (false);
    }
    return vertices[vertex_id - start_vertex_id].start_edge_id;
  }

  __host__ __device__
  int get_end_edge_idx (int vertex_id)
  {
    assert (vertex_id <= end_vertex_id && start_vertex_id <= vertex_id);
    return vertices[vertex_id - start_vertex_id].end_edge_id;
  }
  
  __host__ __device__
  CSR::Edge get_edge (int idx) 
  {
    assert (idx >= edge_start_idx && idx <= edge_end_idx);
    return edges[idx - edge_start_idx];
  }
};

#ifdef USE_CONSTANT_MEM
  __constant__ unsigned char csr_constant_buff[sizeof(CSR)];
#endif

void csr_from_graph (CSR* csr, Graph& graph)
{
  int edge_iterator = 0;
  auto graph_vertices = graph.get_vertices ();
  for (int i = 0; i < graph_vertices.size (); i++) {
    ::Vertex& vertex = graph_vertices[i];
    csr->vertices[i].set_from_graph_vertex (graph_vertices[i]);
    csr->vertices[i].set_start_edge_id (edge_iterator);
    for (auto edge : vertex.get_edges ()) {
      csr->edges[edge_iterator] = edge;
      edge_iterator++;
    }

    csr->vertices[i].set_end_edge_id (edge_iterator-1);
  }
}

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
  // __host__ __device__
  // VectorVertexEmbedding (const VectorVertexEmbedding& embedding)
  // {
  // #if DEBUG
  //   assert (embedding.get_max_size () <= get_max_size ());
  // #endif
  //   filled_size = 0;
  //   for (int i = 0; i < embedding.get_n_vertices (); i++) {
  //     add (embedding.get_vertex (i));
  //   }
  // }
  
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

bool is_cuda_error (cudaError_t error) 
{
  //cudaError_t error = cudaGetLastError ();
  if (error != cudaSuccess) {
    const char* error_string = cudaGetErrorString (error);
    std::cout << "Cuda Error: " << error_string << std::endl;
    return true;
  }

  return false;
}

#define EXECUTE_CUDA_FUNC(x) assert (is_cuda_error (x) == false);

__global__ void get_max_lengths_for_embeddings_first_iter (void* void_csr, void* input, size_t n_embeddings,
                                                          void* void_embedding_storage,
                                                          int global_mem_start_idx,
                                                          unsigned long long int* embeddings_additions_iter,
                                                          void* void_map_orig_embedding_to_additions)
{
  CSR* csr = (CSR*)void_csr;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= n_embeddings) {
    return;
  }

  VectorVertexEmbedding* input_embeddings = (VectorVertexEmbedding*) input;
  VectorVertexEmbedding* input_embedding = &input_embeddings[thread_idx];
  //VertexID* embedding_storage = (VertexID embedding_storage;
  int* map_orig_embedding_to_additions = (int*) void_map_orig_embedding_to_additions;
  unsigned long long int new_edges = 0;
  // printf ("thread idx %d array_start_idx %ld\n", thread_idx, input_embedding->get_array_start_idx ());
  /*Perform a single hop for all vertices in the input embedding*/
  for (int vertex_idx = 0; vertex_idx < input_embedding->get_n_vertices (); vertex_idx++) {
    VertexID vertex = input_embedding->get_vertex (vertex_idx, void_embedding_storage, global_mem_start_idx);
    int start_edge_idx = csr->get_start_edge_idx (vertex);
    const int end_edge_idx = csr->get_end_edge_idx (vertex);

    if (start_edge_idx != -1) {
      int e = (end_edge_idx - start_edge_idx) + 1;
      assert (e >= 0);
      new_edges += e;
    }

    assert (thread_idx == vertex);
  }

  unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
  map_orig_embedding_to_additions[2*thread_idx] = additions_start_iter;
  map_orig_embedding_to_additions[2*thread_idx+1] = new_edges;
  //printf ("thread_idx %d additions %d\n", thread_idx, map_orig_embedding_to_additions[2*thread_idx+1]);
}

__global__ void get_max_lengths_for_embeddings_single_step (void* void_csr, void* input, 
                                                            size_t n_embeddings, 
                                                            void* void_embedding_storage,
                                                            int global_mem_start_idx,  
                                                            unsigned long long int* void_embeddings_additions_iter,
                                                            void* void_map_orig_embedding_to_additions_prev_iter,
                                                            void* void_map_orig_embedding_to_additions_next_iter,
                                                            void* void_map_orig_embedding_to_additions_first_iter)
{
  CSR* csr = (CSR*)void_csr;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= n_embeddings) {
    return;
  }

  VectorVertexEmbedding* input_embeddings = (VectorVertexEmbedding*) input;
  VectorVertexEmbedding* input_embedding = &input_embeddings[thread_idx];
  //VertexID* embedding_storage = (VertexID embedding_storage;
  unsigned long long int* embeddings_additions_iter = void_embeddings_additions_iter;
  int* map_orig_embedding_to_additions_next_iter = (int*)void_map_orig_embedding_to_additions_next_iter;
  int* map_orig_embedding_to_additions_prev_iter = (int*)void_map_orig_embedding_to_additions_prev_iter;
  int* map_orig_embedding_to_additions_first_iter = (int*) void_map_orig_embedding_to_additions_first_iter;
  unsigned long long int new_edges = map_orig_embedding_to_additions_first_iter[2*thread_idx + 1];
  // printf ("thread idx %d array_start_idx %ld\n", thread_idx, input_embedding->get_array_start_idx ());
  /*Perform a single hop for all vertices in the input embedding*/
  for (int vertex_idx = 0; vertex_idx < input_embedding->get_n_vertices (); vertex_idx++) {
    VertexID vertex = input_embedding->get_vertex (vertex_idx, void_embedding_storage, global_mem_start_idx);
    int start_edge_idx = csr->get_start_edge_idx (vertex);
    const int end_edge_idx = csr->get_end_edge_idx (vertex);
    if (start_edge_idx != -1) {
      while (start_edge_idx <= end_edge_idx) {
        int v = csr->get_edges()[start_edge_idx];
        new_edges += map_orig_embedding_to_additions_prev_iter [2*v+1];
        start_edge_idx++;
      }
    }

    assert (thread_idx == vertex);
  }

  //printf ("new_edges %ld\n", new_edges);
  unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
  map_orig_embedding_to_additions_next_iter[2*thread_idx] = additions_start_iter;
  map_orig_embedding_to_additions_next_iter[2*thread_idx+1] = new_edges;
}

__global__ void run_single_step_embedding (int N_HOPS, void* void_csr, int* partition_range, int n_partitions, void* input, size_t n_embeddings, void* void_embedding_storage, uint64_t global_mem_start_idx,
                                           void* void_embeddings_additions, 
                                           size_t embeddings_additions_size,
                                           void* void_map_orig_embedding_to_additions, 
                                           size_t map_orig_embedding_to_additions_size,
                                           int* additions_sizes)
{
  CSR* csr = (CSR*)void_csr;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
#ifdef ENABLE_GRAPH_PARTITION_IN_SHARED_MEM
  __shared__ CSRPartition csr_partition;
  __shared__ char partition_vertex_edges [GRAPH_PARTITION_SIZE - sizeof (CSRPartition)];
  int partition_idx = n_partitions - 1;  

  for (int i = 0; i < n_partitions; i++) {
    if (thread_idx >= partition_range[2*i] && thread_idx <= partition_range[2*i + 1]) {
      partition_idx = i;
      break;
    }
  }
  int partition_start_vertex = partition_range[2*partition_idx];
  int partition_end_vertex = partition_range[2*partition_idx + 1];
  int partition_n_vertices = partition_end_vertex - partition_start_vertex;
  CSR::Vertex* vertex_array = (CSR::Vertex*)&partition_vertex_edges[0];
  int vertex_array_size = (partition_end_vertex - partition_start_vertex + 1)*sizeof(CSR::Vertex);
  CSR::Edge* edge_array = (CSR::Edge*)&partition_vertex_edges[vertex_array_size];
  
  int end_edge = csr->get_end_edge_idx (partition_end_vertex);
  if (end_edge == -1)
    end_edge = csr->get_start_edge_idx (partition_end_vertex);
  int start_edge = csr->get_start_edge_idx (partition_start_vertex);
  int partition_n_edges = end_edge - start_edge + 1;
  //if (!(sizeof (partition_vertex_edges) >= vertex_array_size + partition_n_edges*sizeof (CSR::Edge))) 
   // printf ("sizeof (partition_vertex_edges) %d vertex_array_size %d partition_n_edges*sizeof (CSR::Edge) %d \n", (int)sizeof (partition_vertex_edges), vertex_array_size, (int) partition_n_edges*sizeof (CSR::Edge));
  assert (sizeof (partition_vertex_edges) >= vertex_array_size + partition_n_edges*sizeof (CSR::Edge));
  csr_partition.initialize (partition_start_vertex, partition_end_vertex, start_edge, end_edge, vertex_array, edge_array);
  for (int i = 0; i < partition_n_vertices; i+=blockDim.x) {
    if (i + threadIdx.x <= partition_n_vertices) {
      vertex_array[i + threadIdx.x] = csr->get_vertices () [partition_start_vertex + i + threadIdx.x];
    }
  }

  for (int i = 0; i < partition_n_edges; i+=blockDim.x) {
    if (i + threadIdx.x <= partition_n_edges) {
      edge_array[i + threadIdx.x] = csr->get_edges () [start_edge + i + threadIdx.x];
    }
  }
  
  __syncthreads ();
#endif

  if (thread_idx >= n_embeddings) {
    return;
  }

  VectorVertexEmbedding* input_embeddings = (VectorVertexEmbedding*) input;
  VectorVertexEmbedding* input_embedding = &input_embeddings[thread_idx];
  VertexID* embeddings_additions = (VertexID*)void_embeddings_additions;

  int* map_orig_embedding_to_additions = (int*) void_map_orig_embedding_to_additions;

  unsigned long long int new_edges = 0;
  // printf ("thread idx %d array_start_idx %ld\n", thread_idx, input_embedding->get_array_start_idx ());
  /*Perform a single hop for all vertices in the input embedding*/
  int additions_filled = map_orig_embedding_to_additions[2*thread_idx];
  int start = map_orig_embedding_to_additions[2*thread_idx];
  int size = map_orig_embedding_to_additions[2*thread_idx+1];

  for (int vertex_idx = 0; vertex_idx < input_embedding->get_n_vertices (); vertex_idx++) {
    VertexID vertex = input_embedding->get_vertex (vertex_idx, void_embedding_storage, global_mem_start_idx);
  #ifdef ENABLE_GRAPH_PARTITION_IN_SHARED_MEM
    int start_edge_idx = csr_partition.get_start_edge_idx (vertex);
    const int end_edge_idx = csr_partition.get_end_edge_idx (vertex);
  #else
    int start_edge_idx = csr->get_start_edge_idx (vertex);
    const int end_edge_idx = csr->get_end_edge_idx (vertex);
  #endif
    if (start_edge_idx != -1) {
      while (start_edge_idx <= end_edge_idx) {
        VertexID edge = csr->get_edges ()[start_edge_idx];
        embeddings_additions[additions_filled++] = edge;
        start_edge_idx++;
      }
    }
  }

  int additions_end_idx = additions_filled;
  int additions_start_idx = start;
  int hop = 1;

  while (hop < N_HOPS) {    
    for (int vertex_idx = additions_start_idx; vertex_idx < additions_end_idx; vertex_idx++) {
      int vertex = embeddings_additions [vertex_idx];
#ifdef ENABLE_GRAPH_PARTITION_IN_SHARED_MEM
      int start_edge_idx = (vertex >= partition_start_vertex && vertex <= partition_end_vertex) ? csr_partition.get_start_edge_idx (vertex) : csr->get_start_edge_idx (vertex);
      const int end_edge_idx = (vertex >= partition_start_vertex && vertex <= partition_end_vertex) ? csr_partition.get_end_edge_idx (vertex) : csr->get_end_edge_idx (vertex);
#else
      int start_edge_idx = csr->get_start_edge_idx (vertex);
      const int end_edge_idx = csr->get_end_edge_idx (vertex);
#endif

      if (start_edge_idx != -1) {
        while (start_edge_idx <= end_edge_idx) {
          VertexID edge = csr->get_edges ()[start_edge_idx];
          // bool present = false;
          // for (int i = start; i < additions_filled; i++) {
          //   if (embeddings_additions[i] == edge) {
          //     present = true;
          //     break;
          //   }
          // }
          // if (present == false)
          embeddings_additions[additions_filled++] = edge;
          start_edge_idx++;
        }
      }
    }

    additions_start_idx = additions_end_idx;
    additions_end_idx = additions_filled;

    hop++;
  }

  //if (thread_idx == 0) {
    //printf ("additions_filled %d start %d\n", additions_filled, start);
  //}
  additions_sizes[thread_idx] = additions_filled - start;
}

std::vector <std::unordered_set <VertexID>> n_hop_cpu (CSR* csr, const int N_HOPS)
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

      hops[vertex].insert (vertex_hops[hop].begin (), vertex_hops[hop].end ());
      hop++;
    }

    // for (int __hop = 1; __hop < N_HOPS + 1; __hop++) {
      
    // }
  }

  return hops;
}

#define MAX_VERTICES_PER_TB 10
#if MAX_VERTICES_PER_TB < 1
  #error "MAX_VERTICES_PER_TB should be greater than or equal to 1"
#endif

__global__ void run_think_hybrid_single_step_embedding (int N_HOPS, int hop, void* void_csr,
  void* void_embeddings_additions, 
  size_t embeddings_additions_size,
  int* map_orig_embedding_to_additions,
  int* previous_stage_filled_range,
  int* global_index)
{
  CSR* csr = (CSR*)void_csr;
  __shared__ int vertices[MAX_VERTICES_PER_TB];
  __shared__ int previous_step_end[MAX_VERTICES_PER_TB];
  __shared__ int n_vertex_load;

  VertexID* embeddings_additions = (VertexID*)void_embeddings_additions;

  if (hop != 0) {
    if (*global_index >= gridDim.x) {
      return;
    }

    if (threadIdx.x == 0) {
#if (MAX_VERTICES_PER_TB == 1)
      n_vertex_load = 0;
      vertices[n_vertex_load++] = atomicAdd(global_index, 1);
#else
      int load = 0;
      n_vertex_load = 0;

      while (n_vertex_load < MAX_VERTICES_PER_TB && load < blockDim.x) {
        vertices[n_vertex_load] = atomicAdd(global_index, 1);
        if (vertices[n_vertex_load] >= gridDim.x) {
          break;
        }
        int hops_so_far = previous_stage_filled_range[2*vertices[n_vertex_load] + 1] - previous_stage_filled_range[2*vertices[n_vertex_load]];
        load += hops_so_far;
        n_vertex_load++;
      }
#endif
    }
    assert (n_vertex_load <= MAX_VERTICES_PER_TB);
    __syncthreads ();

    for (int curr_vertex_id = 0; curr_vertex_id < n_vertex_load; curr_vertex_id++) {
      int vertex = vertices[curr_vertex_id];
      int start = map_orig_embedding_to_additions[2*vertex];
      if (threadIdx.x == 0) {
        previous_step_end[curr_vertex_id] = previous_stage_filled_range[2*vertex+1];
      }

      __syncthreads ();
      int previous_step_start = previous_stage_filled_range[2*vertex];
      int hops_so_far = previous_step_end [curr_vertex_id] - previous_step_start;
      int* end = &previous_stage_filled_range[2*vertex + 1];
    
      for (int i = 0; i < hops_so_far/blockDim.x + 1; i++) {
        int hop_idx = i*blockDim.x + threadIdx.x;
        //printf ("Vertex[0] %d hops_so_far %d threadIdx.x %d hop_idx %d\n", vertex[0], hops_so_far, threadIdx.x, hop_idx);

        if (hop_idx >= hops_so_far) {
          break;
        }

        int hop_vertex = embeddings_additions[start + previous_step_start + hop_idx];
        int start_edge_idx = csr->get_start_edge_idx (hop_vertex);
        const int end_edge_idx = csr->get_end_edge_idx (hop_vertex);

        if (end_edge_idx != -1) {
          while (start_edge_idx <= end_edge_idx) {
            VertexID edge = csr->get_edges ()[start_edge_idx];
            int e = atomicAdd (end, 1);
            embeddings_additions[start + e] = edge;
            start_edge_idx++;
          }
        }
      }
      
      if (threadIdx.x == 0) {
        previous_stage_filled_range[2*vertex] = previous_step_end[curr_vertex_id];
      }
    }
  } else {
    int source_vertex = blockIdx.x;

    int start = map_orig_embedding_to_additions[2*source_vertex];
    int start_edge_idx = csr->get_start_edge_idx (source_vertex);
    const int end_edge_idx = csr->get_end_edge_idx (source_vertex);
    const int n_edges = end_edge_idx - start_edge_idx + 1;

    if (end_edge_idx == -1) {
      return;
    }

    int* end = &previous_stage_filled_range[2*source_vertex + 1];

    for (int i = 0; i < n_edges/blockDim.x + 1; i++) {
      int edge_idx = i*blockDim.x + threadIdx.x;
      if (edge_idx >= n_edges) 
        return;
      VertexID edge = csr->get_edges ()[start_edge_idx + edge_idx];
      int e = atomicAdd (end, 1);
      embeddings_additions[start + e] = edge;    
    }

    previous_stage_filled_range[2*source_vertex] = 0;
  }
}

__global__ void run_think_like_an_edge_single_step_embedding (int N_HOPS, int hop, void* void_csr,
  void* void_embeddings_additions, 
  size_t embeddings_additions_size,
  int* map_orig_embedding_to_additions,
  int* previous_stage_filled_range,
  size_t n_edges,
  int* prev_thread_idx_to_edge_in_additions,
  int* thread_idx_to_edge_in_additions,
  int* thread_idx_to_edge_in_additions_size)
{
  CSR* csr = (CSR*)void_csr;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx >= n_edges)
  return;

  VertexID* embeddings_additions = (VertexID*)void_embeddings_additions;

  if (hop != 0) {
    int edge_idx = prev_thread_idx_to_edge_in_additions [2*thread_idx];
    int source_vertex = prev_thread_idx_to_edge_in_additions [2*thread_idx + 1];

    int start = map_orig_embedding_to_additions[2*source_vertex];
    int* end = &previous_stage_filled_range[source_vertex];
  
    int vertex = embeddings_additions[edge_idx];

    int start_edge_idx = csr->get_start_edge_idx (vertex);
    const int end_edge_idx = csr->get_end_edge_idx (vertex);

    if (end_edge_idx != -1) {
      while (start_edge_idx <= end_edge_idx) {
        VertexID edge = csr->get_edges ()[start_edge_idx];
        int e = atomicAdd (end, 1);
        embeddings_additions[start + e] = edge;
        if (hop < N_HOPS) {
          int q = atomicAdd (thread_idx_to_edge_in_additions_size, 2);
          thread_idx_to_edge_in_additions [q] = start + e;
          thread_idx_to_edge_in_additions [q + 1] = source_vertex;
        }

        start_edge_idx++;
      }
    }
  } else {
    int source_vertex = thread_idx;

    int start = map_orig_embedding_to_additions[2*thread_idx];
    int prev_end = 0;
    int* end = &previous_stage_filled_range[source_vertex];

    int vertex = thread_idx;

    int start_edge_idx = csr->get_start_edge_idx (vertex);
    const int end_edge_idx = csr->get_end_edge_idx (vertex);

    if (start_edge_idx != -1) {
      while (start_edge_idx <= end_edge_idx) {
        VertexID edge = csr->get_edges ()[start_edge_idx];
        int e = atomicAdd (end, 1);
        embeddings_additions[start + e] = edge;
        if (hop < N_HOPS) {
          int q = atomicAdd (thread_idx_to_edge_in_additions_size, 2);
          thread_idx_to_edge_in_additions [q] = start + e;
          thread_idx_to_edge_in_additions [q + 1] = vertex;
        }

        start_edge_idx++;
      }
    }
  }
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

  while (true) {
    char line[LINE_SIZE];
    char num_str[LINE_SIZE];
    size_t line_size;

    if (fgets (line, LINE_SIZE, fp) == nullptr) {
      break;
    }

    int id, label;
    int bytes_read;

    bytes_read = sscanf (line, "%d %d", &id, &label);
    Vertex vertex (id, label);
    char* _line = line + chars_in_int (id) + chars_in_int (label);
    do {
      int num;

      bytes_read = sscanf (_line, "%d", &num);
      if (bytes_read > 0) {
        vertex.add_edge (num);
        _line += chars_in_int (num);
        n_edges++;
      }

    } while (bytes_read > 0);

    vertex.sort_edges ();

    vertices.push_back (vertex);
  }

  fclose (fp);

  std::cout << "n_edges "<<n_edges <<std::endl;
  std::cout << "vertices " << vertices.size () << std::endl; 

  Graph graph (vertices, n_edges);

  CSR* csr = new CSR(N, N_EDGES);
  std::cout << "sizeof(CSR)"<< sizeof(CSR)<<std::endl;
  csr_from_graph (csr, graph);
  
#ifdef USE_CONSTANT_MEM
  cudaMemcpyToSymbol (csr_constant_buff, csr, sizeof(CSR));
  //~ CSR* csr_constant = (CSR*) &csr_constant_buff[0];
  //~ csr_constant->n_vertices = csr->get_n_vertices ();
  //~ printf ("csr->get_n_vertices () = %d\n", csr->get_n_vertices ());
  //~ csr_constant->n_edges = csr->get_n_edges ();
  //~ csr_constant->copy_vertices (csr, 0, csr->get_n_vertices ());
  //~ csr_constant->copy_edges (csr, 0, csr->get_n_edges ());
#endif
  int N_THREADS = 512;
  size_t global_mem_size = 15*1024*1024*1024UL;
  #define PINNED_MEMORY
  #ifdef PINNED_MEMORY
    char* global_mem_ptr;
    cudaError_t malloc_error = cudaMallocHost ((void**)&global_mem_ptr, global_mem_size);
    assert (malloc_error == cudaSuccess);
  #else
    char* global_mem_ptr = new char[global_mem_size];
  #endif

  std::cout << "Pinned Memory Allocated" << std::endl;
  GlobalMemAllocator::initialize (global_mem_ptr, global_mem_size);

  std::vector<VectorVertexEmbedding> initial_embeddings = get_initial_embedding_vector (csr);
  std::vector<VectorVertexEmbedding> output;
  size_t new_embeddings_size = 0;
  int iter = 0;
  std::vector<VectorVertexEmbedding>& input_embeddings = initial_embeddings;
  std::vector<VectorVertexEmbedding> iter_1_embeddings;
  {
    run_single_step_initial_vector (input_embeddings, csr, output, iter_1_embeddings);
    input_embeddings = iter_1_embeddings;
  }

  iter = 0;
  double total_stream_time = 0;

  const size_t max_embedding_size_per_iter = (12000000/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
  double_t kernelTotalTime = 0.0;
  std::vector <VectorVertexEmbedding> produced_embeddings;

  void* device_map_orig_embedding_to_additions_prev = nullptr; //Previous iterations map
  int* final_map_orig_embedding_to_additions;

  std::vector<std::pair<int, int>> vertex_partition_positions_vector;
  
#if 0
  //Create Partitions.
  int u = 0;
  while (u < csr->get_n_vertices ()) {
    int n_edges = 0;
    int u_start = u;
    int u_end = csr->get_n_vertices () - 1;
    for (int v = u; v < csr->get_n_vertices (); v++) {
      const int start = csr->get_start_edge_idx (v);
      const int end = csr->get_end_edge_idx (v);
      if (end != -1) {
        n_edges += end - start + 1;
      }      
      if (n_edges * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex) >= GRAPH_PARTITION_SIZE - sizeof (CSRPartition) and (v-u_start + 1) > N_THREADS) {
        std::cout << " v " << v << " n_edges " << n_edges << " u " << u_start  << "  sizeof (CSR::Edge) " << sizeof (CSR::Edge) <<  " sizeof(CSR::Vertex) " << sizeof(CSR::Vertex) << std::endl;
        u = v;
        u_end = v;
        int n_edges_prev = n_edges;
        while (!((u_end - u_start + 1)%N_THREADS == 0 && (n_edges * sizeof (CSR::Edge) + (u_end-u_start + 1)*sizeof(CSR::Vertex) <= GRAPH_PARTITION_SIZE - sizeof (CSRPartition)))) {
          const int start = csr->get_start_edge_idx (u_end);
          const int end = csr->get_end_edge_idx (u_end);
          if (end != -1) {
            int q = (end - start + 1);
            //std::cout << "u_end " << u_end << " q " << q << " n-us " << u_end - u_start + 1 << " sum " << (n_edges * sizeof (CSR::Edge) + (u_end-u_start + 1)*sizeof(CSR::Vertex)) << " " << GRAPH_PARTITION_SIZE - sizeof (CSRPartition) << std::endl;
            n_edges -= q;
          }
          u_end--;
        }

        if (u_end < u_start) {
          std::cout << "u_end : " << u_end << " u_start: "  << u_start  << " n_edges " << n_edges_prev << std::endl;
          std::cout << "ERROR: Cannot create partition " << std::endl;
          assert (false);
        }

        u = u_end + 1;
        break;
      }
    }

    std::cout << "Creating partition start: " << u_start << " end: " << u_end << " n_edges " << n_edges << std::endl; 
    vertex_partition_positions_vector.push_back (std::make_pair (u_start, u_end));

    if (u_end == csr->get_n_vertices () - 1) {
      break;
    }
    //std::cout << "u " << u <<  std::endl;
  }

  int* vertex_partition_positions = new int[vertex_partition_positions_vector.size () * sizeof (int) * 2];
  int n_partitions = vertex_partition_positions_vector.size ();

  for (int i = 0; i < vertex_partition_positions_vector.size (); i++) {
    std::pair <int, int> p = vertex_partition_positions_vector[i];
    vertex_partition_positions[2*i] = p.first;
    vertex_partition_positions[2*i+1] = p.second;
  }

  std::cout << "Partitions: " << std::endl;
  for (auto p : vertex_partition_positions_vector) {
    std::cout << p.first << " " << p.second << std::endl;
  }

  //Check if CSRPartition is correct
  for (auto p : vertex_partition_positions_vector) {
    CSR::Vertex* vertex_array = new CSR::Vertex[p.second - p.first + 1];
    memcpy (vertex_array, &csr->get_vertices ()[p.first], (p.second-p.first + 1)*sizeof(CSR::Vertex));
    int end_edge = csr->get_end_edge_idx (p.second);
    if (end_edge == -1)
      end_edge = csr->get_start_edge_idx (p.second);
    int start_edge = csr->get_start_edge_idx (p.first);
    std::cout << "P " << end_edge << "  " << start_edge << std::endl;
    CSR::Edge* edge_array = new CSR::Edge[end_edge - start_edge + 1];
    memcpy (edge_array, &csr->get_edges ()[start_edge], (end_edge - start_edge + 1)*sizeof (CSR::Edge));
    std::cout << "E " << sizeof (int)*(end_edge - start_edge + 1) << " V " << (p.second - p.first + 1)*sizeof (CSR::Vertex) << std::endl;
    CSRPartition part = CSRPartition (p.first, p.second, start_edge, end_edge, vertex_array, edge_array);
    for (int v = p.first; v <= p.second; v++) {
      assert (part.get_start_edge_idx (v) == csr->get_start_edge_idx (v));
      assert (part.get_end_edge_idx (v) == csr->get_end_edge_idx (v));
      int start = part.get_start_edge_idx (v);
      int end = part.get_end_edge_idx (v);
      if (start != -1 && end != 1) {
        while (start <= end) {
          assert (part.get_edge (start) == csr->get_edges()[start]);
          start++;
        }
      }
    }
  }
#endif 

  /*Code for preparing additions kernels*/
  uint64_t vertices_in_embedding = input_embeddings[0].get_n_vertices ();
  uint64_t global_mem_start_idx = input_embeddings[0].get_array_start_idx ();
  uint64_t global_mem_end_idx = input_embeddings[input_embeddings.size () - 1].get_array_start_idx () + input_embeddings[input_embeddings.size () - 1].get_n_vertices ()*sizeof (VertexID);

  const int N_HOPS = 2;
  // std::cout << "-2   " << input_embeddings[input_embeddings.size () - 2].get_array_start_idx () + input_embeddings[input_embeddings.size () - 2].get_n_vertices ()*sizeof (VertexID) << std::endl;
  std::cout << "Number of input embeddings " << input_embeddings.size() << std::endl;
  std::cout << "global_mem_start_idx " << global_mem_start_idx << " global_mem_end_idx " << global_mem_end_idx << " allocated " << GlobalMemAllocator::allocated () << std::endl;
  std::cout << "vertices_in_embedding " << vertices_in_embedding << std::endl;
  assert (global_mem_end_idx == GlobalMemAllocator::allocated ());
  void* device_csr; //Graph on GPU
  int* device_map_orig_embedding_to_additions; //Map of idx of embedding to the start of how many inputs are added and number of new embeddings
  void* device_map_orig_embedding_to_additions_first;
  void* device_input_embeddings_storage; //Input embeddings copied to GPU from CPU
  void* device_input_embeddings;
  int* device_vertex_partition_positions;
  unsigned long long* device_embeddings_addition_iter;
  size_t map_orig_embedding_to_additions_size = input_embeddings.size () * sizeof (VertexID) * 2;

  // std::cout << "Preparing iteration " << iter << std::endl;
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_csr, sizeof(CSR)));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_csr, csr, sizeof(CSR), cudaMemcpyHostToDevice));
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_input_embeddings_storage, global_mem_end_idx - global_mem_start_idx));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_input_embeddings_storage, (char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx, 
                                  global_mem_end_idx - global_mem_start_idx, cudaMemcpyHostToDevice));
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_input_embeddings,input_embeddings.size()*sizeof(VectorVertexEmbedding)));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_input_embeddings ,&input_embeddings[0], input_embeddings.size()*sizeof(VectorVertexEmbedding), cudaMemcpyHostToDevice));
#if 0
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_vertex_partition_positions, n_partitions*sizeof(int)*2));
  EXECUTE_CUDA_FUNC (cudaMemcpy (device_vertex_partition_positions, vertex_partition_positions, n_partitions*sizeof(int)*2, cudaMemcpyHostToDevice));
#endif
  unsigned long long embeddings_addition_iter = 0;

  double gpu_time = 0;

  for (iter; iter < N_HOPS; iter++) {
    embeddings_addition_iter = 0;
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_embeddings_addition_iter, sizeof(unsigned long long)));
    EXECUTE_CUDA_FUNC (cudaMemcpy (device_embeddings_addition_iter, &embeddings_addition_iter,  sizeof (unsigned long long), cudaMemcpyHostToDevice));

    EXECUTE_CUDA_FUNC (cudaMalloc (&device_map_orig_embedding_to_additions, map_orig_embedding_to_additions_size));
    if (false) {
      VectorVertexEmbedding* __m = (VectorVertexEmbedding*)malloc (input_embeddings.size()*sizeof(VectorVertexEmbedding));
      cudaMemcpy (__m, device_input_embeddings, input_embeddings.size()*sizeof(VectorVertexEmbedding), cudaMemcpyDeviceToHost);
      assert ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx + __m[0].get_array_start_idx () == __m[0].get_array());
      std::cout << "sizeof (VectorVertexEmbedding) " << sizeof(VectorVertexEmbedding) << std::endl;
      for (int i = 0; i < input_embeddings.size (); i++) {
        std::cout << "s " << (void*)((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx + __m[i].get_array_start_idx ()) << " d " <<  __m[i].get_array() << std::endl;
        
        assert ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx + __m[i].get_array_start_idx () == __m[i].get_array());

        std::cout << "i "<< i << " v : " << __m[i].get_vertex (0, (char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx) << " sstart id: " <<  __m[i].get_array_start_idx () << std::endl;
      }

      break;
    }

    std::cout << "Calling cuda kernel for iteration " << iter << std::endl;

    int N_THREADS = 128;
    int N_BLOCKS = (input_embeddings.size()%128 == 0) ? input_embeddings.size()/128 : input_embeddings.size()/128 + 1;
    
    double t1 = convertTimeValToDouble(getTimeOfDay ());
    if (iter == 0) {
      get_max_lengths_for_embeddings_first_iter <<<N_BLOCKS, N_THREADS>>> (device_csr,
                                                    device_input_embeddings, input_embeddings.size(), device_input_embeddings_storage,
                                                    global_mem_start_idx,
                                                    device_embeddings_addition_iter,
                                                    device_map_orig_embedding_to_additions);
    } else {
      get_max_lengths_for_embeddings_single_step <<<N_BLOCKS, N_THREADS>>> (device_csr,
                                                  device_input_embeddings, 
                                                  input_embeddings.size (), 
                                                  device_input_embeddings_storage, 
                                                  global_mem_start_idx,
                                                  device_embeddings_addition_iter,
                                                  device_map_orig_embedding_to_additions_prev,
                                                  device_map_orig_embedding_to_additions,
                                                  device_map_orig_embedding_to_additions_first);
    }
    EXECUTE_CUDA_FUNC (cudaDeviceSynchronize ());
    double t2 = convertTimeValToDouble(getTimeOfDay ());

    gpu_time += t2 - t1;
    if (device_map_orig_embedding_to_additions_prev != nullptr) {
      cudaFree (device_map_orig_embedding_to_additions_prev);
    }

    EXECUTE_CUDA_FUNC (cudaDeviceSynchronize ());
    std::cout << "Cuda Kernel Done " << std::endl;
    is_cuda_error (cudaGetLastError ());    
    EXECUTE_CUDA_FUNC (cudaMemcpy (&embeddings_addition_iter, device_embeddings_addition_iter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::cout << "Embedding Additions " << embeddings_addition_iter << std::endl;

    if (iter == N_HOPS - 1) {
      //In Last iteration perform some cleanup and record the map_orig_embedding_to_additions_size
      final_map_orig_embedding_to_additions = (int*)new char[map_orig_embedding_to_additions_size];
      EXECUTE_CUDA_FUNC (cudaMemcpy (final_map_orig_embedding_to_additions, device_map_orig_embedding_to_additions, map_orig_embedding_to_additions_size, cudaMemcpyDeviceToHost));
      //cudaFree (device_map_orig_embedding_to_additions);
      device_map_orig_embedding_to_additions_prev = nullptr;
    } else {
      device_map_orig_embedding_to_additions_prev = device_map_orig_embedding_to_additions;
    }
    if (iter == 0) {
      device_map_orig_embedding_to_additions_first = device_map_orig_embedding_to_additions;
    }

    cudaFree (device_embeddings_addition_iter);

    //Create new embeddings from the received additions
    // for (int input_embedding_idx = 0; input_embedding_idx < input_embeddings.size (); input_embedding_idx++) {
    //   VectorVertexEmbedding& input_embedding = input_embeddings[input_embedding_idx];
    //   int n_additions = map_orig_embedding_to_additions[2*input_embedding_idx+1];
    //   int start_idx = map_orig_embedding_to_additions[2*input_embedding_idx];
    //   size_t produced_embedding_size = input_embedding.get_n_vertices () + n_additions;
      
    //   size_t global_mem_idx = GlobalMemAllocator::alloc_vertices_array (produced_embedding_size);
    //   // std::cout << "i " << input_embedding_idx << " produced_embedding_size " << produced_embedding_size << " global_mem_idx " << global_mem_idx << std::endl;
    //   VertexID* ptr = (VertexID*) ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_idx);
    //   ((VertexID*)ptr)[0] = ((VertexID*)input_embedding.get_array ())[0];
    //   memcpy (ptr, input_embedding.get_array (), sizeof(VertexID)*input_embedding.get_n_vertices ());
    //   memcpy (ptr + input_embedding.get_n_vertices (), &embedding_additions[start_idx], sizeof(VertexID)*n_additions);
      
    //   produced_embeddings.push_back (VectorVertexEmbedding ((uint32_t)produced_embedding_size, global_mem_idx, true));
    // }

    //input_embeddings = produced_embeddings;
  }
  
  std::cout << "Generating additions" << std::endl;
  int* device_additions_sizes;
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_additions_sizes, sizeof(VertexID)*input_embeddings.size ()*2));
  EXECUTE_CUDA_FUNC (cudaMemset (device_additions_sizes, 0, sizeof(VertexID)*input_embeddings.size ()*2));
  void* device_embeddings_additions; //Storage to store inputs added to each embedding
  size_t embeddings_additions_size = (embeddings_addition_iter+1)*sizeof(VertexID);
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_embeddings_additions, embeddings_additions_size));
  //Now generate all the next hop neighbours
  
  // run_single_step_embedding<<<N_BLOCKS, N_THREADS>>> (N_HOPS, device_csr, device_vertex_partition_positions, n_partitions, device_input_embeddings, input_embeddings.size (), device_input_embeddings_storage, global_mem_start_idx,
  // device_embeddings_additions, 
  //   embeddings_additions_size, 
  //   device_map_orig_embedding_to_additions, 
  //   map_orig_embedding_to_additions_size,
  //   device_additions_sizes);
  int* device_filled_ranges;
  EXECUTE_CUDA_FUNC (cudaMalloc (&device_filled_ranges, sizeof (int)*input_embeddings.size ()));
  n_edges = input_embeddings.size ();
  int* device_prev_thread_idx_to_edge_in_additions = nullptr;

  for (int hop = 0; hop < N_HOPS; hop++) {
    int* device_thread_idx_to_edge_in_additions;
    int* device_thread_idx_to_edge_in_additions_size;
    int* global_index;

    EXECUTE_CUDA_FUNC (cudaMalloc (&global_index, sizeof(int)));
    EXECUTE_CUDA_FUNC (cudaMemset (global_index, 0,  sizeof (int)));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_thread_idx_to_edge_in_additions, embeddings_additions_size*2));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_thread_idx_to_edge_in_additions_size, sizeof (int)));
    EXECUTE_CUDA_FUNC (cudaMemset (device_thread_idx_to_edge_in_additions_size, 0,  sizeof (int)));
    EXECUTE_CUDA_FUNC (cudaMemset (global_index, 0,  sizeof (int)));
    int N_BLOCKS = input_embeddings.size ();

    double t1 = convertTimeValToDouble(getTimeOfDay ());
    run_think_hybrid_single_step_embedding <<<N_BLOCKS, N_THREADS>>> (N_HOPS, hop, device_csr,  
      device_embeddings_additions,
      embeddings_additions_size,
      device_map_orig_embedding_to_additions,
      device_additions_sizes,
      global_index);
    EXECUTE_CUDA_FUNC (cudaDeviceSynchronize ());
    double t2 = convertTimeValToDouble(getTimeOfDay ());
    gpu_time += t2 - t1;

    EXECUTE_CUDA_FUNC (cudaMemcpy (&n_edges, device_thread_idx_to_edge_in_additions_size, sizeof (int), cudaMemcpyDeviceToHost));
    n_edges = n_edges/2;
    device_prev_thread_idx_to_edge_in_additions = device_thread_idx_to_edge_in_additions;
  }
  

  VertexID* embedding_additions = new VertexID[embeddings_additions_size];
  EXECUTE_CUDA_FUNC (cudaMemcpy (embedding_additions, device_embeddings_additions, embeddings_additions_size, cudaMemcpyDeviceToHost));
  int* additions_sizes = new int[input_embeddings.size ()*2];
  EXECUTE_CUDA_FUNC (cudaMemcpy (additions_sizes, device_additions_sizes, input_embeddings.size ()*sizeof(int)*2, cudaMemcpyDeviceToHost));
  
  for (int input_embedding_idx = 0; input_embedding_idx < input_embeddings.size (); input_embedding_idx++) {
      VectorVertexEmbedding& input_embedding = input_embeddings[input_embedding_idx];
      int n_additions = additions_sizes[2*input_embedding_idx + 1];
      int start_idx = final_map_orig_embedding_to_additions[2*input_embedding_idx];
      size_t produced_embedding_size = n_additions;
      if (input_embedding_idx == 48) {
        std::cout << "n_additions " << n_additions << std::endl;
      }
      size_t global_mem_idx = GlobalMemAllocator::alloc_vertices_array (produced_embedding_size);
      //std::cout << "i " << input_embedding_idx << " produced_embedding_size " << produced_embedding_size << " global_mem_idx " << global_mem_idx << std::endl;
      VertexID* ptr = (VertexID*) ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_idx);
      ((VertexID*)ptr)[0] = ((VertexID*)input_embedding.get_array ())[0];
      // memcpy (ptr, input_embedding.get_array (), sizeof(VertexID)*input_embedding.get_n_vertices ());
      memcpy (ptr, &embedding_additions[start_idx], sizeof(VertexID)*n_additions);
      
      VectorVertexEmbedding embedding = VectorVertexEmbedding ((uint32_t)produced_embedding_size, global_mem_idx, true);
      produced_embeddings.push_back (embedding);
      // embedding.print ();
  }

  cudaFree (device_csr);
  cudaFree (device_input_embeddings_storage); 
  cudaFree (device_input_embeddings);
  cudaFree (device_embeddings_additions);

  std::cout << "Generating CPU Embeddings:" << std::endl;
  double cpu_t1 = convertTimeValToDouble (getTimeOfDay ());
  std::vector<std::unordered_set<VertexID>> hops = n_hop_cpu (csr, N_HOPS);
  double cpu_t2 = convertTimeValToDouble (getTimeOfDay ());

  std::cout << "CPU Time: " << (cpu_t2 - cpu_t1) << " secs" << std::endl;
  std::cout << "GPU Time: " << gpu_time << " secs" << std::endl;
  assert (produced_embeddings.size () == hops.size ());
  for (int idx = 0; idx < produced_embeddings.size (); idx++) {
    std::vector<VertexID> vector_hops;
    vector_hops.insert (vector_hops.begin (), hops[idx].begin (), hops[idx].end ());
    std::sort (vector_hops.begin (), vector_hops.end ());
    std::vector<VertexID> gpu_vector = produced_embeddings [idx].to_vector ();
    std::unordered_set<VertexID> gpu_vector_set = std::unordered_set<VertexID> (gpu_vector.begin (), gpu_vector.end ());
    gpu_vector = std::vector<VertexID> (gpu_vector_set.begin (), gpu_vector_set.end ());
    std::sort (gpu_vector.begin (), gpu_vector.end ());

    if (vector_hops != gpu_vector) {
      std::cout << "checking for vertex " << idx << std::endl;
      std::cout << "size " << vector_hops.size () << " " << gpu_vector.size () << std::endl;
      for (int i = 0; i < vector_hops.size (); i++) {
        std::cout << vector_hops[i] << "  " << gpu_vector[i] << std::endl;
      }
    }
    assert (vector_hops == gpu_vector);
  }

#if 0
  //Code for single kernel
  for (iter; iter < 5; iter++) {
    uint64_t vertices_in_embedding = input_embeddings[0].get_n_vertices ();
    uint64_t global_mem_start_idx = input_embeddings[0].get_array_start_idx ();
    uint64_t global_mem_end_idx = input_embeddings[input_embeddings.size () - 1].get_array_start_idx () + input_embeddings[input_embeddings.size () - 1].get_n_vertices ()*sizeof (VertexID);

    // std::cout << "-2   " << input_embeddings[input_embeddings.size () - 2].get_array_start_idx () + input_embeddings[input_embeddings.size () - 2].get_n_vertices ()*sizeof (VertexID) << std::endl;
    std::cout << "Number of input embeddings " << input_embeddings.size() << std::endl;
    std::cout << "global_mem_start_idx " << global_mem_start_idx << " global_mem_end_idx " << global_mem_end_idx << " allocated " << GlobalMemAllocator::allocated () << std::endl;
    assert (global_mem_end_idx == GlobalMemAllocator::allocated ());
    unsigned long long embeddings_addition_iter = 0;
    void* device_csr; //Graph on GPU
    void* device_embeddings_additions; //Storage to store inputs added to each embedding
    void* device_map_orig_embedding_to_additions; //Map of idx of embedding to the start of how many inputs are added and number of new embeddings
    void* device_input_embeddings_storage; //Input embeddings copied to GPU from CPU
    void* device_input_embeddings;
    unsigned long long* device_embeddings_addition_iter;

    size_t embedding_additions_size = sizeof(VertexID);
    for (int j = 0; j < iter; j++) 
      embedding_additions_size *= csr->get_n_edges ();
    size_t map_orig_embedding_to_additions_size = input_embeddings.size () * sizeof (VertexID) * 2;

    std::cout << "Preparing iteration " << iter << std::endl;
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_csr, sizeof(CSR)));
    EXECUTE_CUDA_FUNC (cudaMemcpy (device_csr, csr, sizeof(CSR), cudaMemcpyHostToDevice));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_embeddings_additions, embedding_additions_size));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_map_orig_embedding_to_additions, map_orig_embedding_to_additions_size));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_input_embeddings_storage, global_mem_end_idx - global_mem_start_idx));
    EXECUTE_CUDA_FUNC (cudaMemcpy (device_input_embeddings_storage, (char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx, 
                                   global_mem_end_idx - global_mem_start_idx, cudaMemcpyHostToDevice));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_input_embeddings,input_embeddings.size()*sizeof(VectorVertexEmbedding)));
    EXECUTE_CUDA_FUNC (cudaMemcpy (device_input_embeddings ,&input_embeddings[0], input_embeddings.size()*sizeof(VectorVertexEmbedding), cudaMemcpyHostToDevice));
    EXECUTE_CUDA_FUNC (cudaMalloc (&device_embeddings_addition_iter, sizeof(unsigned long long)));

    EXECUTE_CUDA_FUNC (cudaMemcpy (device_embeddings_addition_iter, &embeddings_addition_iter, sizeof (size_t), cudaMemcpyHostToDevice));

    if (false) {
      VectorVertexEmbedding* __m = (VectorVertexEmbedding*)malloc (input_embeddings.size()*sizeof(VectorVertexEmbedding));
      cudaMemcpy (__m, device_input_embeddings, input_embeddings.size()*sizeof(VectorVertexEmbedding), cudaMemcpyDeviceToHost);
      assert ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx + __m[0].get_array_start_idx () == __m[0].get_array());
      std::cout << "sizeof (VectorVertexEmbedding) " << sizeof(VectorVertexEmbedding) << std::endl;
      for (int i = 0; i < input_embeddings.size (); i++) {
        std::cout << "s " << (void*)((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx + __m[i].get_array_start_idx ()) << " d " <<  __m[i].get_array() << std::endl;
        
        assert ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx + __m[i].get_array_start_idx () == __m[i].get_array());

        std::cout << "i "<< i << " v : " << __m[i].get_vertex (0, (char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_start_idx) << " sstart id: " <<  __m[i].get_array_start_idx () << std::endl;
      }

      break;
    }

    std::cout << "Calling cuda kernel" << std::endl;

    int N_THREADS = 128;
    int N_BLOCKS = (input_embeddings.size()%128 == 0) ? input_embeddings.size()/128 : input_embeddings.size()/128 + 1;
    
    run_single_step_embedding <<<N_BLOCKS, N_THREADS>>> (device_csr, 
                                          device_input_embeddings, input_embeddings.size(), device_input_embeddings_storage,
                                          global_mem_start_idx, 
                                          device_embeddings_additions, embedding_additions_size, 
                                          device_map_orig_embedding_to_additions, map_orig_embedding_to_additions_size,
                                          device_embeddings_addition_iter);
    EXECUTE_CUDA_FUNC (cudaDeviceSynchronize ());
    std::cout << "Cuda Kernel Done " << std::endl;
    is_cuda_error (cudaGetLastError ());
    VertexID* embedding_additions;
    embedding_additions = (VertexID*)new char[embedding_additions_size];
    int* map_orig_embedding_to_additions;
    map_orig_embedding_to_additions = (int*)new char[map_orig_embedding_to_additions_size];
    
    EXECUTE_CUDA_FUNC (cudaMemcpy (&embeddings_addition_iter, device_embeddings_addition_iter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    std::cout << "Embedding Additions " << embeddings_addition_iter << std::endl;
    EXECUTE_CUDA_FUNC (cudaMemcpy (embedding_additions, device_embeddings_additions, embedding_additions_size, cudaMemcpyDeviceToHost));
    EXECUTE_CUDA_FUNC (cudaMemcpy (map_orig_embedding_to_additions, device_map_orig_embedding_to_additions, map_orig_embedding_to_additions_size, cudaMemcpyDeviceToHost));
    
    //Create new embeddings from the received additions
    for (int input_embedding_idx = 0; input_embedding_idx < input_embeddings.size (); input_embedding_idx++) {
      VectorVertexEmbedding& input_embedding = input_embeddings[input_embedding_idx];
      int n_additions = map_orig_embedding_to_additions[2*input_embedding_idx+1];
      int start_idx = map_orig_embedding_to_additions[2*input_embedding_idx];
      size_t produced_embedding_size = input_embedding.get_n_vertices () + n_additions;
      if (input_embedding_idx == 48) {
        std::cout << "n_additions " << n_additions << std::endl;
      }
      size_t global_mem_idx = GlobalMemAllocator::alloc_vertices_array (produced_embedding_size);
      // std::cout << "i " << input_embedding_idx << " produced_embedding_size " << produced_embedding_size << " global_mem_idx " << global_mem_idx << std::endl;
      VertexID* ptr = (VertexID*) ((char*)GlobalMemAllocator::get_global_mem_ptr () + global_mem_idx);
      ((VertexID*)ptr)[0] = ((VertexID*)input_embedding.get_array ())[0];
      memcpy (ptr, input_embedding.get_array (), sizeof(VertexID)*input_embedding.get_n_vertices ());
      memcpy (ptr + input_embedding.get_n_vertices (), &embedding_additions[start_idx], sizeof(VertexID)*n_additions);
      
      produced_embeddings.push_back (VectorVertexEmbedding ((uint32_t)produced_embedding_size, global_mem_idx, true));
    }

    input_embeddings = produced_embeddings;

    cudaFree (device_csr);
    cudaFree (device_embeddings_additions);
    cudaFree (device_map_orig_embedding_to_additions);
    cudaFree (device_input_embeddings_storage); 
    cudaFree (device_input_embeddings);
    cudaFree (device_embeddings_addition_iter);
  }
#endif

#ifdef PINNED_MEMORY
  // cudaFree (global_mem_ptr);
#else
  delete[] global_mem_ptr;
#endif
  std::cout << "Number of embeddings found "<< input_embeddings.size () << std::endl;
  std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  std::cout << "Time spent in Streams " << total_stream_time << std::endl;
}