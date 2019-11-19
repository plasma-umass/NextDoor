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
//#define USE_CSR_IN_SHARED
//#define EMBEDDING_IN_SHARED_MEM_PER_VERTEX
//#define USE_EMBEDDING_IN_GLOBAL_MEM
//#define USE_EMBEDDING_IN_SHARED_MEM
#define EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
#define ALL_THREAD_BLOCK_EMBEDDINGS_IN_SHARED_MEM_PER_VERTEX
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

public:
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


__global__ void run_single_step_embedding (void* void_csr, void* input, size_t n_embeddings, void* void_embedding_storage, uint64_t global_mem_start_idx,
                                           void* void_embeddings_additions, size_t embeddings_additions_size, 
                                           void* void_map_orig_embedding_to_additions, size_t map_orig_embedding_to_additions_size,
                                           unsigned long long int* embeddings_additions_iter)
{
  CSR* csr = (CSR*)void_csr;
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= n_embeddings) {
    return;
  }

  VectorVertexEmbedding* input_embeddings = (VectorVertexEmbedding*) input;
  VectorVertexEmbedding* input_embedding = &input_embeddings[thread_idx];
  VertexID* embeddings_additions = (VertexID*)void_embeddings_additions;
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
  }

  unsigned long long int additions_start_iter = atomicAdd (embeddings_additions_iter, new_edges);
  // printf ("additions_start_iter %ld\n", additions_start_iter);
  map_orig_embedding_to_additions[thread_idx*2] = additions_start_iter;
  map_orig_embedding_to_additions[thread_idx*2 + 1] = new_edges;

  for (int vertex_idx = 0; vertex_idx < input_embedding->get_n_vertices (); vertex_idx++) {
    VertexID vertex = input_embedding->get_vertex (vertex_idx, void_embedding_storage, global_mem_start_idx);
    int start_edge_idx = csr->get_start_edge_idx (vertex);
    const int end_edge_idx = csr->get_end_edge_idx (vertex);
      while (start_edge_idx <= end_edge_idx) {
        VertexID edge = csr->get_edges ()[start_edge_idx];
        embeddings_additions[additions_start_iter++] = edge;
        start_edge_idx++;
      }
    }
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

  iter = 1;
  double total_stream_time = 0;

  const size_t max_embedding_size_per_iter = (12000000/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
  double_t kernelTotalTime = 0.0;
  std::vector <VectorVertexEmbedding> produced_embeddings;

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

#ifdef PINNED_MEMORY
  // cudaFree (global_mem_ptr);
#else
  delete[] global_mem_ptr;
#endif
  std::cout << "Number of embeddings found "<< input_embeddings.size () << std::endl;
  std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  std::cout << "Time spent in Streams " << total_stream_time << std::endl;
}