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
#define ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING true
//#define USE_CSR_IN_SHARED
//#define EMBEDDING_IN_SHARED_MEM_PER_VERTEX
//#define USE_EMBEDDING_IN_GLOBAL_MEM
//#define USE_EMBEDDING_IN_SHARED_MEM
//#define ALL_THREAD_BLOCK_EMBEDDINGS_IN_SHARED_MEM_PER_VERTEX
#define USE_EMBEDDING_IN_LOCAL_MEM
#define PROCESS_EMBEDDINGS_PER_VERTEX
//#define EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
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

#define NEW_EMBEDDING_BUFFER_SIZE 10*1024 //Size in terms of Bytes //Setting it to 128 MB makes citeseer performs a lot better

//#define USE_CONSTANT_MEM

typedef uint8_t SharedMemElem;

//citeseer.graph
const int N = 3312;
const int N_EDGES = 9074;

//micro.graph
//const int N = 100000;
//const int N_EDGES = 2160312;

enum BUFFER_STATUS {
  GPU_USING,
  CPU_COPYING,
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

  int set_id (int _id) {id = _id;}
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

//template <size_t N> using VertexEmbedding = std::bitset<N>;

#define CVT_TO_NEXT_MULTIPLE(n,k) ((n) %(k) ==0 ? (n) : ((n)/(k)+1)*(k))
class BitVectorVertexEmbedding
{
private:
#ifdef USE_EMBEDDING_IN_SHARED_MEM
  unsigned char array[CVT_TO_NEXT_MULTIPLE(N/8, 32*sizeof(SharedMemElem))];
#else
  unsigned char array[(N/8)];
#endif

public:
  __device__ __host__
  BitVectorVertexEmbedding ()
  {
    //array = new unsigned char[convert_to_bytes_multiple(N)/8];
    assert (array != nullptr);
    reset ();
    assert (all_false () == true);
  }

  __host__ __device__
  size_t convert_to_bytes_multiple (size_t n)
  {
    return (n/8)*8;
  }

  __host__ __device__
  BitVectorVertexEmbedding (const BitVectorVertexEmbedding& embedding)
  {
    //array = new unsigned char[convert_to_bytes_multiple(N)/8];
    for (int i = 0; i <  convert_to_bytes_multiple(N)/8; i++) {
      array[i] = embedding.array[i];
    }
  }

  __host__ __device__
  void set (int index)
  {
    assert (index >= 0 and index < N);
    array[index/8] = array[index/8] | (1 << (index %8));
  }

  __host__ __device__
  void set ()
  {
    for (int i = 0; i < convert_to_bytes_multiple(N)/8; i++) {
      array[i] = (unsigned char) (~(0UL));
    }
  }

  __host__ __device__
  void reset ()
  {
    for (int i = 0; i < convert_to_bytes_multiple(N)/8; i++) {
      array[i] = 0;
    }
  }

  __host__ __device__
  void reset (int index)
  {
    assert (index >= 0 and index < N);
    array[index/8] = array[index/8] & (~(1UL << (index %8)));
  }

  __host__ __device__
  bool test (int index)
  {
    return (bool) ((array[index/8] >> (index % 8))&1);
  }

  __host__ __device__
  bool all_false ()
  {
    for (int i = 0; i < convert_to_bytes_multiple(N)/8; i++) {
      if (array[i] != 0UL) {
        return false;
      }
    }

    return true;
  }
  
  __host__ __device__
  int get_n_vertices () 
  {
    int n_vertices = 0;
    for (int i = 0; i < N; i++) {
      if (test(i) == true) {
        n_vertices++;
      }
    }
    
    return n_vertices;
  }
  
  __host__ __device__
  ~BitVectorVertexEmbedding ()
  {
    //delete[] array;
  }
};

//typedef BitVectorVertexEmbedding VertexEmbedding;

template <uint32_t size> 
class VectorVertexEmbedding
{
private:
  uint32_t array[size];
  uint32_t filled_size;
  
public:
  __device__ __host__
  VectorVertexEmbedding ()
  {
    filled_size = 0;
  }

  __host__ __device__
  VectorVertexEmbedding (const VectorVertexEmbedding<size>& embedding)
  {
  #if DEBUG
    assert (embedding.get_max_size () <= get_max_size ());
  #endif
    filled_size = 0;
    for (int i = 0; i < embedding.get_n_vertices (); i++) {
      add (embedding.get_vertex (i));
    }
  }
  
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
    return;
    int pos = 0;
    
    for (int i = 0; i < filled_size; i++) {
      if (array[i] > v) {
        pos = i;
        break;
      }
    }
    
    for (int i = filled_size-1; i >= pos ; i--) {
      array[i+1] = array[i];
    }
    
    array[pos] = v;
    filled_size++;
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
  
  __host__ __device__
  const bool has_logn (int v)
  {
    int l = 0;
    int r = filled_size-1;
    
    while (l <= r) {
      int m = l+(r-l)/2;
      
      if (array[m] == v)
        return true;
      
      if (array[m] < v)
        l = m + 1;
      else
        r = m - 1;
    }
    
    return false;
  }
  
  __host__ __device__
  const bool has (int v)
  {
    for (int i = 0; i < filled_size; i++) {
      if (array[i] == v) {
        return true;
      }
    }
    
    return false;
  }
  
  __host__ __device__
  const size_t get_n_vertices () const
  {
    return filled_size;
  }
  
  __host__ __device__
  const int get_vertex (int index) const
  {
    return array[index];
  }
  
  __host__ __device__
  const int get_last_vertex () const
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

template <uint32_t size>
__host__ __device__
void vector_embedding_from_one_less_size (VectorVertexEmbedding<size> const & vec_emb1,
                                          VectorVertexEmbedding<size+1>& vec_emb2)
{
  //TODO: Optimize here, filled_size++ in add is being called several times
  //but can be called only once too
  //if  (false and vec_emb1.get_n_vertices () != size) {
  //  printf ("vec_emb1.get_n_vertices () %ld != size %d\n", vec_emb1.get_n_vertices (), size);
  //  assert (false);
  //}
  for (int i = 0; i < vec_emb1.get_n_vertices (); i++) {
    vec_emb2.add (vec_emb1.get_vertex (i));
  }
}

template <uint32_t size> 
void bitvector_to_vector_embedding (BitVectorVertexEmbedding& bit_emb, 
                                    VectorVertexEmbedding<size>& vec_emb)
{
  for (int u = 0; u < N; u++) {
    if (bit_emb.test(u)) {
      vec_emb.add (u);
    }
  }
}

void print_embedding (BitVectorVertexEmbedding embedding, std::ostream& os);

std::vector<BitVectorVertexEmbedding> get_extensions_bitvector (BitVectorVertexEmbedding& embedding, CSR* csr)
{
  std::vector<BitVectorVertexEmbedding> extensions;

  if (embedding.all_false ()) {
    for (int u = 0; u < N; u++) {
      BitVectorVertexEmbedding extension;
      extension.set(u);
      //print_embedding (extension, std::cout);
      //std::cout << " " << u << std::endl;
      extensions.push_back (extension);
    }
  } else {
    for (int u = 0; u < N; u++) {
      if (embedding.test(u)) {
        for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
          int v = csr->get_edges () [e];
          if (embedding.test (v) == false) {
            BitVectorVertexEmbedding extension = BitVectorVertexEmbedding(embedding);
            extension.set(v);
            extensions.push_back(extension);
          }
        }
      }
    }
  }

  return extensions;
}

template <uint32_t size>
std::vector<VectorVertexEmbedding<size+1>> get_extensions_vector (VectorVertexEmbedding<size>& embedding, CSR* csr)
{
  std::vector<VectorVertexEmbedding<size+1>> extensions;

  if (embedding.get_n_vertices () == 0) {
    for (int u = 0; u < N; u++) {
      VectorVertexEmbedding<size+1> extension;
      extension.add(u);
      //print_embedding (extension, std::cout);
      //std::cout << " " << u << std::endl;
      extensions.push_back (extension);
    }
  } else {
    for (int i = 0; i < embedding.get_n_vertices (); i++) {
      int u = embedding.get_vertex (i);
      for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
        int v = csr->get_edges () [e];
        if (embedding.has (v) == false) {
          VectorVertexEmbedding<size+1> extension;
          vector_embedding_from_one_less_size (embedding, extension);
          extension.add(v);
          extensions.push_back(extension);
        }
      }
    }
  }

  return extensions;
}

std::vector<BitVectorVertexEmbedding> get_initial_embedding_bitvector (CSR* csr)
{
  BitVectorVertexEmbedding embedding;
  std::vector <BitVectorVertexEmbedding> embeddings;

  embeddings.push_back (embedding);

  return embeddings;
}

std::vector<VectorVertexEmbedding<0>> get_initial_embedding_vector (CSR* csr)
{
  VectorVertexEmbedding<0> embedding;
  std::vector <VectorVertexEmbedding<0>> embeddings;

  embeddings.push_back (embedding);

  return embeddings;
}

bool (*filter) (CSR* csr, BitVectorVertexEmbedding& embedding);
void (*process) (std::vector<BitVectorVertexEmbedding>& output, BitVectorVertexEmbedding& embedding);

__host__ __device__
bool clique_filter (CSR* csr, BitVectorVertexEmbedding* embedding)
{
  for (int u = 0; u < N; u++) {
    if (embedding->test(u)) {
      for (int v = 0; v < N; v++) {
        if (u != v and embedding->test(v)) {
          if (!csr->has_edge (u, v)) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

template <int size>
void print_embedding (VectorVertexEmbedding<size>* embedding)
{
  std::cout << "{";
  for (int i = 0; i < size; i++) {
    std::cout << embedding->get_vertex (i) << ", ";
  }
  std::cout << "}";
}

template <int size>
std::string embedding_to_string (VectorVertexEmbedding<size>* embedding)
{
  std::string s = "";
  s = "{";
  for (int i = 0; i < size; i++) {
    s += embedding->get_vertex (i) + ", ";
  }
  s += "}";
  return s;
}

template <uint32_t size>
__host__ __device__
bool clique_filter_vector (CSR* csr, VectorVertexEmbedding<size>* embedding)
{
  for (int i = 0; i < embedding->get_n_vertices (); i++) {
    int u = embedding->get_vertex (i);
    for (int j = 0; j < embedding->get_n_vertices (); j++) {
      int v = embedding->get_vertex (j);
      if (u != v) {
        if (!csr->has_edge (u, v)) {
          return false;
        }
      }
    }
  }

  return true;
}

template <uint32_t size>
__host__ __device__
bool clique_filter_vector_optimized (CSR* csr, VectorVertexEmbedding<size> const * embedding, int last_vertex)
{
  for (int i = 0; i < embedding->get_n_vertices (); i++) {
    int u = embedding->get_vertex (i);
    if (!csr->has_edge (u, last_vertex)) {
        return false;
    }
  }

  return true;
}

void clique_process_bit_vector (std::vector<BitVectorVertexEmbedding>& output, BitVectorVertexEmbedding& embedding)
{
  output.push_back (embedding);
}

template <uint32_t size>
void clique_process_vector (std::vector<VectorVertexEmbedding<size>>& output, VectorVertexEmbedding<size>& embedding)
{
  output.push_back (embedding);
}

void run_single_step_initial_bitvector (void* input, int n_embeddings, CSR* csr,
                      std::vector<BitVectorVertexEmbedding>& output,
                      std::vector<BitVectorVertexEmbedding>& next_step)
{
  BitVectorVertexEmbedding* embeddings = (BitVectorVertexEmbedding*)input;

  for (int i = 0; i < n_embeddings; i++) {
    BitVectorVertexEmbedding embedding = embeddings[i];
    std::vector<BitVectorVertexEmbedding> extensions = get_extensions_bitvector (embedding, csr);

    for (auto extension : extensions) {
      if (clique_filter (csr, &extension)) {
        clique_process_bit_vector (output, extension);
        next_step.push_back (extension);
      }
    }
  }
}

void run_single_step_initial_vector (void* input, int n_embeddings, CSR* csr,
                      std::vector<VectorVertexEmbedding<1>>& output,
                      std::vector<VectorVertexEmbedding<1>>& next_step)
{
  VectorVertexEmbedding<0>* embeddings = (VectorVertexEmbedding<0>*)input;

  for (int i = 0; i < n_embeddings; i++) {
    VectorVertexEmbedding<0> embedding = embeddings[i];
    std::vector<VectorVertexEmbedding<1>> extensions = get_extensions_vector (embedding, csr);
    std::cout << "extensions " << extensions.size () << std::endl;
    for (auto extension : extensions) {
      if (clique_filter_vector (csr, &extension)) {
        clique_process_vector (output, extension);
        next_step.push_back (extension);
      }
    }
  }
}

__device__
void printf_embedding (BitVectorVertexEmbedding* embedding)
{
  printf ("[");
  for (int u = 0; u < N; u++) {
    if (embedding->test(u)) {
      printf ("%d, ", u);
    }
  }

  printf ("]\n");
}

/*__global__
void run_single_step_bitvector_embedding (void* input, int n_embeddings, CSR* csr,
                      void* output_ptr,
                      int* n_output,
                      void* next_step, int* n_next_step,
                      int* n_output_1, int* n_next_step_1)
{
  int id;

#ifdef USE_CSR_IN_SHARED
  __shared__ unsigned char csr_shared_buff[sizeof (CSR)];
  id = threadIdx.x;
  CSR* csr_shared = (CSR*) csr_shared_buff;
  csr_shared->n_vertices = csr->get_n_vertices ();
  csr_shared->n_edges = csr->get_n_edges ();

  int vertices_per_thread = csr->get_n_vertices ()/THREAD_BLOCK_SIZE + 1;
  csr_shared->copy_vertices (csr, id*vertices_per_thread,
                             (id+1)*vertices_per_thread < csr->get_n_vertices () ? (id+1)*vertices_per_thread : csr->get_n_vertices ());

  int edges_per_thread = csr->get_n_edges ()/THREAD_BLOCK_SIZE + 1;
  csr_shared->copy_edges (csr, id*edges_per_thread,
                          (id+1)*edges_per_thread < csr->get_n_edges () ? (id+1)*edges_per_thread : csr->get_n_edges ());
  csr = csr_shared;
  __syncthreads ();
#else
#ifdef USE_CONSTANT_MEM
  csr = (CSR*) csr_constant_buff;
#endif

#endif

  BitVectorVertexEmbedding* embeddings = (BitVectorVertexEmbedding*)input;
  BitVectorVertexEmbedding* new_embeddings = (BitVectorVertexEmbedding*)next_step;
  BitVectorVertexEmbedding* output = ((BitVectorVertexEmbedding*)output_ptr);
#ifdef USE_EMBEDDING_IN_LOCAL_MEM
  unsigned char temp_buffer [sizeof(BitVectorVertexEmbedding)];
#endif
  id = blockIdx.x*blockDim.x + threadIdx.x;
  int start = id, end = id+1;
  //printf ("running id %d\n", id);
#ifdef USE_FIXED_THREADS
  if (n_embeddings >= MAX_CUDA_THREADS) {
    int embeddings_per_thread = n_embeddings/MAX_CUDA_THREADS+1;

    start = id*embeddings_per_thread;
    end = (id+1)*embeddings_per_thread < n_embeddings ? (id+1)*embeddings_per_thread : n_embeddings;
  } else {
    if (id >= n_embeddings)
      return;

    start = id;
    end = id+1;
  }
#else
  if (id >= n_embeddings)
      return;

  start = id;
  end = id+1;
#endif

  int q[1000] = {0};

#ifdef USE_EMBEDDING_IN_SHARED_MEM
  #if 0
    const int shared_mem_size = 49152;
    const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

    assert (per_thread_shared_mem_size >= sizeof (VertexEmbedding));
    //per_thread_shared_mem_size = sizeof (VertexEmbedding);
    __shared__ SharedMemElem shared_buff[per_thread_shared_mem_size/sizeof (SharedMemElem)];

    SharedMemElem* local_shared_buff = &shared_buff[0];
  #else
    const int shared_mem_size = 49152;
    const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

    //assert (per_thread_shared_mem_size >= sizeof (VertexEmbedding));
    //per_thread_shared_mem_size = sizeof (VertexEmbedding);
    __shared__ SharedMemElem shared_buff[shared_mem_size/sizeof (SharedMemElem)];

    SharedMemElem* local_shared_buff = &shared_buff[per_thread_shared_mem_size/sizeof(SharedMemElem)*threadIdx.x];
  #endif
#endif

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  for (int i = start; i < end; i++) {
    #ifdef USE_EMBEDDING_IN_SHARED_MEM
      #ifdef SHARED_MEM_NON_COALESCING
        memcpy (local_shared_buff, &embeddings[i], sizeof(BitVectorVertexEmbedding));

        BitVectorVertexEmbedding* embedding = (BitVectorVertexEmbedding*) local_shared_buff;
      #else
        const int thread_block_size = WARP_SIZE;
        const int last_emb = WARP_SIZE*(warp_id+1);
        if (blockIdx.x*blockDim.x + (warp_id+1)*WARP_SIZE > n_embeddings) {
          assert (false);
          //thread_block_size = n_embeddings - blockIdx.x*blockDim.x -
          //                    warp_id*WARP_SIZE;
          //last_emb = warp_id*WARP_SIZE + thread_block_size;
        }

        for (int emb = WARP_SIZE*warp_id; emb < last_emb; emb++) {
          SharedMemElem* embedding_buff = (SharedMemElem*) &embeddings[emb+blockIdx.x*blockDim.x];

          for (int j = 0; j < sizeof (BitVectorVertexEmbedding)/sizeof (SharedMemElem);
               j += thread_block_size) {
            int idx = per_thread_shared_mem_size/sizeof(SharedMemElem)*emb;
            if (true or j + lane_id  < sizeof (BitVectorVertexEmbedding)/sizeof (SharedMemElem)) { //TODO: Remove this if by doing padding with VertexEmbedding
              shared_buff[idx + j + lane_id] = embedding_buff[j + lane_id];
            }
          }
        }

        BitVectorVertexEmbedding* embedding = (BitVectorVertexEmbedding*) local_shared_buff;
      #endif
      //embedding = &embeddings[i];
    #elif defined(USE_EMBEDDING_IN_LOCAL_MEM)
      //memcpy (&temp[0], &embeddings[i], sizeof (VertexEmbedding));
      memcpy (&temp_buffer[0], &embeddings[i], sizeof(BitVectorVertexEmbedding));
      BitVectorVertexEmbedding* embedding = (BitVectorVertexEmbedding*)&temp_buffer[0];
      //VertexEmbedding* embedding = &embeddings[i];
    #elif defined(USE_EMBEDDING_IN_GLOBAL_MEM)
      BitVectorVertexEmbedding* embedding = &embeddings[i];
    #else
      #error "None of USE_EMBEDDING_IN_*_MEM option defined"
    #endif

  #if 1
    typedef uint32_t VertexEmbeddingChange;
    const uint32_t max_changes = 32; //max changes per warp
    //__shared__ int32_t shared_mem [max_changes*THREAD_BLOCK_SIZE/WARP_SIZE+THREAD_BLOCK_SIZE/WARP_SIZE+1];
    __shared__ VertexEmbeddingChange changes[max_changes*THREAD_BLOCK_SIZE/WARP_SIZE];
    __shared__ uint32_t changed_thread_ids[max_changes*THREAD_BLOCK_SIZE/WARP_SIZE];
    //For each warp record new embeddings (or changes) using atomic operations
    __shared__ int32_t n_extensions[THREAD_BLOCK_SIZE/WARP_SIZE];
    n_extensions[warp_id] = 0;
    __shared__ uint32_t prev_n_outputs [THREAD_BLOCK_SIZE/WARP_SIZE];
    __shared__ uint32_t prev_n_next_steps [THREAD_BLOCK_SIZE/WARP_SIZE];
    
    const uint32_t mask = ~0U;
    for (int u = 0; u < N; u++) {
      int e = csr->get_start_edge_idx(u);
      
      while (true) {
        
        int predicate = 1;
        //n_extensions[warp_id] = 0;

        if (e <= csr->get_end_edge_idx(u)) {
          int v = csr->get_edges () [e];
          if (embedding->test(u) and embedding->test (v) == false) {
            BitVectorVertexEmbedding* extension = embedding;
            extension->set(v);
            if (clique_filter (csr, extension)) {
              int prev_n_extensions = atomicAdd (&n_extensions[warp_id], 1);
              changes[warp_id*max_changes+prev_n_extensions] = v;
              changed_thread_ids[warp_id*max_changes+prev_n_extensions] = id;
              //memcpy (&output[atomicAdd(n_output,1)], extension, sizeof (VertexEmbedding));
              //memcpy (&new_embeddings[atomicAdd(n_next_step,1)], extension, sizeof (VertexEmbedding));
            }

            extension->reset(v);
          }

          e++;

          if (e > csr->get_end_edge_idx(u)) {
            predicate = 0;
          }
        } else {
          predicate = 0;
        }

        int32_t n_changes = n_extensions[warp_id];

        if (n_changes > max_changes)
          assert (false);

        //assert (__activemask () == ~0U);
        uint32_t orig_prev_n_output = 0;
        uint32_t orig_prev_n_next_step = 0;
        
        if (lane_id == 0) {
          if (n_changes > 0) {
            n_extensions[warp_id] = 0;
            orig_prev_n_output = atomicAdd (n_output, n_changes);
            orig_prev_n_next_step = atomicAdd (n_next_step, n_changes);
            //if (warp_id == 2 && (threadIdx.x == 64 || threadIdx.x == 65) && blockIdx.x == 0) {
            //  printf ("prev_n_next_steps[warp_id]: %d\n", prev_n_next_steps[warp_id]);
            //}
          }
        }
        
        //__syncwarp ();
        
        if (n_changes > 0) {
          
          uint32_t prev_n_output = __shfl_sync (__activemask (), orig_prev_n_output, 0);
          uint32_t prev_n_next_step = __shfl_sync (__activemask (), orig_prev_n_next_step, 0);
          
          for (int i = 0; i < n_changes; i++) {
            
            int changes_idx = warp_id*max_changes + i;
            uint32_t expected_thread_id = changed_thread_ids[changes_idx];
            if (expected_thread_id == id) {
              int v = changes[changes_idx];
              BitVectorVertexEmbedding* extension = embedding;
              extension->set (v);
              
              memcpy (&output[prev_n_output + i], extension, 
                      sizeof(BitVectorVertexEmbedding));
              memcpy (&new_embeddings[prev_n_next_step + i], extension,
                      sizeof (BitVectorVertexEmbedding));
              extension->reset (v);
            }
          }
        }

        if (n_extensions[warp_id] != 0) {
          printf ("n_extensions[warp_id] not zero but is %d\n", n_extensions[warp_id]);
        }
        if (!__any_sync (__activemask (), predicate)) {
          break;
        }
        //n_extensions[warp_id] = 0;
      }
    }
  #else
    for (int u = 0; u < N; u++) {
      if (embedding->test(u)) {
        for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
          int v = csr->get_edges () [e];
          if (embedding->test (v) == false) {
            BitVectorVertexEmbedding* extension = embedding;
            extension->set(v);
            if (clique_filter (csr, extension)) {
              memcpy (&output[atomicAdd(n_output,1)], extension, sizeof (BitVectorVertexEmbedding));
              memcpy (&new_embeddings[atomicAdd(n_next_step,1)], extension, sizeof (BitVectorVertexEmbedding));
            }
            extension->reset(v);
          }
        }
      }
    }
  #endif

  }

  //printf ("embeddings generated [1000, 2000)= %d and [2000, 3000) = %d\n", q[0], q[1]);
  //printf ("embeddings at i = 1500: %d\n", q[2]);
  //for (int i = 100; i < 1000; i++) {
  //  printf ("embeddings at i = %d %d\n", i, q[i]);
  //}
}*/


template <size_t size>
__host__ __device__
inline bool is_embedding_canonical (CSR* csr, VectorVertexEmbedding<size> const * embedding, int v)
{
  if (embedding->get_vertex (0) > v)
    return false;
  
  //if (size <= 2)
  //  return true;
  
  bool found_neighbor = false;
  for (int j = 0; j < embedding->get_n_vertices (); j++) {
    int v_j = embedding->get_vertex (j);
    if (found_neighbor == false && csr->has_edge (v_j, v)) {
      found_neighbor = true;
    } else if (found_neighbor == true && v_j > v) {
      return false;
    }
  }

  return true;
}

template <size_t embedding_size> 
__global__
void run_single_step_vectorvertex_embedding (void* input, int n_embeddings, CSR* csr,
                      void* output_ptr,
                      int* n_output,
                      void* next_step_1, int* n_next_step_1, volatile BUFFER_STATUS* buff_1_status, //Should be volatile because of the busy wait loop
                      void* next_step_2, int* n_next_step_2, volatile BUFFER_STATUS* buff_2_status, //Should be volatile because of the busy wait loop
                      volatile int* curr_step_storage_id, //Should be volatile because threads coordinate based on this value
                      int only_copy_change)
{
  int id;

#ifdef USE_CSR_IN_SHARED
  __shared__ unsigned char csr_shared_buff[sizeof (CSR)];
  id = threadIdx.x;
  CSR* csr_shared = (CSR*) csr_shared_buff;
  csr_shared->n_vertices = csr->get_n_vertices ();
  csr_shared->n_edges = csr->get_n_edges ();

  int vertices_per_thread = csr->get_n_vertices ()/THREAD_BLOCK_SIZE + 1;
  csr_shared->copy_vertices (csr, id*vertices_per_thread,
                             (id+1)*vertices_per_thread < csr->get_n_vertices () ? (id+1)*vertices_per_thread : csr->get_n_vertices ());

  int edges_per_thread = csr->get_n_edges ()/THREAD_BLOCK_SIZE + 1;
  csr_shared->copy_edges (csr, id*edges_per_thread,
                          (id+1)*edges_per_thread < csr->get_n_edges () ? (id+1)*edges_per_thread : csr->get_n_edges ());
  csr = csr_shared;
  __syncthreads ();
#else
#ifdef USE_CONSTANT_MEM
  csr = (CSR*) csr_constant_buff;
#endif

#endif

  VectorVertexEmbedding<embedding_size>* embeddings = (VectorVertexEmbedding<embedding_size>*)input;
  
#ifdef USE_EMBEDDING_IN_LOCAL_MEM
  unsigned char temp_buffer [sizeof(VectorVertexEmbedding<embedding_size+1>)];
#endif
  id = blockIdx.x*blockDim.x + threadIdx.x;
  int start = id, end = id+1;
  //printf ("running id %d\n", id);
#ifdef USE_FIXED_THREADS
  if (n_embeddings >= MAX_CUDA_THREADS) {
    int embeddings_per_thread = n_embeddings/MAX_CUDA_THREADS+1;

    start = id*embeddings_per_thread;
    end = (id+1)*embeddings_per_thread < n_embeddings ? (id+1)*embeddings_per_thread : n_embeddings;
  } else {
    if (id >= n_embeddings)
      return;

    start = id;
    end = id+1;
  }
#else
  if (id >= n_embeddings)
      return;

  start = id;
  end = id+1;
#endif

  int q[1000] = {0};

#ifdef USE_EMBEDDING_IN_SHARED_MEM
//TODO: Support VectorVertexEmbedding
  #if 0
    const int shared_mem_size = 49152;
    const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

    assert (per_thread_shared_mem_size >= sizeof (VectorVertexEmbedding<embedding_size>));
    //per_thread_shared_mem_size = sizeof (VertexEmbedding);
    __shared__ SharedMemElem shared_buff[per_thread_shared_mem_size/sizeof (SharedMemElem)];

    SharedMemElem* local_shared_buff = &shared_buff[0];
  #else
    const int shared_mem_size = 49152;
    const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

    //assert (per_thread_shared_mem_size >= sizeof (VertexEmbedding));
    //per_thread_shared_mem_size = sizeof (VertexEmbedding);
    __shared__ SharedMemElem shared_buff[shared_mem_size/sizeof (SharedMemElem)];

    SharedMemElem* local_shared_buff = &shared_buff[per_thread_shared_mem_size/sizeof(SharedMemElem)*threadIdx.x];
  #endif
#endif

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  for (int i = start; i < end; i++) {
    #ifdef USE_EMBEDDING_IN_SHARED_MEM
    //TODO: Support VectorVertexEmbedding, size+1
      #ifdef SHARED_MEM_NON_COALESCING
        memcpy (local_shared_buff, &embeddings[i], sizeof(VectorVertexEmbedding<embedding_size>));

        VectorVertexEmbedding<embedding_size>* embedding = (VectorVertexEmbedding<embedding_size>*) local_shared_buff;
      #else
        const int thread_block_size = WARP_SIZE;
        const int last_emb = WARP_SIZE*(warp_id+1);
        if (blockIdx.x*blockDim.x + (warp_id+1)*WARP_SIZE > n_embeddings) {
          assert (false);
          /*thread_block_size = n_embeddings - blockIdx.x*blockDim.x -
                              warp_id*WARP_SIZE;
          last_emb = warp_id*WARP_SIZE + thread_block_size;*/
        }

        for (int emb = WARP_SIZE*warp_id; emb < last_emb; emb++) {
          SharedMemElem* embedding_buff = (SharedMemElem*) &embeddings[emb+blockIdx.x*blockDim.x];

          for (int j = 0; j < sizeof (VectorVertexEmbedding<embedding_size>)/sizeof (SharedMemElem);
               j += thread_block_size) {
            int idx = per_thread_shared_mem_size/sizeof(SharedMemElem)*emb;
            if (true or j + lane_id  < sizeof (VectorVertexEmbedding<embedding_size>)/sizeof (SharedMemElem)) { //TODO: Remove this if by doing padding with VertexEmbedding
              shared_buff[idx + j + lane_id] = embedding_buff[j + lane_id];
            }
          }
        }

        VectorVertexEmbedding<embedding_size>* embedding = (VectorVertexEmbedding<embedding_size>*) local_shared_buff;
      #endif
      //embedding = &embeddings[i];
    #elif defined(USE_EMBEDDING_IN_LOCAL_MEM)
      //memcpy (&temp[0], &embeddings[i], sizeof (VertexEmbedding));
      //memcpy (&temp_buffer[0], &embeddings[i], sizeof(VectorVertexEmbedding<embedding_size>));
      VectorVertexEmbedding<embedding_size+1>* embedding = (VectorVertexEmbedding<embedding_size+1>*)&temp_buffer[0];
      embedding->clear ();
      vector_embedding_from_one_less_size (embeddings[i], *embedding);
      //VertexEmbedding* embedding = &embeddings[i];
    #elif defined(USE_EMBEDDING_IN_GLOBAL_MEM)
    //TODO: Support VectorVertexEmbedding with size + 1, below is wrong
      VectorVertexEmbedding<embedding_size+1>* embedding = &embeddings[i];
    #else
      #error "None of USE_EMBEDDING_IN_*_MEM option defined"
    #endif

  #if 0
  //TODO: Support VectorVertexEmbedding with size + 1.
    typedef uint32_t VertexEmbeddingChange;
    const uint32_t max_changes = 32; //max changes per warp
    //__shared__ int32_t shared_mem [max_changes*THREAD_BLOCK_SIZE/WARP_SIZE+THREAD_BLOCK_SIZE/WARP_SIZE+1];
    __shared__ VertexEmbeddingChange changes[max_changes*THREAD_BLOCK_SIZE/WARP_SIZE];
    __shared__ uint32_t changed_thread_ids[max_changes*THREAD_BLOCK_SIZE/WARP_SIZE];
    //For each warp record new embeddings (or changes) using atomic operations
    __shared__ int32_t n_extensions[THREAD_BLOCK_SIZE/WARP_SIZE];
    n_extensions[warp_id] = 0;
    __shared__ uint32_t prev_n_outputs [THREAD_BLOCK_SIZE/WARP_SIZE];
    __shared__ uint32_t prev_n_next_steps [THREAD_BLOCK_SIZE/WARP_SIZE];
    
    const uint32_t mask = ~0U;
    for (int u = 0; u < N; u++) {
      int e = csr->get_start_edge_idx(u);
      
      while (true) {
        
        int predicate = 1;
        //n_extensions[warp_id] = 0;

        if (e <= csr->get_end_edge_idx(u)) {
          int v = csr->get_edges () [e];
          if (embedding->test(u) and embedding->test (v) == false) {
            BitVectorVertexEmbedding* extension = embedding;
            extension->set(v);
            if (clique_filter (csr, extension)) {
              int prev_n_extensions = atomicAdd (&n_extensions[warp_id], 1);
              changes[warp_id*max_changes+prev_n_extensions] = v;
              changed_thread_ids[warp_id*max_changes+prev_n_extensions] = id;
              //memcpy (&output[atomicAdd(n_output,1)], extension, sizeof (VertexEmbedding));
              //memcpy (&new_embeddings[atomicAdd(n_next_step,1)], extension, sizeof (VertexEmbedding));
            }

            extension->reset(v);
          }

          e++;

          if (e > csr->get_end_edge_idx(u)) {
            predicate = 0;
          }
        } else {
          predicate = 0;
        }

        int32_t n_changes = n_extensions[warp_id];

        if (n_changes > max_changes)
          assert (false);

        //assert (__activemask () == ~0U);
        uint32_t orig_prev_n_output = 0;
        uint32_t orig_prev_n_next_step = 0;
        
        if (lane_id == 0) {
          if (n_changes > 0) {
            n_extensions[warp_id] = 0;
            orig_prev_n_output = atomicAdd (n_output, n_changes);
            orig_prev_n_next_step = atomicAdd (n_next_step, n_changes);
            //if (warp_id == 2 && (threadIdx.x == 64 || threadIdx.x == 65) && blockIdx.x == 0) {
            //  printf ("prev_n_next_steps[warp_id]: %d\n", prev_n_next_steps[warp_id]);
            //}
          }
        }
        
        //__syncwarp ();
        
        if (n_changes > 0) {
          
          uint32_t prev_n_output = __shfl_sync (__activemask (), orig_prev_n_output, 0);
          uint32_t prev_n_next_step = __shfl_sync (__activemask (), orig_prev_n_next_step, 0);
          
          for (int i = 0; i < n_changes; i++) {
            
            int changes_idx = warp_id*max_changes + i;
            uint32_t expected_thread_id = changed_thread_ids[changes_idx];
            if (expected_thread_id == id) {
              int v = changes[changes_idx];
              BitVectorVertexEmbedding* extension = embedding;
              extension->set (v);
              
              memcpy (&output[prev_n_output + i], extension, 
                      sizeof(BitVectorVertexEmbedding));
              memcpy (&new_embeddings[prev_n_next_step + i], extension,
                      sizeof (BitVectorVertexEmbedding));
              extension->reset (v);
            }
          }
        }

        if (n_extensions[warp_id] != 0) {
          printf ("n_extensions[warp_id] not zero but is %d\n", n_extensions[warp_id]);
        }
        if (!__any_sync (__activemask (), predicate)) {
          break;
        }
        //n_extensions[warp_id] = 0;
      }
    }
  #else
    for (int i = 0; i < embedding->get_n_vertices (); i++) {
      int u = embedding->get_vertex (i);
      for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
        
        int v = csr->get_edges () [e];

        if (is_embedding_canonical<embedding_size+1> (csr, embedding, v) && embedding->has (v) == false) {
          VectorVertexEmbedding<embedding_size+1>* extension = embedding;
          extension->add_unsorted (v);
          
          if (clique_filter_vector (csr, extension)) {
            //VectorVertexEmbedding<embedding_size+1> extension = *embedding;
            //extension.add_last_in_sort_order ();
            //int o = atomicAdd(n_output,1);
            //int n = atomicAdd(n_next_step_1,1);
            
            if (only_copy_change) {
              int o = atomicAdd(n_output, 1);
              int n = atomicAdd(n_next_step_1, 1);
              int* new_embeddings = (int*) next_step_1;
              int* output = (int*) output_ptr;

              new_embeddings[2*n] = id;
              new_embeddings[2*n+1] = v;
              output[2*o] = id;
              output[2*o+1] = v;
            }
            else {
              int storage_id = *curr_step_storage_id;
              const size_t max_n_embeddings = NEW_EMBEDDING_BUFFER_SIZE/sizeof (VectorVertexEmbedding<embedding_size+1>);
              //const int storage_id = 0;
              int n = 0;
              switch (storage_id) {
                case 0: {
                  int o = atomicAdd(n_output, 1);
                  n = atomicAdd(n_next_step_1, 1);
                  //Switch from buff1 to buff2
                  while (n >= max_n_embeddings) {//TODO: change it to do-while 
                    if (*curr_step_storage_id == 0) {
                      n = atomicSub (n_next_step_1, 1); //TODO: can remove that
                      while (*buff_2_status == BUFFER_STATUS::CPU_COPYING) {
                        /*unsigned long i = 0;
                        while (i <= (1UL<<30)) {
                          i++;
                        }*/
                      }
                      *curr_step_storage_id = 1;
                      *buff_1_status = BUFFER_STATUS::CPU_COPYING;
                      *buff_2_status = BUFFER_STATUS::GPU_USING;
                      n = atomicAdd(n_next_step_2, 1);
                    } else {
                      n = atomicSub (n_next_step_2, 1); //TODO: can remove that
                      while (*buff_1_status == BUFFER_STATUS::CPU_COPYING) {
                        /*unsigned long i = 0;
                        while (i <= (1UL<<30)) {
                          i++;
                        }*/
                      }
                      *curr_step_storage_id = 0;
                      *buff_2_status = BUFFER_STATUS::CPU_COPYING;
                      *buff_1_status = BUFFER_STATUS::GPU_USING;
                      n = atomicAdd(n_next_step_1, 1);
                    }
                  }
                  
                  if (*curr_step_storage_id == 1) {
                    //n = atomicAdd (n_next_step_2, 1);
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_2;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    //memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  } else {
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_1;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    //memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  }
                  break;
                }

                case 1: {
                  int o = atomicAdd(n_output, 1);
                  n = atomicAdd(n_next_step_2, 1);
                  if (n >= max_n_embeddings) {
                    //Switch from buff2 to buff1
                    atomicSub (n_next_step_2, 1); //TODO: can remove that
                    *curr_step_storage_id = 0;
                    *buff_2_status = BUFFER_STATUS::CPU_COPYING;
                    while (*buff_1_status == BUFFER_STATUS::CPU_COPYING) {
                      /*unsigned long i = 0;
                      while (i <= (1UL<<30)) {
                        i++;
                      }*/
                    }
                    *buff_1_status = BUFFER_STATUS::GPU_USING;
                    n = atomicAdd (n_next_step_1, 1);
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_1;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    //memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  } else {
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_2;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    //memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  }
                }
              }
            }
            //output[o].add_last_in_sort_order ();
            //new_embeddings[n].add_last_in_sort_order ();
          }
          extension->remove_last ();
        }
      }
    }
  #endif
  }

  //printf ("embeddings generated [1000, 2000)= %d and [2000, 3000) = %d\n", q[0], q[1]);
  //printf ("embeddings at i = 1500: %d\n", q[2]);
  //for (int i = 100; i < 1000; i++) {
  //  printf ("embeddings at i = %d %d\n", i, q[i]);
  //}
}

__host__ __device__ struct PairEmbIdxVertexIdx {
  int emb_idx;
  int vertex_idx;

  PairEmbIdxVertexIdx (long _emb_idx, int _vertex_idx) {
    emb_idx = _emb_idx;
    vertex_idx = _vertex_idx;
  }
};

template <size_t embedding_size> 
__global__
void run_single_step_vectorvertex_embedding_per_vertex (void* input, int n_embeddings, CSR* csr,
                      void* embs_per_parts, 
                      void* output_ptr,
                      int* n_output,
                      void* next_step_1, int* n_next_step_1, volatile BUFFER_STATUS* buff_1_status, //Should be volatile because of the busy wait loop
                      void* next_step_2, int* n_next_step_2, volatile BUFFER_STATUS* buff_2_status, //Should be volatile because of the busy wait loop
                      volatile int* curr_step_storage_id, //Should be volatile because threads coordinate based on this value
                      int* n_next_step_1_done, int* n_next_step_2_done, 
                      int only_copy_change)
{
  int id;

#ifdef USE_CSR_IN_SHARED
  __shared__ unsigned char csr_shared_buff[sizeof (CSR)];
  id = threadIdx.x;
  CSR* csr_shared = (CSR*) csr_shared_buff;
  csr_shared->n_vertices = csr->get_n_vertices ();
  csr_shared->n_edges = csr->get_n_edges ();

  int vertices_per_thread = csr->get_n_vertices ()/THREAD_BLOCK_SIZE + 1;
  csr_shared->copy_vertices (csr, id*vertices_per_thread,
                             (id+1)*vertices_per_thread < csr->get_n_vertices () ? (id+1)*vertices_per_thread : csr->get_n_vertices ());

  int edges_per_thread = csr->get_n_edges ()/THREAD_BLOCK_SIZE + 1;
  csr_shared->copy_edges (csr, id*edges_per_thread,
                          (id+1)*edges_per_thread < csr->get_n_edges () ? (id+1)*edges_per_thread : csr->get_n_edges ());
  csr = csr_shared;
  __syncthreads ();
#else
#ifdef USE_CONSTANT_MEM
  csr = (CSR*) csr_constant_buff;
#endif

#endif

  VectorVertexEmbedding<embedding_size>* embeddings = (VectorVertexEmbedding<embedding_size>*)input;
  PairEmbIdxVertexIdx* embeddings_per_partitions = (PairEmbIdxVertexIdx*) embs_per_parts;

id = blockIdx.x*blockDim.x + threadIdx.x;

#if defined(EMBEDDING_IN_SHARED_MEM_PER_VERTEX)
//TODO: Support VectorVertexEmbedding
  #if 0
    const int shared_mem_size = 49152;
    const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

    assert (per_thread_shared_mem_size >= sizeof (VectorVertexEmbedding<embedding_size>));
    //per_thread_shared_mem_size = sizeof (VertexEmbedding);
    __shared__ SharedMemElem shared_buff[per_thread_shared_mem_size/sizeof (SharedMemElem)];

    SharedMemElem* local_shared_buff = &shared_buff[0];
  #else
    const int shared_mem_size = THREAD_BLOCK_SIZE*sizeof(VectorVertexEmbedding<embedding_size+1>);
    //std::static_assert (shared_mem_size <= 49152);
    assert (shared_mem_size <= 49152);
    const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

    //assert (per_thread_shared_mem_size >= sizeof (VertexEmbedding));
    //per_thread_shared_mem_size = sizeof (VertexEmbedding);
    __shared__ SharedMemElem shared_buff[shared_mem_size];///sizeof (SharedMemElem)];

    /*SharedMemElem* local_shared_buff = &shared_buff[per_thread_shared_mem_size/sizeof(SharedMemElem)*threadIdx.x];
    memcpy (local_shared_buff, &embeddings[id], embedding_size);
    __syncthreads ();
    */
  #endif
#else 
#  if defined(ALL_THREAD_BLOCK_EMBEDDINGS_IN_SHARED_MEM_PER_VERTEX)
  const int shared_mem_size = THREAD_BLOCK_SIZE*sizeof(VectorVertexEmbedding<embedding_size>);
  //std::static_assert (shared_mem_size <= 49152);
  assert (shared_mem_size <= 49152);
  const int per_thread_shared_mem_size = shared_mem_size/THREAD_BLOCK_SIZE;

  //assert (per_thread_shared_mem_size >= sizeof (VertexEmbedding));
  //per_thread_shared_mem_size = sizeof (VertexEmbedding);
  __shared__ SharedMemElem shared_buff[shared_mem_size];

  SharedMemElem* local_shared_buff = &shared_buff[per_thread_shared_mem_size/sizeof(SharedMemElem)*threadIdx.x];
  memcpy (local_shared_buff, &embeddings[id], sizeof (VectorVertexEmbedding<embedding_size>));
  __syncthreads ();

#  else //#ifdef USE_EMBEDDING_IN_LOCAL_MEM
  unsigned char temp_buffer [sizeof(VectorVertexEmbedding<embedding_size>)];
#  endif
#endif
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  
  int start = id, end = id+1;
  //printf ("running id %d\n", id);
#ifndef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK 
  if (id >= n_embeddings)
      return;
#endif

  int thread_block_start_idx = blockIdx.x*blockDim.x * embedding_size;
  const bool enable_edge_pulling = false;
  for (int load = 0; load < embedding_size; load++) {
    //printf ("load %d embedding_size %d\n", load, embedding_size);
    int load_ids[2];
    
#if defined (EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK)
    load_ids[0] = thread_block_start_idx + threadIdx.x + load*blockDim.x;
    load_ids[1] = thread_block_start_idx + embedding_size*THREAD_BLOCK_SIZE - 1 - (threadIdx.x + load*blockDim.x);
    
    //blockIdx.x*blockDim.x + load*n_embeddings + (load+1)*blockDim.x - 1 - threadIdx.x;
    /*if (load == 0)
      printf ("P1 id: %d blockIdx: %d threadIdx: %d load_ids[0]: %d load: %d\n",
            id, blockIdx.x, threadIdx.x, load_ids[0], load);
            */
    if (load_ids[0] >= n_embeddings*embedding_size)
            return;
    /*if (load == 0)
      printf ("P2 id: %d blockIdx: %d threadIdx: %d load_ids[0]: %d load: %d\n",
            id, blockIdx.x, threadIdx.x, load_ids[0], load);
    */
#else
    load_ids[0] = id + load*n_embeddings;
    load_ids[1] = (load+1)*n_embeddings - 1 - id;
#endif
    //printf ("t-id %d ; load_ids[0] %d ; load_ids[1] %d\n", id, load_ids[0], load_ids[1]);
    int n_loads;
#if defined (EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK)
    if (enable_edge_pulling && threadIdx.x + load*blockDim.x < embedding_size*THREAD_BLOCK_SIZE/2 && 
        load_ids[1] < n_embeddings*embedding_size) {
#else
    if (enable_edge_pulling && id < n_embeddings/2) {
#endif

      //Vertices are sorted in increasing order. a vertex in first half will pull edges from 
      //vertex in second half.
      n_loads = 2;
    } else {
      n_loads = 1;
    }

    for (int i = 0; i < n_loads; i++) {
#if defined (EMBEDDING_IN_SHARED_MEM_PER_VERTEX)
      int emb_idx = embeddings_per_partitions[load_ids[i]].emb_idx;
      VectorVertexEmbedding<embedding_size>* embedding = &((VectorVertexEmbedding<embedding_size>*)&shared_buff[0])[threadIdx.x];
      memcpy (embedding, &embeddings[embeddings_per_partitions[load_ids[i]].emb_idx], sizeof (VectorVertexEmbedding<embedding_size>));
      //embedding->clear ();
      //vector_embedding_from_one_less_size (embeddings[embeddings_per_partitions[load_ids[i]].emb_idx], *embedding);
#else
#  if defined (ALL_THREAD_BLOCK_EMBEDDINGS_IN_SHARED_MEM_PER_VERTEX)
      int emb_idx = embeddings_per_partitions[load_ids[i]].emb_idx;
      VectorVertexEmbedding<embedding_size> const * embedding = &((VectorVertexEmbedding<embedding_size>*)&shared_buff[0])[emb_idx%256];
      //embedding->clear ();
      //vector_embedding_from_one_less_size (embeddings[emb_idx], *embedding);
#  else
      VectorVertexEmbedding<embedding_size>* embedding = (VectorVertexEmbedding<embedding_size>*)&temp_buffer[0];
      embedding->clear ();
      memcpy (embedding, &embeddings[embeddings_per_partitions[load_ids[i]].emb_idx], sizeof (VectorVertexEmbedding<embedding_size>));
#  endif
#endif

      int u = embedding->get_vertex (embeddings_per_partitions[load_ids[i]].vertex_idx);
      int start_edge_idx;
      int end_edge_idx;

      if (enable_edge_pulling && i == 1) {
        //Can only happen when vertex is in first half
        assert (n_loads == 2);
        int v = embeddings[embeddings_per_partitions[load_ids[0]].emb_idx].get_vertex (embeddings_per_partitions[load_ids[0]].vertex_idx);
        int first_n_edges = csr->get_end_edge_idx (v) - csr->get_start_edge_idx (v) + 1;
        int second_n_edges = csr->get_end_edge_idx (u) - csr->get_start_edge_idx (u) + 1;
        /*if (!(first_n_edges <= second_n_edges)) {
          printf ("id: %d blockIdx: %d threadIdx: %d load_ids[0]: %d load_idx[1]: %d first %d second %d\n",
          id, blockIdx.x, threadIdx.x, load_ids[0], load_ids[1], first_n_edges, second_n_edges);
        }*/
        assert (first_n_edges <= second_n_edges);
        int total_edges_to_process = (second_n_edges + first_n_edges)/2;
        int remaining_edges = total_edges_to_process - first_n_edges; //first n edges have already been done
        start_edge_idx = csr->get_end_edge_idx (u) - remaining_edges + 1;
        end_edge_idx = csr->get_end_edge_idx (u);
      } else {
        if (enable_edge_pulling && n_loads == 1 && load_ids[1] < n_embeddings*embedding_size) {
          //Can only happen when vertex is in second half
          int v = embeddings[embeddings_per_partitions[load_ids[1]].emb_idx].get_vertex (embeddings_per_partitions[load_ids[1]].vertex_idx);
          int first_n_edges = csr->get_end_edge_idx(u) - csr->get_start_edge_idx(u) + 1;
          int second_n_edges = csr->get_end_edge_idx(v) - csr->get_start_edge_idx(v) + 1;
          if (!(first_n_edges >= second_n_edges)) {
            printf ("id: %d blockIdx: %d threadIdx: %d load_ids[0]: %d load_idx[1]: %d first %d second %d\n",
            id, blockIdx.x, threadIdx.x, load_ids[0], load_ids[1], first_n_edges, second_n_edges);
          }
          assert (first_n_edges >= second_n_edges);
          int edges_to_process = ((second_n_edges + first_n_edges) % 2 == 0) ? (second_n_edges + first_n_edges)/2 : (second_n_edges + first_n_edges)/2+1;
          //int edges_to_process = (second_n_edges + first_n_edges)/2;
          start_edge_idx = csr->get_start_edge_idx (u);
          end_edge_idx = start_edge_idx + edges_to_process - 1;
        }
        else {
          //Can only happen when vertex is in first half
          start_edge_idx = csr->get_start_edge_idx(u);
          end_edge_idx = csr->get_end_edge_idx(u);
        }
      }

      for (int e = start_edge_idx; e <= end_edge_idx; e++) {
        
        int v = csr->get_edges () [e];

        if (is_embedding_canonical<embedding_size> (csr, embedding, v) && 
            ((VectorVertexEmbedding<embedding_size>*)embedding)->has (v) == false) { //TODO: Make both these checks in same loop
          VectorVertexEmbedding<embedding_size> const * extension = embedding;
          //extension->add_unsorted (v);
          
          if (clique_filter_vector_optimized (csr, extension, v)) {
            //VectorVertexEmbedding<embedding_size+1> extension = *embedding;
            //extension.add_last_in_sort_order ();
            //int o = atomicAdd(n_output,1);
            //int n = atomicAdd(n_next_step_1,1);
            
            if (only_copy_change) {
              int o = atomicAdd(n_output, 1);
              int n = atomicAdd(n_next_step_1, 1);
              int* new_embeddings = (int*) next_step_1;
              int* output = (int*) output_ptr;

              new_embeddings[2*n] = id;
              new_embeddings[2*n+1] = v;
              output[2*o] = id;
              output[2*o+1] = v;
            }
            else {
              int storage_id = *curr_step_storage_id;
              /*assert (! (*buff_1_status == BUFFER_STATUS::CPU_COPYING) && (*buff_2_status == BUFFER_STATUS::CPU_COPYING));
              assert (! (*buff_1_status == BUFFER_STATUS::GPU_USING) && (*buff_2_status == BUFFER_STATUS::GPU_USING));
              if ((*buff_1_status == BUFFER_STATUS::FREE) && (*buff_2_status == BUFFER_STATUS::FREE)) {
                storage_id = 0;
              } else {
                
              }*/
              
              const size_t max_n_embeddings = NEW_EMBEDDING_BUFFER_SIZE/sizeof (VectorVertexEmbedding<embedding_size+1>);
              //const int storage_id = 0;
              int n = 0;
              switch (storage_id) {
                case 0: {
                  n = atomicAdd(n_next_step_1, 1);
                  //Switch from buff1 to buff2
                  while (n >= max_n_embeddings) {//TODO: change it to do-while 
                    if (storage_id == 0) {
                      n = atomicSub (n_next_step_1, 1); //TODO: can remove that
                      while (*buff_2_status == BUFFER_STATUS::CPU_COPYING) {
                        /*unsigned long i = 0;
                        while (i <= (1UL<<30)) {
                          i++;
                        }*/
                      }
                      *curr_step_storage_id = 1;
                      storage_id = 1;
                      *buff_2_status = BUFFER_STATUS::GPU_USING;
                      n = atomicAdd(n_next_step_2, 1);
                    } else {
                      n = atomicSub (n_next_step_2, 1); //TODO: can remove that
                      while (*buff_1_status == BUFFER_STATUS::CPU_COPYING) {
                        /*unsigned long i = 0;
                        while (i <= (1UL<<30)) {
                          i++;
                        }*/
                      }
                      *curr_step_storage_id = 0;
                      storage_id = 0;
                      
                      *buff_1_status = BUFFER_STATUS::GPU_USING;
                      n = atomicAdd(n_next_step_1, 1);
                    }
                  }
                  
                  if (storage_id == 1) {
                    //n = atomicAdd (n_next_step_2, 1);
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_2;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    #ifdef ADD_TO_OUTPUT
                    int o = atomicAdd(n_output, 1);
                    memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    #endif
                    //memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    vector_embedding_from_one_less_size (*extension, new_embeddings[n]);
                    new_embeddings[n].add_unsorted (v);
                    if (atomicAdd ((int*)n_next_step_2_done, 1) >= max_n_embeddings) {
                      *buff_2_status = BUFFER_STATUS::CPU_COPYING;
                    }
                  } else {
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_1;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    #ifdef ADD_TO_OUTPUT
                    int o = atomicAdd(n_output, 1);
                    memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    #endif
                    //memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    vector_embedding_from_one_less_size (*extension, new_embeddings[n]);
                    new_embeddings[n].add_unsorted (v);
                    if (atomicAdd ((int*)n_next_step_1_done, 1) >= max_n_embeddings) {
                      *buff_1_status = BUFFER_STATUS::CPU_COPYING;
                    }
                  }
                  break;
                }

                case 1: {
                  n = atomicAdd(n_next_step_2, 1);

                  while (n >= max_n_embeddings) {//TODO: change it to do-while 
                    if (storage_id == 0) {
                      n = atomicSub (n_next_step_1, 1); //TODO: can remove that
                      
                      while (*buff_2_status == BUFFER_STATUS::CPU_COPYING) {
                        /*unsigned long i = 0;
                        while (i <= (1UL<<30)) {
                          i++;
                        }*/
                      }
                      *buff_2_status = BUFFER_STATUS::GPU_USING;
                      *curr_step_storage_id = 1;
                      storage_id = 1;
                      //*buff_1_status = BUFFER_STATUS::CPU_COPYING;
                      n = atomicAdd(n_next_step_2, 1);
                    } else {
                      n = atomicSub (n_next_step_2, 1); //TODO: can remove that
                      while (*buff_1_status == BUFFER_STATUS::CPU_COPYING) {
                        /*unsigned long i = 0;
                        while (i <= (1UL<<30)) {
                          i++;
                        }*/
                      }
                      *curr_step_storage_id = 0;
                      storage_id = 0;
                      //*buff_2_status = BUFFER_STATUS::CPU_COPYING;
                      *buff_1_status = BUFFER_STATUS::GPU_USING;
                      n = atomicAdd(n_next_step_1, 1);
                    }
                  }
                  
                  if (storage_id == 1) {
                    //n = atomicAdd (n_next_step_2, 1);
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_2;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    #ifdef ADD_TO_OUTPUT
                    int o = atomicAdd(n_output, 1);
                    memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    #endif
                    //memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    vector_embedding_from_one_less_size (*extension, new_embeddings[n]);
                    new_embeddings[n].add_unsorted (v);
                    if (atomicAdd ((int*)n_next_step_2_done, 1) >= max_n_embeddings) {
                      *buff_2_status = BUFFER_STATUS::CPU_COPYING;
                    }
                  } else {
                    VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_1;
                    VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                    #ifdef ADD_TO_OUTPUT
                    int o = atomicAdd(n_output, 1);
                    memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    #endif
                    //memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                    vector_embedding_from_one_less_size (*extension, new_embeddings[n]);
                    new_embeddings[n].add_unsorted (v);
                    if (atomicAdd ((int*)n_next_step_1_done, 1) >= max_n_embeddings) {
                      *buff_1_status = BUFFER_STATUS::CPU_COPYING;
                    }
                  }

                  // if (n >= max_n_embeddings) {
                  //   //Switch from buff2 to buff1
                  //   atomicSub (n_next_step_2, 1); //TODO: can remove that
                  //   *curr_step_storage_id = 0;
                  //   *buff_2_status = BUFFER_STATUS::CPU_COPYING;
                  //   while (*buff_1_status == BUFFER_STATUS::CPU_COPYING) {
                  //     /*unsigned long i = 0;
                  //     while (i <= (1UL<<30)) {
                  //       i++;
                  //     }*/
                  //   }
                  //   *buff_1_status = BUFFER_STATUS::GPU_USING;
                  //   n = atomicAdd (n_next_step_1, 1);
                  //   VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_1;
                  //   VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                  //   #ifdef ADD_TO_OUTPUT
                  //   int o = atomicAdd(n_output, 1);
                  //   memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  //   #endif
                  //   //memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  //   vector_embedding_from_one_less_size (*extension, new_embeddings[n]);
                  //   new_embeddings[n].add_unsorted (v);
                  // } else {
                  //   VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step_2;
                  //   VectorVertexEmbedding<embedding_size+1>* output = (VectorVertexEmbedding<embedding_size+1>*)output_ptr;
                  //   #ifdef ADD_TO_OUTPUT
                  //   int o = atomicAdd(n_output, 1);
                  //   memcpy (&output[o], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  //   #endif
                  //   //memcpy (&new_embeddings[n], extension, sizeof (VectorVertexEmbedding<embedding_size+1>));
                  //   vector_embedding_from_one_less_size (*extension, new_embeddings[n]);
                  //   new_embeddings[n].add_unsorted (v);
                  // }
                }
              }
            }
            //output[o].add_last_in_sort_order ();
            //new_embeddings[n].add_last_in_sort_order ();
          }
          //extension->remove_last ();
        }
      }
    }
  }

  //printf ("embeddings generated [1000, 2000)= %d and [2000, 3000) = %d\n", q[0], q[1]);
  //printf ("embeddings at i = 1500: %d\n", q[2]);
  //for (int i = 100; i < 1000; i++) {
  //  printf ("embeddings at i = %d %d\n", i, q[i]);
  //}
}


void print_embedding (BitVectorVertexEmbedding embedding, std::ostream& os)
{
  os << "[";
  for (int u = 0; u < N; u++) {
    if (embedding.test(u)) {
      os << u << ", ";
    }
  }
  os << "]";
}

__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

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

enum EmbeddingType {
  VectorVertex,
  BitVector,
};

class Partition
{
private:
  int start_vertex;
  int end_vertex;
  int id;
  int start_idx;
  int end_idx;

public:
  Partition (int s, int e) : start_vertex (s), end_vertex (e)
  {}
  
};

size_t get_emb_per_parts_size (size_t n_embeddings, int vertices_per_embedding) 
{
  return vertices_per_embedding * n_embeddings * sizeof(PairEmbIdxVertexIdx);
}

template <uint size>
void collect_vertex_embeddings_in_partitions (CSR* csr, VectorVertexEmbedding<size>* embeddings, size_t start_idx, 
                                              size_t n_embeddings,
                                              std::vector<std::vector<int>>& partitions, std::vector<int>& vertex_to_partition,
                                              PairEmbIdxVertexIdx* embs_for_parts, size_t embs_for_parts_size)
{
  std::vector<int> embs_per_parts_size(partitions.size (), 0);
  
  for (int i = start_idx; i < start_idx + n_embeddings; i++) {
    VectorVertexEmbedding<size>& emb = embeddings[i];
    //print_embedding<size> (&emb);
    //std::cout << std::endl;
    for (int v_i = 0; v_i < size; v_i++) {
      int v = emb.get_vertex (v_i);
      int part_idx = vertex_to_partition [v];
      embs_per_parts_size[part_idx]++;
    }
  }

  /*for (int i = 0; i < embs_per_parts_size.size (); i++) {
    std::cout << i << ":" << embs_per_parts_size[i] << std::endl;
  }*/
  std::vector<int> parts_pushed_idx (partitions.size (), 0);
  std::vector<int> part_idx_in_embs_for_parts (partitions.size (), 0);
  for (int i = 1; i < embs_per_parts_size.size (); i++) {
    part_idx_in_embs_for_parts[i] = part_idx_in_embs_for_parts[i-1] + embs_per_parts_size[i-1];
  }

  /*for (int i = 0; i < embs_per_parts_size.size (); i++) {
    std::cout << "part_idx_in_embs_for_parts " << i << ":" << part_idx_in_embs_for_parts[i] << std::endl;
  }*/

  //std::cout << "embs_for_parts_size " << embs_for_parts_size <<  ":" << get_emb_per_parts_size (n_embeddings, size) << std::endl;
  assert (get_emb_per_parts_size (n_embeddings, size) == embs_for_parts_size);

  for (int i = start_idx; i < start_idx + n_embeddings; i++) {
    VectorVertexEmbedding<size>& emb = embeddings[i];
    for (int v_i = 0; v_i < size; v_i++) {
      int v = emb.get_vertex (v_i);
      int part_idx = vertex_to_partition [v];
      embs_for_parts[part_idx_in_embs_for_parts[part_idx] + parts_pushed_idx[part_idx]] = PairEmbIdxVertexIdx (i, v_i);
      parts_pushed_idx[part_idx]++;
    }
  }
}

template <uint size>
void collect_vertex_embeddings_in_partitions_per_threadblock (CSR* csr, VectorVertexEmbedding<size>* embeddings, size_t n_embeddings,
                                                              std::vector<std::vector<int>>& partitions, std::vector<int>& vertex_to_partition,
                                                              PairEmbIdxVertexIdx* embs_for_parts, size_t embs_for_parts_size)
{
  int n_threadblocks = n_embeddings%THREAD_BLOCK_SIZE == 0 ? n_embeddings/THREAD_BLOCK_SIZE : n_embeddings/THREAD_BLOCK_SIZE + 1;
  size_t embs_for_parts_size_per_threadblock = get_emb_per_parts_size (THREAD_BLOCK_SIZE, size);
  size_t embs_for_parts_processed = 0;
  size_t n_embeddings_processed = 0;
  for (int i = 0; i < ((n_embeddings%THREAD_BLOCK_SIZE == 0) ? n_threadblocks : n_threadblocks - 1); i++) {
    collect_vertex_embeddings_in_partitions (csr, embeddings, n_embeddings_processed, 
                                             THREAD_BLOCK_SIZE, partitions, vertex_to_partition,
                                             embs_for_parts + embs_for_parts_processed, 
                                             embs_for_parts_size_per_threadblock);
    //std::cout << "embs_for_parts_processed " << embs_for_parts_processed << std::endl;
    embs_for_parts_processed += embs_for_parts_size_per_threadblock/sizeof (PairEmbIdxVertexIdx);
    n_embeddings_processed += THREAD_BLOCK_SIZE;    
  }

  if (n_embeddings%THREAD_BLOCK_SIZE != 0) {
    size_t remaining = n_embeddings - n_embeddings_processed;
    collect_vertex_embeddings_in_partitions (csr, embeddings, n_embeddings_processed, 
                                             remaining, partitions, vertex_to_partition,
                                             embs_for_parts + embs_for_parts_processed, 
                                             get_emb_per_parts_size (remaining, size));
    embs_for_parts_processed += embs_for_parts_size_per_threadblock;
  }


  //for (int i = 0; i < n_embeddings*size; i++) {
  //  PairEmbIdxVertexIdx p = embs_for_parts[i];
  //  printf ("%d: [%d, %d]:\n", i, p.emb_idx, p.vertex_idx);
    //VectorVertexEmbedding<2>* emb = &((VectorVertexEmbedding<2>*)input_embeddings_ptr)[p.emb_idx];
    //print_embedding<2> (emb);
    //std::cout << std::endl;
  //}
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

#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
  std::cout << "Sorting " << std::endl;
  std::vector <Vertex> old_vertices = vertices;
  std::vector <int> new_to_old_vertex_ids;
  std::vector <int> old_to_new_vertex_ids;
  for (size_t i = 0; i < vertices.size (); i++) {
    new_to_old_vertex_ids.push_back (vertices[i].get_id ());
    old_to_new_vertex_ids.push_back (vertices[i].get_id ());
  }

  std::sort (vertices.begin (), vertices.end (), Vertex::compare_vertex);
  assert (Vertex::compare_vertex(vertices[0], vertices[vertices.size () - 1]));
  for (size_t i = 0; i < vertices.size (); i++) {
    int old_id = vertices[i].get_id ();
    int new_id = i;
    vertices[i].set_id (new_id);
    new_to_old_vertex_ids[new_id] = old_id;
    old_to_new_vertex_ids[old_id] = new_id;
  }

  for (size_t i = 0; i < vertices.size (); i++) {
    std::vector <int>& edges = vertices[i].get_edges ();
    for (size_t j = 0; j < edges.size (); j++) {
      edges[j] = old_to_new_vertex_ids[edges[j]];
    }
  }

  int partition_size = 1;
  assert (vertices.size () % partition_size == 0);
  int num_partition = vertices.size ()/partition_size;
  std::vector <std::vector <int> > partitions;
  std::vector <int> vertex_to_partition (vertices.size (), 0);

  for (int i = 0; i < vertices.size (); i++) {
    partitions.push_back (std::vector<int> {i});
    vertex_to_partition[i] = i;
  }
#endif

  Graph graph (vertices, n_edges);

  CSR* csr = new CSR(N, N_EDGES);
  std::cout << "sizeof(CSR)"<< sizeof(CSR)<<std::endl;
  std::cout <<"sizeof(VertexEmbedding)" << sizeof(BitVectorVertexEmbedding) << std::endl;
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

  std::vector<VectorVertexEmbedding<0>> initial_embeddings = get_initial_embedding_vector (csr);
  std::vector<VectorVertexEmbedding<1>> output_1;
  std::vector<VectorVertexEmbedding<2>> output_2;
  std::vector<VectorVertexEmbedding<3>> output_3;
  std::vector<VectorVertexEmbedding<4>> output_4;
  std::vector<VectorVertexEmbedding<5>> output_5;
  std::vector<VectorVertexEmbedding<6>> output_6;
  std::vector<VectorVertexEmbedding<7>> output_7;
  std::vector<VectorVertexEmbedding<8>> output_8;
  std::vector<std::pair<void*, size_t>> embeddings;

  //filter = clique_filter;
  //process = clique_process;
  size_t new_embeddings_size = 0;
  int iter = 0;
  {
    std::vector<VectorVertexEmbedding<1>> new_embeddings;
    run_single_step_initial_vector (&initial_embeddings[0], 1, csr, 
                                    output_1, new_embeddings);
    new_embeddings_size = new_embeddings.size ();
    embeddings.push_back (std::make_pair (malloc (sizeof (VectorVertexEmbedding<1>)*new_embeddings_size), new_embeddings_size));
    for (int i = 0; i < new_embeddings_size; i++) {
      ((VectorVertexEmbedding<1>*)embeddings[0].first)[i] = new_embeddings[i];
      int v = ((VectorVertexEmbedding<1>*)embeddings[0].first)[i].get_vertex (0);
      assert (v >= 0);
    }
  }

  iter = 1;
  double total_stream_time = 0;
  size_t global_mem_size = 15*1024*1024*1024UL;
#define PINNED_MEMORY
#ifdef PINNED_MEMORY
  char* global_mem_ptr;
  cudaError_t malloc_error = cudaMallocHost ((void**)&global_mem_ptr, global_mem_size);
  assert (malloc_error == cudaSuccess);
#else
  char* global_mem_ptr = new char[global_mem_size];
#endif

  const size_t max_embedding_size_per_iter = (12000000/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
  double_t kernelTotalTime = 0.0;
  for (iter; iter < 8 && new_embeddings_size > 0; iter++) {
    std::cout << "iter " << iter << " embeddings " << new_embeddings_size << std::endl;
    
    size_t remaining_embeddings = new_embeddings_size;
    size_t n_embeddings = new_embeddings_size;
    #ifdef DEBUG
      memset (global_mem_ptr, 0, global_mem_size);
    #endif

    //Copy all embeddings to global memory
    size_t embedding_size = 0;
    size_t new_embedding_size = 0;
    size_t global_mem_iter = 0;
    switch (iter) {
      case 1: {
        embedding_size = sizeof (VectorVertexEmbedding<1>);
        new_embedding_size = sizeof (VectorVertexEmbedding<2>);
        
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<1>*)global_mem_ptr)[global_mem_iter] = ((VectorVertexEmbedding<1>*) iter.first)[i];
            int v = ((VectorVertexEmbedding<1>*)global_mem_ptr)[global_mem_iter].get_vertex (0);
            global_mem_iter++;
            assert (v >= 0);
          }
        }
        break;
      }      
      case 2: {
        embedding_size = sizeof (VectorVertexEmbedding<2>);
        new_embedding_size = sizeof (VectorVertexEmbedding<3>);
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<2>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<2>*)iter.first)[i];
          }
        }
        break;
      }
      
      case 3: {
        embedding_size = sizeof (VectorVertexEmbedding<3>);
        new_embedding_size = sizeof (VectorVertexEmbedding<4>);
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<3>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<3>*)iter.first)[i];
          }
        }
        break;
      }
      
      case 4: {
          embedding_size = sizeof (VectorVertexEmbedding<4>);
          new_embedding_size = sizeof (VectorVertexEmbedding<5>);
          for (auto iter: embeddings) {
            for (int i = 0; i < iter.second; i++) {
              ((VectorVertexEmbedding<4>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<4>*)iter.first)[i];
            }
          }
        break;
      }
      case 5: {
        embedding_size = sizeof (VectorVertexEmbedding<5>);
        new_embedding_size = sizeof (VectorVertexEmbedding<6>);
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<5>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<5>*)iter.first)[i];
          }
        }
        break;
      }
      case 6: {
        embedding_size = sizeof (VectorVertexEmbedding<6>);
        new_embedding_size = sizeof (VectorVertexEmbedding<7>);
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<6>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<6>*)iter.first)[i];
          }
        }
        break;
      }
      case 7: {
        embedding_size = sizeof (VectorVertexEmbedding<7>);
        new_embedding_size = sizeof (VectorVertexEmbedding<8>);
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<7>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<7>*)iter.first)[i];
          }
        }
        break;
      }
      case 8: {
        embedding_size = sizeof (VectorVertexEmbedding<8>);
        new_embedding_size = sizeof (VectorVertexEmbedding<9>);
        for (auto iter: embeddings) {
          for (int i = 0; i < iter.second; i++) {
            ((VectorVertexEmbedding<8>*)global_mem_ptr)[global_mem_iter++] = ((VectorVertexEmbedding<8>*)iter.first)[i];
          }
        }
        break;
      }
    }

    //delete embeddings too because there is a memory leak?
    if (iter > 1) {
      for (auto iter: embeddings) {
        free(iter.first);
      }
    }

    embeddings.clear ();
    std::cout << "Copying to global_mem_ptr done. global mem used " << global_mem_iter*embedding_size << std::endl;
    
    void* embeddings_ptr = global_mem_ptr;

    size_t n_next_step_embeddings = 0;
    n_embeddings = 0;

    void* orig_new_embeddings_ptr = ((char*)global_mem_ptr) + (global_mem_iter)*(new_embedding_size); //Size of next embedding will be one more
    //size_t max_embeddings = 40000000; //There is something with this value which makes it perform better, may be alignment?
    size_t max_embeddings = NEW_EMBEDDING_BUFFER_SIZE/(new_embedding_size);
    printf ("new_embedding_size %ld\n", new_embedding_size);
    void* orig_output_ptr = (char*)orig_new_embeddings_ptr + (max_embeddings)*(new_embedding_size);
    size_t orig_outputs_size = max_embeddings*(new_embedding_size);
    cudaError_t error;
    double stream_time_1 = convertTimeValToDouble (getTimeOfDay ());
    
    while (remaining_embeddings != 0) {
      n_embeddings = (n_embeddings/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
      std::cout << "iter " << iter << " n_embeddings " << new_embeddings_size << " remaining_embeddings " << remaining_embeddings << std::endl;
      embeddings_ptr = ((char*)global_mem_ptr) + embedding_size*(new_embeddings_size - remaining_embeddings);
      //printf ("embeddings_ptr %x\n", embeddings_ptr);
      n_embeddings = std::min (remaining_embeddings, max_embedding_size_per_iter);

      remaining_embeddings -= n_embeddings;
      //n_embeddings = (n_embeddings/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
      
      const int N_STREAMS = 1;
      //assert (max_embeddings/N_STREAMS >= 30000000);
      int only_copy_change = 0;
      assert (only_copy_change == 0); //TODO: Streams with only copy change
      void* new_embeddings_ptr[N_STREAMS];
      assert (max_embeddings%N_STREAMS == 0);
      for (int i = 0; i < N_STREAMS; i++) {
        new_embeddings_ptr[i] = (char*)orig_new_embeddings_ptr + i*new_embedding_size*max_embeddings/N_STREAMS;
      }

      void* output_ptr[N_STREAMS];
      for (int i = 0; i < N_STREAMS; i++) {
        output_ptr[i] = (char*)orig_output_ptr + i*orig_outputs_size/N_STREAMS;
      }

      int n_new_embeddings_1[N_STREAMS] = {0};
      int n_new_embeddings_2[N_STREAMS] = {0};
      int n_output[N_STREAMS] = {0};
      int n_output_1[N_STREAMS] = {0};
      int curr_new_embeddings_idx[N_STREAMS] = {0};
      char* device_embeddings[N_STREAMS];
      char *device_new_embeddings_1[N_STREAMS];
      char *device_new_embeddings_2[N_STREAMS];
      BUFFER_STATUS *device_new_embeddings_1_status[N_STREAMS];
      BUFFER_STATUS *device_new_embeddings_2_status[N_STREAMS];
      int* device_n_embeddings_1[N_STREAMS];
      int* device_n_embeddings_2[N_STREAMS];
      int* device_curr_new_embeddings_idx[N_STREAMS];
      char *device_outputs[N_STREAMS];
      int* device_n_outputs[N_STREAMS];
      int* device_n_outputs_1[N_STREAMS];
      BUFFER_STATUS new_embeddings_1_status[N_STREAMS];
      BUFFER_STATUS new_embeddings_2_status[N_STREAMS];
      char *device_embeddings_per_partitions[N_STREAMS];
      for (int i = 0; i < N_STREAMS; i++) {
        new_embeddings_1_status[i] = BUFFER_STATUS::FREE;
        new_embeddings_2_status[i] = BUFFER_STATUS::FREE;
      }
      
      CSR* device_csr[N_STREAMS];
      
      assert (N_STREAMS >= 1);

      cudaStream_t streams[N_STREAMS];
      for (int i = 0; i < N_STREAMS; i++) {
        cudaError_t err = cudaStreamCreate (&streams[i]);
        assert (err == cudaSuccess);
      }

      const bool stream_synchronize = true;
      size_t per_stream_embeddings_done = 0;
      size_t per_stream_n_embeddings = n_embeddings/N_STREAMS;
      assert (N_STREAMS == 1);
      for (int i = 0; i < N_STREAMS; i++) {
        const bool unified_mem = false;
        if (false && iter == 2) {
          if (i == 0) {
            per_stream_n_embeddings = (n_embeddings*9)/10;
          } else {
            per_stream_n_embeddings = (n_embeddings*1)/10;
          }
        }

#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
        assert (N_STREAMS == 1);
        char* embeddings_per_partitions_ptr = (char*)orig_output_ptr + orig_outputs_size;
        size_t embeddings_per_partitions_ptr_size = get_emb_per_parts_size (per_stream_n_embeddings, iter);
        assert ((size_t)embeddings_per_partitions_ptr + embeddings_per_partitions_ptr_size < (size_t)global_mem_ptr + global_mem_size);
        std::cout << "embeddings_per_partitions_ptr_size: " << embeddings_per_partitions_ptr_size << std::endl;
        char* input_embeddings_ptr = (char*)embeddings_ptr + per_stream_embeddings_done*embedding_size;
        switch (iter) {
          case 1: {
#   ifdef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
            collect_vertex_embeddings_in_partitions_per_threadblock (csr, (VectorVertexEmbedding<1>*)input_embeddings_ptr, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                                    (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   else
            collect_vertex_embeddings_in_partitions (csr, (VectorVertexEmbedding<1>*)input_embeddings_ptr, 0, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                     (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   endif
            break;
          }
          case 2: {
#   ifdef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
            collect_vertex_embeddings_in_partitions_per_threadblock (csr, (VectorVertexEmbedding<2>*)input_embeddings_ptr, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                                    (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   else
            collect_vertex_embeddings_in_partitions (csr, (VectorVertexEmbedding<2>*)input_embeddings_ptr, 0, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                     (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   endif
            std::cout << "elements in embs_per_parts: " << per_stream_n_embeddings*iter << std::endl;
            
            for (int i = 0; i < per_stream_n_embeddings*iter; i++) {
              //PairEmbIdxVertexIdx p = ((PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr)[i];
              //printf ("%d: [%d, %d]:", i, p.emb_idx, p.vertex_idx);
              //VectorVertexEmbedding<2>* emb = &((VectorVertexEmbedding<2>*)input_embeddings_ptr)[p.emb_idx];
              //print_embedding<2> (emb);
              //std::cout << std::endl;
            }
            break;
          } 
          case 3: {
#   ifdef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
            collect_vertex_embeddings_in_partitions_per_threadblock (csr, (VectorVertexEmbedding<3>*)input_embeddings_ptr, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                                    (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   else
            collect_vertex_embeddings_in_partitions (csr, (VectorVertexEmbedding<3>*)input_embeddings_ptr, 0, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                     (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   endif
            std::cout << "iter = 3 elements in embs_per_parts: " << per_stream_n_embeddings*iter << std::endl;
            
            for (int i = 0; i < per_stream_n_embeddings*iter; i++) {
              PairEmbIdxVertexIdx p = ((PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr)[i];
              if (i == 1416192) {
                std::cout << "vertex_idx: " << p.vertex_idx << std::endl;
              }
              assert (p.vertex_idx < iter);
              //printf ("%d: [%d, %d]:", i, p.emb_idx, p.vertex_idx);
              //VectorVertexEmbedding<3>* emb = &((VectorVertexEmbedding<2>*)input_embeddings_ptr)[p.emb_idx];
              //print_embedding<2> (emb);
              //std::cout << std::endl;
            }
            
            break;
          }
          case 4: {
#   ifdef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
            collect_vertex_embeddings_in_partitions_per_threadblock (csr, (VectorVertexEmbedding<4>*)input_embeddings_ptr, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                                    (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   else
            collect_vertex_embeddings_in_partitions (csr, (VectorVertexEmbedding<4>*)input_embeddings_ptr, 0, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                     (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   endif
            std::cout << "elements in embs_per_parts: " << per_stream_n_embeddings*iter << std::endl;
            
            for (int i = 0; i < per_stream_n_embeddings*iter; i++) {
              //PairEmbIdxVertexIdx p = ((PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr)[i];
              //printf ("%d: [%d, %d]:", i, p.emb_idx, p.vertex_idx);
              //VectorVertexEmbedding<2>* emb = &((VectorVertexEmbedding<2>*)input_embeddings_ptr)[p.emb_idx];
              //print_embedding<2> (emb);
              //std::cout << std::endl;
            }
            break;
          }
          case 5: {
#   ifdef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
            collect_vertex_embeddings_in_partitions_per_threadblock (csr, (VectorVertexEmbedding<5>*)input_embeddings_ptr, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                                    (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   else
            collect_vertex_embeddings_in_partitions (csr, (VectorVertexEmbedding<5>*)input_embeddings_ptr, 0, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                     (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   endif
            std::cout << "elements in embs_per_parts: " << per_stream_n_embeddings*iter << std::endl;
            
            for (int i = 0; i < per_stream_n_embeddings*iter; i++) {
              //PairEmbIdxVertexIdx p = ((PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr)[i];
              //printf ("%d: [%d, %d]:", i, p.emb_idx, p.vertex_idx);
              //VectorVertexEmbedding<2>* emb = &((VectorVertexEmbedding<2>*)input_embeddings_ptr)[p.emb_idx];
              //print_embedding<2> (emb);
              //std::cout << std::endl;
            }
            break;
          }
          case 6: {
#   ifdef EMBEDDING_PER_PARTITIONS_IN_THREADBLOCK
            collect_vertex_embeddings_in_partitions_per_threadblock (csr, (VectorVertexEmbedding<6>*)input_embeddings_ptr, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                                    (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   else
            collect_vertex_embeddings_in_partitions (csr, (VectorVertexEmbedding<6>*)input_embeddings_ptr, 0, per_stream_n_embeddings, partitions, vertex_to_partition, 
                                                     (PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr, embeddings_per_partitions_ptr_size);
#   endif
            std::cout << "elements in embs_per_parts: " << per_stream_n_embeddings*iter << std::endl;
            
            for (int i = 0; i < per_stream_n_embeddings*iter; i++) {
              //PairEmbIdxVertexIdx p = ((PairEmbIdxVertexIdx*)embeddings_per_partitions_ptr)[i];
              //printf ("%d: [%d, %d]:", i, p.emb_idx, p.vertex_idx);
              //VectorVertexEmbedding<2>* emb = &((VectorVertexEmbedding<2>*)input_embeddings_ptr)[p.emb_idx];
              //print_embedding<2> (emb);
              //std::cout << std::endl;
            }
            break;
          }
          default: {
            assert (false);
          }
        }
#endif

        if (unified_mem == true) {
          //cudaMallocManaged (embeddings_ptr, n_embeddings*embedding_size);
          //device_embeddings = (char*)embeddings_ptr;
          assert(false);
        } else {
          cudaMalloc (&device_embeddings[i], per_stream_n_embeddings*embedding_size);
          cudaMemcpyAsync (device_embeddings[i], (char*)embeddings_ptr + per_stream_embeddings_done*embedding_size,
                           per_stream_n_embeddings*embedding_size, cudaMemcpyHostToDevice, streams[i]);
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
          cudaMalloc (&device_embeddings_per_partitions[i], embeddings_per_partitions_ptr_size);
          cudaMemcpyAsync (device_embeddings_per_partitions[i], 
                           (char*)embeddings_per_partitions_ptr, 
                           embeddings_per_partitions_ptr_size, cudaMemcpyHostToDevice, streams[i]);                           
          {
            cudaError_t error = cudaGetLastError ();
            if (error != cudaSuccess) {
              const char* error_string = cudaGetErrorString (error);
              std::cout << __LINE__<<" embeddings_per_partitions Cuda host to device copy error " << error_string << std::endl;
            } else {
              std::cout << "Cuda host to device copy success " << std::endl;
            }
          }
#endif
        }

        cudaMalloc (&device_new_embeddings_1[i], max_embeddings/N_STREAMS*(new_embedding_size));
        cudaMalloc (&device_new_embeddings_2[i], max_embeddings/N_STREAMS*(new_embedding_size));
        cudaMalloc (&device_n_embeddings_1[i], sizeof(int));
        cudaMalloc (&device_outputs[i], orig_outputs_size/N_STREAMS);
        cudaMalloc (&device_n_embeddings_1[i], sizeof (int));
        cudaMalloc (&device_n_embeddings_2[i], sizeof (int));
        cudaMalloc (&device_n_outputs[i], sizeof (int));
        cudaMalloc (&device_n_outputs_1[i], sizeof (int));
        cudaMalloc (&device_csr[i], sizeof(CSR)); //TODO: Remove copying CSR graph again and again
        cudaMalloc (&device_curr_new_embeddings_idx[i], sizeof (int));
        cudaMalloc (&device_new_embeddings_1_status[i], sizeof (BUFFER_STATUS));
        cudaMalloc (&device_new_embeddings_2_status[i], sizeof (BUFFER_STATUS));
        assert (N_STREAMS == 1); //TODO: for more than 1 N_STREAMS
        cudaMemcpyAsync (device_new_embeddings_1_status[i], &new_embeddings_1_status[i], 
                         sizeof (new_embeddings_1_status[i]), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync (device_new_embeddings_2_status[i], &new_embeddings_2_status[i], 
                         sizeof (new_embeddings_2_status[i]), cudaMemcpyHostToDevice, streams[i]);
        {
          cudaError_t error = cudaGetLastError ();
          if (error != cudaSuccess) {
            const char* error_string = cudaGetErrorString (error);
            std::cout << __LINE__<<" Cuda host to device copy error " << error_string << std::endl;
          } else {
            std::cout << __LINE__<< " Cuda host to device copy success " << std::endl;
          }
        }
        cudaMemcpyAsync (device_n_embeddings_1[i], &n_new_embeddings_1[i],
                         sizeof (n_new_embeddings_1[i]), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync (device_n_outputs[i], &n_output[i], sizeof (n_output[i]),
                         cudaMemcpyHostToDevice, streams[i]);
        
        {
          cudaError_t error = cudaGetLastError ();
          if (error != cudaSuccess) {
            const char* error_string = cudaGetErrorString (error);
            std::cout << __LINE__<<" Cuda host to device copy error " << error_string << std::endl;
          } else {
            std::cout << __LINE__<< " Cuda host to device copy success " << std::endl;
          }
        }                         
        
        assert (cudaMemcpyAsync (device_n_embeddings_2[i], &n_new_embeddings_2[i],
                         sizeof (n_new_embeddings_2[i]), cudaMemcpyHostToDevice, streams[i]) == cudaSuccess);
        cudaMemcpyAsync (device_n_outputs_1[i], &n_output_1[i], sizeof (n_output_1[i]),
                         cudaMemcpyHostToDevice, streams[i]);
        
        {
          cudaError_t error = cudaGetLastError ();
          if (error != cudaSuccess) {
            const char* error_string = cudaGetErrorString (error);
            std::cout << __LINE__<<" Cuda host to device copy error " << error_string << std::endl;
          } else {
            std::cout << __LINE__<<" Cuda host to device copy success " << std::endl;
          }
        }
        cudaMemcpyAsync (device_csr[i], csr, sizeof (CSR), cudaMemcpyHostToDevice, streams[i]);
        
        cudaMemcpyAsync (device_curr_new_embeddings_idx[i], &curr_new_embeddings_idx[i], 
                         sizeof (curr_new_embeddings_idx[i]), cudaMemcpyHostToDevice,  
                         streams[i]);
        
        cudaError_t error = cudaGetLastError ();
        if (error != cudaSuccess) {
          const char* error_string = cudaGetErrorString (error);
          std::cout << __LINE__<<" Cuda host to device copy error " << error_string << std::endl;
        } else {
          std::cout << __LINE__<<" Cuda host to device copy success " << std::endl;
        }

        per_stream_embeddings_done += per_stream_n_embeddings;
      }

      for (int i = 0; i < N_STREAMS; i++) {
        double t1 = convertTimeValToDouble (getTimeOfDay ());
        if (stream_synchronize)
          cudaStreamSynchronize (streams[i]);
        
        std::cout << "starting kernel with n_embeddings: " << per_stream_n_embeddings ;
    #ifdef USE_FIXED_THREADS
        //std::cout << " threads: " << MAX_CUDA_THREADS/THREAD_BLOCK_SIZE << std::endl;
        int thread_blocks = MAX_CUDA_THREADS/THREAD_BLOCK_SIZE;
    #else
        #ifdef PROCESS_EMBEDDINGS_PER_VERTEX
          int thread_blocks = (per_stream_n_embeddings%THREAD_BLOCK_SIZE != 0) ? ((per_stream_n_embeddings)/THREAD_BLOCK_SIZE+1) : (per_stream_n_embeddings)/THREAD_BLOCK_SIZE;
        #else
          int thread_blocks = (per_stream_n_embeddings%THREAD_BLOCK_SIZE != 0) ? (per_stream_n_embeddings/THREAD_BLOCK_SIZE+1) : per_stream_n_embeddings/THREAD_BLOCK_SIZE;
        #endif
    #endif
        std::cout << " threads: " << thread_blocks << std::endl;
        int* device_n_next_step_1_done;
        int* device_n_next_step_2_done;
        cudaMalloc (&device_n_next_step_1_done, sizeof(int));
        cudaMalloc (&device_n_next_step_2_done, sizeof(int));
        int qq = 0;
        cudaMemcpyAsync (device_n_next_step_1_done, &qq, 
          sizeof (qq), cudaMemcpyHostToDevice, streams[i]);
          cudaMemcpyAsync (device_n_next_step_2_done, &qq, 
            sizeof (qq), cudaMemcpyHostToDevice, streams[i]);
        switch (iter) {
          case 1: {
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
            run_single_step_vectorvertex_embedding_per_vertex <1><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
              device_embeddings_per_partitions[i], device_outputs[i], device_n_outputs[i],
              device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
              device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
              device_curr_new_embeddings_idx[i], 
              device_n_next_step_1_done,
              device_n_next_step_2_done,
              only_copy_change);
#else
            run_single_step_vectorvertex_embedding<1><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
                                  device_curr_new_embeddings_idx[i], only_copy_change);
#endif                                  
            break;
          }
          case 2: {
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
            run_single_step_vectorvertex_embedding_per_vertex <2><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
              device_embeddings_per_partitions[i], device_outputs[i], device_n_outputs[i],
              device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
              device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
              device_curr_new_embeddings_idx[i], device_n_next_step_1_done,
              device_n_next_step_2_done, only_copy_change);
#else
            run_single_step_vectorvertex_embedding<2><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
#endif
            break;
          }
          case 3: {
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
            run_single_step_vectorvertex_embedding_per_vertex <3><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
              device_embeddings_per_partitions[i], device_outputs[i], device_n_outputs[i],
              device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
              device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
              device_curr_new_embeddings_idx[i], device_n_next_step_1_done,
              device_n_next_step_2_done, only_copy_change);
#else
            run_single_step_vectorvertex_embedding<3><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
#endif                                  
            break;
          }
          case 4: {
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
            run_single_step_vectorvertex_embedding_per_vertex <4><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
              device_embeddings_per_partitions[i], device_outputs[i], device_n_outputs[i],
              device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
              device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
              device_curr_new_embeddings_idx[i], device_n_next_step_1_done,
              device_n_next_step_2_done, only_copy_change);
#else
            run_single_step_vectorvertex_embedding<4><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
#endif
            break;
          }
          case 5: {
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
            run_single_step_vectorvertex_embedding_per_vertex <5><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
              device_embeddings_per_partitions[i], device_outputs[i], device_n_outputs[i],
              device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
              device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
              device_curr_new_embeddings_idx[i], device_n_next_step_1_done,
              device_n_next_step_2_done, only_copy_change);
#else
            run_single_step_vectorvertex_embedding<5><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
#endif
            break;
          }
          case 6: {
#ifdef PROCESS_EMBEDDINGS_PER_VERTEX
            run_single_step_vectorvertex_embedding_per_vertex <6><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
              device_embeddings_per_partitions[i], device_outputs[i], device_n_outputs[i],
              device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
              device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i], 
              device_curr_new_embeddings_idx[i], device_n_next_step_1_done,
              device_n_next_step_2_done, only_copy_change);
#else
            run_single_step_vectorvertex_embedding<6><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
#endif
            break;
          }
          case 7: {
            run_single_step_vectorvertex_embedding<7><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
            break;
          }
          case 8: {
            run_single_step_vectorvertex_embedding<8><<<thread_blocks, THREAD_BLOCK_SIZE, 0, streams[i]>>> (device_embeddings[i], per_stream_n_embeddings, device_csr[i],
                                  device_outputs[i], device_n_outputs[i],
                                  device_new_embeddings_1[i], device_n_embeddings_1[i], device_new_embeddings_1_status[i],
                                  device_new_embeddings_2[i], device_n_embeddings_2[i], device_new_embeddings_2_status[i],
                                  device_curr_new_embeddings_idx[i], only_copy_change);
            break;
          }
        }
        
        //cudaDeviceSynchronize ();
        if (iter >= 1 && ENABLE_NEW_EMBEDDINGS_ON_THE_FLY_COPYING) {
          int curr_step_storage_id = 0;
          std::cout << "copying to n_outputs_1" << std::endl; 
          cudaStream_t t;
          int prev_curr_step_storage_id = 0;
          cudaStreamCreate (&t);

          while (cudaStreamQuery (streams[i]) == cudaErrorNotReady) {
            BUFFER_STATUS __buff_1_status, __buff_2_status;
            do {
              assert (cudaMemcpyAsync (&__buff_1_status, device_new_embeddings_1_status[i], sizeof (__buff_1_status), cudaMemcpyDeviceToHost, t) == cudaSuccess);
              assert (cudaMemcpyAsync (&__buff_2_status, device_new_embeddings_2_status[i], sizeof (__buff_2_status), cudaMemcpyDeviceToHost, t) == cudaSuccess);
              cudaStreamSynchronize (t);
              usleep (GPU_QUERY_WAIT_TIME);
            } while (__buff_1_status != BUFFER_STATUS::CPU_COPYING &&
                     __buff_2_status != BUFFER_STATUS::CPU_COPYING &&
                     cudaStreamQuery (streams[i]) == cudaErrorNotReady);
            //Above loop ends only when the storage id has changed or 
            //the stream's kernel has ended.
            usleep (1000000UL);
            if (__buff_1_status == BUFFER_STATUS::CPU_COPYING)
              prev_curr_step_storage_id = 0;
            else if (__buff_2_status == BUFFER_STATUS::CPU_COPYING)
              prev_curr_step_storage_id = 1;

            std::cout << "prev_curr_step_storage_id = " << prev_curr_step_storage_id <<
                         " curr_step_storage_id = " << curr_step_storage_id << 
                         " cudaStreamQuery (streams[i]) == cudaErrorNotReady " << 
                         (cudaStreamQuery (streams[i]) == cudaErrorNotReady) << std::endl;
            if (cudaStreamQuery (streams[i]) == cudaSuccess) {
              //Stream's kernel has completed, get out of loop.
              break;
            }
            std::cout << "n_new_embeddings_1[i] + n_new_embeddings_2[i] " << n_new_embeddings_1[i] + n_new_embeddings_2[i] << " * new_embedding_size " << 
            (n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size << std::endl;
            //copy
            switch (prev_curr_step_storage_id) {
              case 0: {
                //new_embeddings_1_status[i] = BUFFER_STATUS::CPU_COPYING;
                //assert (cudaMemcpyAsync (&device_new_embeddings_1_status[i], &new_embeddings_1_status[i], 
                //        sizeof (new_embeddings_1_status[i]), cudaMemcpyHostToDevice, t) == 0);
                int p = 0;
                assert (cudaMemcpyAsync (&p, device_n_embeddings_1[i], sizeof (p), cudaMemcpyDeviceToHost, t) == cudaSuccess);
                std::cout << "Copying " << p << " new embeddings each of size " << new_embedding_size << std::endl;
                
                if (p != max_embeddings/N_STREAMS) {
                  std::cout << "SYNCHRONIZATION ISSUE: device_n_embeddings_1[i] " << p << " > per_stream_n_embeddings " << max_embeddings/N_STREAMS << std::endl;
                } 
                if (true) {
                  cudaError_t err = cudaMemcpyAsync ((char*)new_embeddings_ptr[i] + (n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size, device_new_embeddings_1[i], max_embeddings/N_STREAMS*new_embedding_size, cudaMemcpyDeviceToHost, t);
                  cudaStreamSynchronize (t);
                  char* __new_ptr = (char*)malloc (p*new_embedding_size);
                  cudaMemcpyAsync (__new_ptr, device_new_embeddings_1[i], p*new_embedding_size, cudaMemcpyDeviceToHost, t);
                  for (int ii = 0; ii < p; ii++) {
                    VectorVertexEmbedding<2>& e = ((VectorVertexEmbedding<2>*)__new_ptr)[ii];
                    print_embedding<2> (&e);
                    std::cout << std::endl;
                  }
                  free (__new_ptr);
                  cudaStreamSynchronize (t);
                  //if (n_new_embeddings_1[i] + n_new_embeddings_2[i] >= 778462776)
                  //    exit(EXIT_SUCCESS);
                  //std::cout << "total embeddings found: " << n_new_embeddings_1[i] + n_new_embeddings_2[i] << std::endl;
                  if (err != cudaSuccess) {
                    //if (n_new_embeddings_1[i] + n_new_embeddings_2[i] >= 778462776)
                    //  exit(EXIT_SUCCESS);
                    std::cout << cudaGetErrorString (err) << std::endl;
                    assert (false);
                  }

                  // char* __new_ptr = (char*)new_embeddings_ptr[i] + (n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size;
                  //exit (EXIT_SUCCESS);
                }
                n_new_embeddings_1[i] += p;
                p = 0;
                assert (cudaMemcpyAsync (device_n_embeddings_1[i], &p, sizeof (p), cudaMemcpyHostToDevice, t) == cudaSuccess);
                new_embeddings_1_status[i] = BUFFER_STATUS::FREE;
                cudaError_t e = cudaMemcpyAsync (device_new_embeddings_1_status[i], &new_embeddings_1_status[i], 
                                                 sizeof (new_embeddings_1_status[i]), cudaMemcpyHostToDevice, t);
                //std::cout << cudaGetErrorString (e) << std::endl;
                assert (e == cudaSuccess);
                break;
              }

              case 1: {
                int p = 0;
                //new_embeddings_2_status[i] = BUFFER_STATUS::CPU_COPYING;
                //assert (cudaMemcpyAsync (&device_new_embeddings_2_status[i], &new_embeddings_2_status[i], 
                //        sizeof (new_embeddings_2_status[i]), cudaMemcpyHostToDevice, t) == 0);
                assert (cudaMemcpyAsync (&p, device_n_embeddings_2[i], sizeof (p), cudaMemcpyDeviceToHost, t) == cudaSuccess);
                std::cout << "Copying " << p << " new embeddings each of size " << new_embedding_size << std::endl;
                if (p > max_embeddings/N_STREAMS) {
                  std::cout << "SYNCHRONIZATION ISSUE: device_n_embeddings_1[i] " << p << " > per_stream_n_embeddings " << max_embeddings/N_STREAMS << std::endl;
                } 
                if (true) {
                  cudaError_t err = cudaMemcpyAsync ((char*)new_embeddings_ptr[i]+(n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size, device_new_embeddings_2[i], max_embeddings/N_STREAMS*new_embedding_size, cudaMemcpyDeviceToHost, t);
                  cudaStreamSynchronize (t);
                  char* __new_ptr = (char*)malloc (p*new_embedding_size);
                  cudaMemcpyAsync (__new_ptr, device_new_embeddings_2[i], p*new_embedding_size, cudaMemcpyDeviceToHost, t);
                  for (int ii = 0; ii < p; ii++) {
                    VectorVertexEmbedding<2>& e = ((VectorVertexEmbedding<2>*)__new_ptr)[ii];
                    print_embedding<2> (&e);
                    std::cout << std::endl;
                  }
                  free (__new_ptr);
                  cudaStreamSynchronize (t);
                  //if (n_new_embeddings_1[i] + n_new_embeddings_2[i] >= 778462776)
                   //   exit(EXIT_SUCCESS);
                  //std::cout << "total embeddings found: " << n_new_embeddings_1[i] + n_new_embeddings_2[i] << std::endl;
                  if (err != cudaSuccess) {
                   // if (n_new_embeddings_1[i] + n_new_embeddings_2[i] >= 778462776)
                   //   exit(EXIT_SUCCESS);
                    std::cout << "total embeddings found: " << n_new_embeddings_1[i] + n_new_embeddings_2[i] << std::endl;
                    std::cout << cudaGetErrorString (err) << std::endl;
                    assert (false);
                  }

                  // char* __new_ptr = (char*)new_embeddings_ptr[i] + (n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size;
                }
                n_new_embeddings_2[i] += p;
                p = 0;
                assert (cudaMemcpyAsync (device_n_embeddings_2[i], &p, sizeof (p), cudaMemcpyHostToDevice, t) == cudaSuccess);
                new_embeddings_2_status[i] = BUFFER_STATUS::FREE;
                assert (cudaMemcpyAsync (device_new_embeddings_2_status[i], &new_embeddings_2_status[i], 
                        sizeof (new_embeddings_2_status[i]), cudaMemcpyHostToDevice, t) == 0);
                break;
              }
            }
            prev_curr_step_storage_id = curr_step_storage_id;
            cudaStreamSynchronize (t);
            std::cout << "Copying done " << std::endl;
            //cudaStreamQuery (streams[i]);
          }

          cudaStreamDestroy (t);
        }

        //cudaDeviceSynchronize ();
        if (stream_synchronize)
          cudaStreamSynchronize (streams[i]);

      double t2 = convertTimeValToDouble (getTimeOfDay ());

      std::cout << "Execution time " << (t2-t1) << " secs" << std::endl;
      kernelTotalTime += (t2-t1);

        error = cudaGetLastError ();
        if (error != cudaSuccess) {
          const char* error_string = cudaGetErrorString (error);
          std::cout << "Cuda kernel error " << error_string << std::endl;
        } else {
          std::cout << "Cuda success " << std::endl;
        }
      }

      // char* __new_ptr = (char*)new_embeddings_ptr[0];

      // for (int ii = 0; ii < n_new_embeddings_1[0] + n_new_embeddings_2[0]; ii++) {
      //   VectorVertexEmbedding<2>& e = ((VectorVertexEmbedding<2>*)__new_ptr)[ii];
      //   print_embedding<2> (&e);
      //   std::cout << ii << std::endl;
      // }

      for (int i = 0; i < N_STREAMS; i++) 
      {
        //cudaStreamSynchronize (streams[i]);

        if (only_copy_change) {
          assert (false);
          //TODO: Change this to make an array of such ptrs
          cudaMemcpyAsync (new_embeddings_ptr, device_new_embeddings_1[i], n_new_embeddings_1[i]*2*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
          cudaMemcpyAsync (output_ptr, device_outputs[i], n_output[i]*2*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        }
        else {
          std::cout << "n_output[" << i << "] = " << n_output[i] << std::endl;
          int _n_new = 0;
          assert (cudaMemcpyAsync (&_n_new, device_n_embeddings_1[i], sizeof(0), cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);
          std::cout << "_n_new " << _n_new << std::endl;
          if (_n_new > 0 ) {
            assert (cudaMemcpyAsync ((char*)new_embeddings_ptr[i] + (n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size,
                                     device_new_embeddings_1[i], _n_new*(new_embedding_size), cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);
            int p = _n_new;
            char* __new_ptr = (char*)malloc (p*new_embedding_size);
            cudaMemcpyAsync (__new_ptr, device_new_embeddings_1[i], p*new_embedding_size, cudaMemcpyDeviceToHost, streams[i]);
            for (int ii = 0; ii < p; ii++) {
              VectorVertexEmbedding<2>& e = ((VectorVertexEmbedding<2>*)__new_ptr)[ii];
              print_embedding<2> (&e);
              std::cout << std::endl;
            }
            free (__new_ptr);
            cudaStreamSynchronize(streams[i]);
            n_new_embeddings_1[i] += _n_new;
          }
         
          assert (cudaMemcpyAsync (&_n_new, device_n_embeddings_2[i], sizeof(0), cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);
          std::cout << "_n_new " << _n_new << std::endl;
          if (_n_new > 0 ) {
            assert (cudaMemcpyAsync ((char*)new_embeddings_ptr[i] + (n_new_embeddings_1[i] + n_new_embeddings_2[i])*new_embedding_size,
                                     device_new_embeddings_2[i], _n_new*(new_embedding_size), cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);
            int p = _n_new;
            char* __new_ptr = (char*)malloc (p*new_embedding_size);
            cudaMemcpyAsync (__new_ptr, device_new_embeddings_2[i], p*new_embedding_size, cudaMemcpyDeviceToHost, streams[i]);
            for (int ii = 0; ii < p; ii++) {
              VectorVertexEmbedding<2>& e = ((VectorVertexEmbedding<2>*)__new_ptr)[ii];
              print_embedding<2> (&e);
              std::cout << std::endl;
            }
            free (__new_ptr);
            cudaStreamSynchronize(streams[i]);
            n_new_embeddings_2[i] += _n_new;
          }
          
          assert (cudaMemcpyAsync (&n_output[i], device_n_outputs[i], sizeof(0), cudaMemcpyDeviceToHost, streams[i]) == cudaSuccess);

          //assert (cudaMemcpyAsync (output_ptr[i], device_outputs[i], n_output[i]*(new_embedding_size), cudaMemcpyDeviceToHost, streams[i])== cudaSuccess);
          n_output[i] = 0;
        }
        //cudaMemcpyAsync (&n_new_embeddings_1[i], device_n_embeddings_1[i], sizeof(0), cudaMemcpyDeviceToHost, streams[i]);
        //cudaMemcpyAsync (&n_output_1[i], device_n_outputs_1[i], sizeof(0), cudaMemcpyDeviceToHost, streams[i]);

        error = cudaGetLastError ();
        if (error != cudaSuccess) {
          const char* error_string = cudaGetErrorString (error);
          std::cout << "Cuda device to host copy error " << error_string << std::endl;
        } else {
          std::cout << "Cuda device to host copy success " << std::endl;
        }

        std::cout << "Stream " << i << std::endl;
        std::cout << "n_new_embeddings "<<n_new_embeddings_1[i] + n_new_embeddings_2[i]<<std::endl;
        std::cout << "n_new_embeddings_1 "<<n_new_embeddings_1[i];
        std::cout << " n_output "<<n_output[i];
        std::cout << " n_output_1 "<<n_output_1[i]<<std::endl;
      }
      
      for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamSynchronize (streams[i]);
        cudaError_t err = cudaStreamDestroy (streams[i]);
        assert (err == cudaSuccess);
      }

      double stream_time_2 = convertTimeValToDouble (getTimeOfDay ());

      total_stream_time += (stream_time_2-stream_time_1);

      //TODO: wait for all kernels and data transfers to finish
      for (int i = 0; i < N_STREAMS; i++) {
        n_next_step_embeddings += n_new_embeddings_1[i]+ n_new_embeddings_2[i];
      }
      switch (iter) {
        case 1: {
          VectorVertexEmbedding<2>* new_embeddings = (VectorVertexEmbedding<2>*)malloc (sizeof (VectorVertexEmbedding<2>)*n_next_step_embeddings);
          
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream] + n_new_embeddings_2[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<2> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<1>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                new_embeddings [j] = embedding;
                j++;
              }
              else {
                VectorVertexEmbedding<2> embedding = ((VectorVertexEmbedding<2>*)(new_embeddings_ptr[stream]))[i];
                new_embeddings [j] = embedding;
                //print_embedding<2> (&embedding);
                //std::cout << std::endl;
                
                #ifdef DEBUG
                if (embedding.get_n_vertices () != (iter + 1)) {
                  printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
                }
                #endif
                j++;
              }
            }
          }
          //return;
          assert (j == n_next_step_embeddings);
          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));
          exit (EXIT_SUCCESS);
          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<2> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<1>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_2.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_2.push_back (((VectorVertexEmbedding<2>*)output_ptr[stream])[i]);
              }
            }
          }
          
          break;
        }
        
        case 2: {
          VectorVertexEmbedding<3>* new_embeddings = (VectorVertexEmbedding<3>*)malloc (sizeof (VectorVertexEmbedding<3>)*n_next_step_embeddings);
          
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream] + n_new_embeddings_2[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<3> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<2>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                assert (false);
                new_embeddings [i] = embedding;
              }
              else {
                VectorVertexEmbedding<3> embedding = ((VectorVertexEmbedding<3>*)(new_embeddings_ptr[stream]))[i];
                new_embeddings [j] = embedding;
                //print_embedding<3> (&embedding);
                //std::cout << std::endl;
                //s = embedding_to_string<3> (&embedding);
                //#ifdef DEBUG
                //if (embedding.get_n_vertices () != (iter + 1)) {
                //  printf ("embedding has %ld vertices\n", embedding.get_n_vertices ());
                //}
                //#endif
                j++;
              }
            }
          }

          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));
          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<3> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<2>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_3.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_3.push_back (((VectorVertexEmbedding<3>*)output_ptr[stream])[i]);
              }
            }
          }

          break;
        }
        
        case 3: {
          VectorVertexEmbedding<4>* new_embeddings = (VectorVertexEmbedding<4>*)malloc (sizeof (VectorVertexEmbedding<4>)*n_next_step_embeddings);
          
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<4> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<3>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                new_embeddings [i] = embedding;
              }
              else {
                VectorVertexEmbedding<4> embedding = ((VectorVertexEmbedding<4>*)(new_embeddings_ptr[stream]))[i];
                new_embeddings [j] = embedding;
                j++;
                #ifdef DEBUG
                if (embedding.get_n_vertices () != (iter + 1)) {
                  printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
                }
                #endif
              }
            }
          }
          
          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));

          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<4> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<3>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_4.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_4.push_back (((VectorVertexEmbedding<4>*)output_ptr[stream])[i]);
              }
            }
          }
          break;
        }
        
        case 4: {
          VectorVertexEmbedding<5>* new_embeddings = (VectorVertexEmbedding<5>*)malloc (sizeof (VectorVertexEmbedding<5>)*n_next_step_embeddings);
          
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<5> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<4>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                new_embeddings [i] = embedding;
              }
              else {
                VectorVertexEmbedding<5> embedding = ((VectorVertexEmbedding<5>*)new_embeddings_ptr[stream])[i];
                new_embeddings [j] = embedding;
                j++;
                #ifdef DEBUG
                if (embedding.get_n_vertices () != (iter + 1)) {
                  printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
                }
                #endif
              }
            }
          }
          
          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));
          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<5> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<4>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_5.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_5.push_back (((VectorVertexEmbedding<5>*)output_ptr[stream])[i]);
              }
            }
          }

          break;
        }
        
        case 5: {
          VectorVertexEmbedding<6>* new_embeddings = (VectorVertexEmbedding<6>*)malloc (sizeof (VectorVertexEmbedding<6>)*n_next_step_embeddings);
          
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<6> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<5>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                new_embeddings [i] = embedding;
              }
              else {
                VectorVertexEmbedding<6> embedding = ((VectorVertexEmbedding<6>*)new_embeddings_ptr[stream])[i];
                new_embeddings [j] = embedding;
                j++;
                #ifdef DEBUG
                if (embedding.get_n_vertices () != (iter + 1)) {
                  printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
                }
                #endif
              }
            }
          }
          
          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));
          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<6> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<5>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_6.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_6.push_back (((VectorVertexEmbedding<6>*)output_ptr[stream])[i]);
              }
            }
          }
          break;
        }
        
        case 6: {
          VectorVertexEmbedding<7>* new_embeddings = (VectorVertexEmbedding<7>*)malloc (sizeof (VectorVertexEmbedding<7>)*n_next_step_embeddings);
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<7> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<6>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                new_embeddings [i] = embedding;
              }
              else {
                VectorVertexEmbedding<7> embedding = ((VectorVertexEmbedding<7>*)new_embeddings_ptr[stream])[i];
                new_embeddings [j] = embedding;
                j++;
                #ifdef DEBUG
                if (embedding.get_n_vertices () != (iter + 1)) {
                  printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
                }
                #endif
              }
            }
          }
          
          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));
          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<7> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<6>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_7.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_7.push_back (((VectorVertexEmbedding<7>*)output_ptr[stream])[i]);
              }
            }
          }
          break;
        }
        
        case 7: {
          VectorVertexEmbedding<8>* new_embeddings = (VectorVertexEmbedding<8>*)malloc (sizeof(VectorVertexEmbedding<8>)*n_next_step_embeddings);
          size_t j = 0;
          for (int stream = 0; stream < N_STREAMS; stream++) {
            for (int i = 0; i < n_new_embeddings_1[stream]; i++) {
              if (only_copy_change) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<8> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<7>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                new_embeddings [i] = embedding;
              }
              else {
                VectorVertexEmbedding<8> embedding = ((VectorVertexEmbedding<8>*)new_embeddings_ptr[stream])[i];
                new_embeddings [j] = embedding;
                j++;
                #ifdef DEBUG
                if (embedding.get_n_vertices () != (iter + 1)) {
                  printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
                }
                #endif
              }
            }
          }
          
          embeddings.push_back (std::make_pair (&new_embeddings[0], n_next_step_embeddings));
          for (int stream = 0; stream < N_STREAMS; stream++) {
            if (only_copy_change) {
              for (int i = 0; i < n_output[stream]; i++) {
                int id = ((int*)new_embeddings_ptr)[2*i];
                int v = ((int*)new_embeddings_ptr)[2*i+1];
                VectorVertexEmbedding<8> embedding;
                vector_embedding_from_one_less_size (((VectorVertexEmbedding<7>*)embeddings_ptr)[id], embedding);
                embedding.add (v);
                output_8.push_back (embedding);
              }
            } else {
              for (int i = 0; i < n_output[stream]; i++) {
                output_8.push_back (((VectorVertexEmbedding<8>*)output_ptr[stream])[i]);
              }
            }
          }

          break;
        }
      }
      
      //embeddings = new_embeddings;

      for (int i = 0; i < N_STREAMS; i++) {
        cudaFree (device_embeddings_per_partitions[i]);
        cudaFree (device_embeddings[i]);
        cudaFree (device_new_embeddings_1[i]);
        cudaFree (device_new_embeddings_2[i]);
        cudaFree (device_n_embeddings_1[i]);
        cudaFree (device_outputs[i]);
        cudaFree (device_n_outputs[i]);
        cudaFree (device_csr[i]);
      }
    }

    new_embeddings_size = n_next_step_embeddings;
  }

#ifdef PINNED_MEMORY
  cudaFree (global_mem_ptr);
#else
  delete[] global_mem_ptr;
#endif
  std::cout << "Number of embeddings found "<< (output_1.size () + output_2.size () + output_3.size () + output_4.size () + output_5.size () + output_6.size () + output_7.size () + output_8.size ()) << std::endl;
  std::cout << "Time spent in GPU kernel execution " << kernelTotalTime << std::endl;
  std::cout << "Time spent in Streams " << total_stream_time << std::endl;
  
  /* For BitVectorVertexEmbedding
   for (iter; iter < 10 && embeddings.size () > 0; iter++) {
    std::cout << "iter " << iter << " embeddings " << embeddings.size () << std::endl;
    size_t global_mem_size = 3*1024*1024*1024UL;
    char* global_mem_ptr = new char[global_mem_size];
  #ifdef DEBUG
    memset (global_mem_ptr, 0, global_mem_size);
  #endif
    int n_embeddings = embeddings.size ();
    //n_embeddings = (n_embeddings/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE;
    std::cout << "iter " << iter << " n_embeddings " << n_embeddings << std::endl;
  
    for (int i = 0; i < n_embeddings; i++) {
      ((BitVectorVertexEmbedding*)global_mem_ptr)[i] = embeddings[i];
    }
    void* embeddings_ptr = global_mem_ptr;

    int n_new_embeddings = 0;
    int n_new_embeddings_1 = 0;
    void* new_embeddings_ptr = (char*)embeddings_ptr + (n_embeddings)*sizeof(BitVectorVertexEmbedding);
    int max_embeddings = 1000000;
    void* output_ptr = (char*)new_embeddings_ptr + (max_embeddings)*sizeof(BitVectorVertexEmbedding);
    int n_output = 0;
    int n_output_1 = 0;
    char* device_embeddings;
    char *device_new_embeddings;
    int* device_n_embeddings;
    int* device_n_embeddings_1;
    char *device_outputs;
    int* device_n_outputs;
    int* device_n_outputs_1;
    CSR* device_csr;

    cudaMalloc (&device_embeddings, n_embeddings*sizeof(BitVectorVertexEmbedding));
    cudaMemcpy (device_embeddings, embeddings_ptr,
                n_embeddings*sizeof(BitVectorVertexEmbedding),
                cudaMemcpyHostToDevice);
    cudaMalloc (&device_new_embeddings, max_embeddings*sizeof (BitVectorVertexEmbedding));
    cudaMalloc (&device_outputs, max_embeddings*sizeof (BitVectorVertexEmbedding));
    cudaMalloc (&device_n_embeddings, sizeof (0));
    cudaMalloc (&device_n_embeddings_1, sizeof (0));
    cudaMalloc (&device_n_outputs, sizeof (0));
    cudaMalloc (&device_n_outputs_1, sizeof (0));
    cudaMalloc (&device_csr, sizeof(CSR));

    cudaMemcpy (device_n_embeddings, &n_new_embeddings,
                sizeof (n_new_embeddings), cudaMemcpyHostToDevice);
    cudaMemcpy (device_n_outputs, &n_output, sizeof (n_output),
                cudaMemcpyHostToDevice);

    cudaMemcpy (device_n_embeddings_1, &n_new_embeddings_1,
                sizeof (n_new_embeddings_1), cudaMemcpyHostToDevice);
    cudaMemcpy (device_n_outputs_1, &n_output_1, sizeof (n_output_1),
                cudaMemcpyHostToDevice);

    cudaMemcpy (device_csr, csr, sizeof (CSR), cudaMemcpyHostToDevice);

    std::cout << "starting kernel with n_embeddings: " << n_embeddings;

    double t1 = convertTimeValToDouble (getTimeOfDay ());
#ifdef USE_FIXED_THREADS
    std::cout << " threads: " << MAX_CUDA_THREADS/THREAD_BLOCK_SIZE << std::endl;
      run_single_step_bitvector_embedding<<<MAX_CUDA_THREADS/THREAD_BLOCK_SIZE,THREAD_BLOCK_SIZE>>> (device_embeddings, n_embeddings, device_csr,
                              device_outputs, device_n_outputs,
                              device_new_embeddings, device_n_embeddings,
                              device_n_outputs_1, device_n_embeddings_1);
#else
    int thread_blocks = (n_embeddings%THREAD_BLOCK_SIZE != 0) ? (n_embeddings/THREAD_BLOCK_SIZE+1) : n_embeddings/THREAD_BLOCK_SIZE;
    std::cout << " threads: " << n_embeddings/THREAD_BLOCK_SIZE << std::endl;
    run_single_step_bitvector_embedding<<<thread_blocks, THREAD_BLOCK_SIZE>>> (device_embeddings, n_embeddings, device_csr,
                              device_outputs, device_n_outputs,
                              device_new_embeddings, device_n_embeddings,
                              device_n_outputs_1, device_n_embeddings_1);
#endif

    cudaDeviceSynchronize ();

    double t2 = convertTimeValToDouble (getTimeOfDay ());

    std::cout << "Execution time " << (t2-t1) << " secs" << std::endl;
    kernelTotalTime += (t2-t1);

    cudaError_t error = cudaGetLastError ();
    if (error != cudaSuccess) {
      const char* error_string = cudaGetErrorString (error);
      std::cout << error_string << std::endl;
    } else {
      std::cout << "Cuda success " << std::endl;
    }

    cudaMemcpy (new_embeddings_ptr, device_new_embeddings, max_embeddings*sizeof(BitVectorVertexEmbedding), cudaMemcpyDeviceToHost);
    cudaMemcpy (output_ptr, device_outputs, max_embeddings*sizeof(BitVectorVertexEmbedding), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_new_embeddings, device_n_embeddings, sizeof(0), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_output, device_n_outputs, sizeof(0), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_new_embeddings_1, device_n_embeddings_1, sizeof(0), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_output_1, device_n_outputs_1, sizeof(0), cudaMemcpyDeviceToHost);

    std::cout << "n_new_embeddings "<<n_new_embeddings<<std::endl;
    std::cout << "n_new_embeddings_1 "<<n_new_embeddings_1;
    std::cout << " n_output "<<n_output;
    std::cout << " n_output_1 "<<n_output_1<<std::endl;
    std::vector<BitVectorVertexEmbedding> new_embeddings;
  
    for (int i = 0; i < n_new_embeddings; i++) {
      BitVectorVertexEmbedding embedding = ((BitVectorVertexEmbedding*)new_embeddings_ptr)[i];
      new_embeddings.push_back (embedding);
    #ifdef DEBUG
      if (embedding.get_n_vertices () != (iter + 1)) {
        printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
      }
    #endif
    }
    for (int i = 0; i < n_output; i++) {
      output.push_back (((BitVectorVertexEmbedding*)output_ptr)[i]);
    }
    embeddings = new_embeddings;

    cudaFree (device_embeddings);
    cudaFree (device_new_embeddings);
    cudaFree (device_n_embeddings);
    cudaFree (device_outputs);
    cudaFree (device_n_outputs);
    cudaFree (device_csr);
    delete[] global_mem_ptr;
    #endif
  } 
   */
}
