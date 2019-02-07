#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <unordered_set>
#include <time.h>
#include <sys/time.h>

#include <string.h>
#include <assert.h>

#define LINE_SIZE 1024*1024
//#define USE_FIXED_THREADS
#define MAX_CUDA_THREADS (96*96)
#define THREAD_BLOCK_SIZE 96
#define WARP_SIZE 32
//#define USE_CSR_IN_SHARED
//#define USE_EMBEDDING_IN_SHARED_MEM
//#define USE_EMBEDDING_IN_GLOBAL_MEM
#define USE_EMBEDDING_IN_LOCAL_MEM
//#define SHARED_MEM_NON_COALESCING

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

//#define USE_CONSTANT_MEM

typedef uint8_t SharedMemElem;

const int N = 3312;
const int N_EDGES = 9074;
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

  int get_id () {return id;}
  int get_label () {return label;}
  void add_edge (int vertexID) {edges.push_back (vertexID);}
  std::vector <int>& get_edges () {return edges;}
  void print (std::ostream& os)
  {
    os << id << " " << label << " ";
    for (auto edge : edges) {
      os << edge << " ";
    }

    os << std::endl;
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
    short int id;
    short int label;
    short int start_edge_id;
    short int end_edge_id;
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

    void set_start_edge_id (short int start) {start_edge_id = start;}
    void set_end_edge_id (short int end) {end_edge_id = end;}
  };

  typedef short int Edge;

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
           edge_iter < vertices[i].end_edge_id; edge_iter++) {
        os << edges[edge_iter] << " ";
      }
      os << std::endl;
    }
  }

  __host__ __device__
  int get_start_edge_idx (int vertex_id)
  {
    assert (vertex_id < n_vertices && 0 <= vertex_id);
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
class VertexEmbedding
{
private:
#ifdef USE_EMBEDDING_IN_SHARED_MEM
  unsigned char array[CVT_TO_NEXT_MULTIPLE(N/8, 32*sizeof(SharedMemElem))];
#else
  unsigned char array[(N/8)];
#endif

public:
  __device__ __host__
  VertexEmbedding ()
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
  VertexEmbedding (const VertexEmbedding& embedding)
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
  ~VertexEmbedding ()
  {
    //delete[] array;
  }
};

void print_embedding (VertexEmbedding embedding, std::ostream& os);


std::vector<VertexEmbedding> get_extensions (VertexEmbedding& embedding, CSR* csr)
{
  std::vector<VertexEmbedding> extensions;

  if (embedding.all_false ()) {
    for (int u = 0; u < N; u++) {
      VertexEmbedding extension;
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
            VertexEmbedding extension = VertexEmbedding(embedding);
            extension.set(v);
            extensions.push_back(extension);
          }
        }
      }
    }
  }

  return extensions;
}

std::vector<VertexEmbedding> get_initial_embedding (CSR* csr)
{
  VertexEmbedding embedding;
  std::vector <VertexEmbedding> embeddings;

  embeddings.push_back (embedding);

  return embeddings;
}

bool (*filter) (CSR* csr, VertexEmbedding& embedding);
void (*process) (std::vector<VertexEmbedding>& output, VertexEmbedding& embedding);

__host__ __device__
bool clique_filter (CSR* csr, VertexEmbedding* embedding)
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

void clique_process (std::vector<VertexEmbedding>& output, VertexEmbedding& embedding)
{
  output.push_back (embedding);
}

void run_single_step_initial (void* input, int n_embeddings, CSR* csr,
                      std::vector<VertexEmbedding>& output,
                      std::vector<VertexEmbedding>& next_step)
{
  VertexEmbedding* embeddings = (VertexEmbedding*)input;

  for (int i = 0; i < n_embeddings; i++) {
    VertexEmbedding embedding = embeddings[i];
    std::vector<VertexEmbedding> extensions = get_extensions (embedding, csr);

    for (auto extension : extensions) {
      if (clique_filter (csr, &extension)) {
        clique_process (output, extension);
        next_step.push_back (extension);
      }
    }
  }
}

__device__
void printf_embedding (VertexEmbedding* embedding)
{
  printf ("[");
  for (int u = 0; u < N; u++) {
    if (embedding->test(u)) {
      printf ("%d, ", u);
    }
  }

  printf ("]\n");
}

__global__
void run_single_step (void* input, int n_embeddings, CSR* csr,
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

  VertexEmbedding* embeddings = (VertexEmbedding*)input;
  VertexEmbedding* new_embeddings = (VertexEmbedding*)next_step;
  VertexEmbedding* output = ((VertexEmbedding*)output_ptr);
#ifdef USE_EMBEDDING_IN_LOCAL_MEM
  unsigned char temp_buffer [sizeof(VertexEmbedding)];
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
        memcpy (local_shared_buff, &embeddings[i], sizeof(VertexEmbedding));

        VertexEmbedding* embedding = (VertexEmbedding*) local_shared_buff;
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

          for (int j = 0; j < sizeof (VertexEmbedding)/sizeof (SharedMemElem);
               j += thread_block_size) {
            int idx = per_thread_shared_mem_size/sizeof(SharedMemElem)*emb;
            if (true or j + lane_id  < sizeof (VertexEmbedding)/sizeof (SharedMemElem)) { //TODO: Remove this if by doing padding with VertexEmbedding
              shared_buff[idx + j + lane_id] = embedding_buff[j + lane_id];
            }
          }
        }

        VertexEmbedding* embedding = (VertexEmbedding*) local_shared_buff;
      #endif
      //embedding = &embeddings[i];
    #elif defined(USE_EMBEDDING_IN_LOCAL_MEM)
      //memcpy (&temp[0], &embeddings[i], sizeof (VertexEmbedding));
      memcpy (&temp_buffer[0], &embeddings[i], sizeof(VertexEmbedding));
      VertexEmbedding* embedding = (VertexEmbedding*)&temp_buffer[0];
      //VertexEmbedding* embedding = &embeddings[i];
    #elif defined(USE_EMBEDDING_IN_GLOBAL_MEM)
      VertexEmbedding* embedding = &embeddings[i];
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
            VertexEmbedding* extension = embedding;
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
              VertexEmbedding* extension = embedding;
              extension->set (v);
              
              memcpy (&output[prev_n_output + i], extension, 
                      sizeof(VertexEmbedding));
              memcpy (&new_embeddings[prev_n_next_step + i], extension,
                      sizeof (VertexEmbedding));
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
            VertexEmbedding* extension = embedding;
            extension->set(v);
            if (clique_filter (csr, extension)) {
              memcpy (&output[atomicAdd(n_output,1)], extension, sizeof (VertexEmbedding));
              memcpy (&new_embeddings[atomicAdd(n_next_step,1)], extension, sizeof (VertexEmbedding));
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
}

void print_embedding (VertexEmbedding embedding, std::ostream& os)
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

    vertices.push_back (vertex);
  }

  fclose (fp);

  //std::cout << "n_edges "<<n_edges <<std::endl;
  Graph graph (vertices, n_edges);

  CSR* csr = new CSR(N, N_EDGES);
  std::cout << "sizeof(CSR)"<< sizeof(CSR)<<std::endl;
  std::cout <<"sizeof(VertexEmbedding)" << sizeof(VertexEmbedding) << std::endl;
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

  std::vector<VertexEmbedding> initial_embeddings = get_initial_embedding (csr);
  std::vector<VertexEmbedding> output;
  std::vector<VertexEmbedding> embeddings = initial_embeddings;
  //filter = clique_filter;
  //process = clique_process;
  int iter = 0;
  {
    std::vector<VertexEmbedding> new_embeddings;
    run_single_step_initial (&initial_embeddings[0], 1, csr, output, new_embeddings);

    embeddings = new_embeddings;
  }

  iter = 1;

  double_t kernelTotalTime = 0.0;

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
      ((VertexEmbedding*)global_mem_ptr)[i] = embeddings[i];
    }
    void* embeddings_ptr = global_mem_ptr;

    int n_new_embeddings = 0;
    int n_new_embeddings_1 = 0;
    void* new_embeddings_ptr = (char*)embeddings_ptr + (n_embeddings)*sizeof(VertexEmbedding);
    int max_embeddings = 1000000;
    void* output_ptr = (char*)new_embeddings_ptr + (max_embeddings)*sizeof(VertexEmbedding);
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

    cudaMalloc (&device_embeddings, n_embeddings*sizeof(VertexEmbedding));
    cudaMemcpy (device_embeddings, embeddings_ptr,
                n_embeddings*sizeof(VertexEmbedding),
                cudaMemcpyHostToDevice);
    cudaMalloc (&device_new_embeddings, max_embeddings*sizeof (VertexEmbedding));
    cudaMalloc (&device_outputs, max_embeddings*sizeof (VertexEmbedding));
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
      run_single_step<<<MAX_CUDA_THREADS/THREAD_BLOCK_SIZE,THREAD_BLOCK_SIZE>>> (device_embeddings, n_embeddings, device_csr,
                              device_outputs, device_n_outputs,
                              device_new_embeddings, device_n_embeddings,
                              device_n_outputs_1, device_n_embeddings_1);
#else
    int thread_blocks = (n_embeddings%THREAD_BLOCK_SIZE != 0) ? (n_embeddings/THREAD_BLOCK_SIZE+1) : n_embeddings/THREAD_BLOCK_SIZE;
    std::cout << " threads: " << n_embeddings/THREAD_BLOCK_SIZE << std::endl;
    run_single_step<<<thread_blocks, THREAD_BLOCK_SIZE>>> (device_embeddings, n_embeddings, device_csr,
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

    cudaMemcpy (new_embeddings_ptr, device_new_embeddings, max_embeddings*sizeof(VertexEmbedding), cudaMemcpyDeviceToHost);
    cudaMemcpy (output_ptr, device_outputs, max_embeddings*sizeof(VertexEmbedding), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_new_embeddings, device_n_embeddings, sizeof(0), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_output, device_n_outputs, sizeof(0), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_new_embeddings_1, device_n_embeddings_1, sizeof(0), cudaMemcpyDeviceToHost);
    cudaMemcpy (&n_output_1, device_n_outputs_1, sizeof(0), cudaMemcpyDeviceToHost);

    std::cout << "n_new_embeddings "<<n_new_embeddings<<std::endl;
    std::cout << "n_new_embeddings_1 "<<n_new_embeddings_1;
    std::cout << " n_output "<<n_output;
    std::cout << " n_output_1 "<<n_output_1<<std::endl;
    std::vector<VertexEmbedding> new_embeddings;
  
    for (int i = 0; i < n_new_embeddings; i++) {
      VertexEmbedding embedding = ((VertexEmbedding*)new_embeddings_ptr)[i];
      new_embeddings.push_back (embedding);
    #ifdef DEBUG
      if (embedding.get_n_vertices () != (iter + 1)) {
        printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
      }
    #endif
    }
    for (int i = 0; i < n_output; i++) {
      output.push_back (((VertexEmbedding*)output_ptr)[i]);
    }
    embeddings = new_embeddings;

    cudaFree (device_embeddings);
    cudaFree (device_new_embeddings);
    cudaFree (device_n_embeddings);
    cudaFree (device_outputs);
    cudaFree (device_n_outputs);
    cudaFree (device_csr);
    delete[] global_mem_ptr;
  }

  std::cout << "Number of embeddings found "<< output.size () << std::endl;
  std::cout << "Time spent in execution " << kernelTotalTime << std::endl;
  /*for (auto embedding : output) {
    print_embedding (embedding, std::cout);
    std::cout << std::endl;
  }*/
}
