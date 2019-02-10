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

#define __global__
#define __host__
#define __device__

#define LINE_SIZE 1024*1024
//citeseer.graph
//const int N = 3312;
//const int N_EDGES = 9074;

//micro.graph
const int N = 100000;
const int N_EDGES = 2160312;

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

int atomicAdd (int* x, int i)
{
  int old = *x;
  *x = *x + i;
  return old;
}

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
    int id;
    int label;
    int start_edge_id;
    int end_edge_id;
    
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
      assert (id >= 0);
      assert (label >= 0);
    }
    
    void set_start_edge_id (int start) {start_edge_id = start;}
    void set_end_edge_id (int end) {end_edge_id = end;}
  };
  
  typedef int Edge;
  
public:
  CSR::Vertex* vertices;
  CSR::Edge* edges;
  int n_vertices;
  int n_edges;
  
public:
  CSR (int _n_vertices, int _n_edges) 
  { 
    vertices = new CSR::Vertex[N];
    edges = new CSR::Edge[N_EDGES];
    assert (N == _n_vertices);
    assert (N_EDGES == _n_edges);
    n_vertices = _n_vertices;
    n_edges = _n_edges;
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
};

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
  
//template <size_t N> using BitVectorVertexEmbedding = std::bitset<N>;

class BitVectorVertexEmbedding
{
private:
  unsigned char array[((N/8)+1)];

public:
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
    return ((n/8)+1)*8;
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
  ~BitVectorVertexEmbedding ()
  {
    //delete[] array;
  }
};

template <size_t size> 
class VectorVertexEmbedding
{
private:
  uint32_t array[size];
  size_t filled_size;
  
public:
  __device__ __host__
  VectorVertexEmbedding ()
  {
    filled_size = 0;
  }

  __host__ __device__
  VectorVertexEmbedding (const VectorVertexEmbedding<size>& embedding)
  {
    assert (embedding.get_max_size () <= get_max_size ());
    filled_size = 0;
    for (int i = 0; i < embedding.get_n_vertices (); i++) {
      add (embedding.get_vertex (i));
    }
  }
  
  __host__ __device__
  void add (int v)
  {
    if (!(size != 0 and filled_size < size)) {
      printf ("filled_size %ld, size %ld\n", filled_size, size);
      //assert (size != 0 and filled_size < size);
      assert (false);
    }
    array[filled_size++] = v;
  }

  __host__ __device__
  void remove (int v)
  {
    printf ("Do not support remove\n");
    assert (false);
  }

  __host__ __device__
  bool has (int index)
  {
    for (int i = 0; i < filled_size; i++) {
      if (array[i] == index) {
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
  int get_vertex (int index) const
  {
    return array[index];
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

template <size_t size>
void vector_embedding_from_one_less_size (VectorVertexEmbedding<size>& vec_emb1,
                                          VectorVertexEmbedding<size+1>& vec_emb2)
{
  //TODO: Optimize here, filled_size++ in add is being called several times
  //but can be called only once too
  for (int i = 0; i < vec_emb1.get_n_vertices (); i++) {
    vec_emb2.add (vec_emb1.get_vertex (i));
  }
}

template <size_t size> 
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

void get_extensions_bitvector (BitVectorVertexEmbedding& embedding, CSR* csr, 
                std::vector<BitVectorVertexEmbedding>& extensions)
{  
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
}

template <size_t size>
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
bool clique_filter_bitvector (CSR* csr, BitVectorVertexEmbedding& embedding)
{
  for (int u = 0; u < N; u++) {
    if (embedding.test(u)) {
      for (int v = 0; v < N; v++) {
        if (u != v and embedding.test(v)) {
          if (!csr->has_edge (u, v)) {
            return false;
          }
        }
      }
    }
  }
  
  return true;
}

template <size_t size>
__host__ __device__
bool clique_filter_vector (CSR* csr, VectorVertexEmbedding<size>* embedding)
{
  for (int i = 0; i < embedding->get_n_vertices (); i++) {
    int u = embedding->get_vertex (i);
    for (int j = 0; j < embedding->get_n_vertices (); j++) {
      int v = embedding->get_vertex (j);
      if (u != v and embedding->has(v)) {
        if (!csr->has_edge (u, v)) {
          return false;
        }
      }
    }
  }

  return true;
}

void clique_process_bitvector (std::vector<BitVectorVertexEmbedding>& output, BitVectorVertexEmbedding& embedding)
{
  output.push_back (embedding);
}

template <size_t size>
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
    std::vector<BitVectorVertexEmbedding> extensions;
    get_extensions_bitvector (embedding, csr, extensions);
    
    for (auto extension : extensions) {
      if (clique_filter_bitvector (csr, extension)) {
        clique_process_bitvector (output, extension);
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
void printf_embedding_bitvector (BitVectorVertexEmbedding& embedding) 
{
  printf ("[");
  for (int u = 0; u < N; u++) {
    if (embedding.test(u)) {
      printf ("%d, ", u);
    }
  }
  
  printf ("]\n");
}

__global__
void run_single_step_bitvector (void* input, int n_embeddings, CSR* csr,
                      void* output_ptr, 
                      int* n_output,
                      void* next_step, int* n_next_step)
{
  printf ("run_single_step: start, n_embeddings %d\n", n_embeddings);  
  BitVectorVertexEmbedding* embeddings = (BitVectorVertexEmbedding*)input;
  BitVectorVertexEmbedding* new_embeddings = (BitVectorVertexEmbedding*)next_step;
  
  //int i = blockIdx.x*blockDim.x + threadIdx.x;
  //if (i >= n_embeddings)
  //  return;
  for (int i = 0; i < n_embeddings; i++) {
    //printf ("i %d ", i);
    BitVectorVertexEmbedding& embedding = embeddings[i];
    //std::vector<BitVectorVertexEmbedding> extensions = get_extensions (embedding, csr);
    /*if (i ==1500) {
      printf ("Embedding at 1500");
      printf_embedding (embedding);
    }*/
    for (int u = 0; u < N; u++) {
      if (embedding.test(u)) {
        if (i >= 3229)
          ;//printf ("u %d\n", u);
        for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
          int v = csr->get_edges () [e];
          if (i == 3311)
            ;//printf ("v = %d\n", v);
          if (embedding.test (v) == false) {
            BitVectorVertexEmbedding extension = BitVectorVertexEmbedding(embedding);
            extension.set(v);
            /*if (i == 1500) {
              printf ("Extension at 1500");
              printf_embedding (extension);
            }*/
            
            if (clique_filter_bitvector (csr, extension)) {
              /*if (i == 1500) printf ("extension passed\n");
              if (i >= 1000 && i < 2000) q[0]++;
              if (i >= 2000 && i < 3000) q[1]++;
              if (i == 1500) q[2]++;*/
              //clique_process (output, extension);
              //(*n_output)++;
              ((BitVectorVertexEmbedding*)output_ptr)[*n_output] = BitVectorVertexEmbedding(extension);
              (*n_output)++;
              //(*n_next_step)++; //make it atomic
              new_embeddings[*n_next_step] = BitVectorVertexEmbedding(extension);
              (*n_next_step)++;
            }
          }
        }
        
        if (i >= 3229)
          ;//printf ("u done %d\n", u);
      }
    }
  }
  
  printf ("step done\n");
}

template <size_t embedding_size>
void run_single_step_vector (void* input, int n_embeddings, CSR* csr,
                      void* output_ptr, 
                      int* n_output,
                      void* next_step, int* n_next_step)
{
  printf ("run_single_step: start, n_embeddings %d\n", n_embeddings);  
  VectorVertexEmbedding<embedding_size>* embeddings = (VectorVertexEmbedding<embedding_size>*)input;
  VectorVertexEmbedding<embedding_size+1>* new_embeddings = (VectorVertexEmbedding<embedding_size+1>*)next_step;
  
  for (int i = 0; i < n_embeddings; i++) {
    //printf ("i %d ", i);
    VectorVertexEmbedding<embedding_size>& embedding = embeddings[i];
    //std::vector<BitVectorVertexEmbedding> extensions = get_extensions (embedding, csr);
    /*if (i ==1500) {
      printf ("Embedding at 1500");
      printf_embedding (embedding);
    }*/
    VectorVertexEmbedding<embedding_size+1> extension;
    vector_embedding_from_one_less_size (embedding, extension);
    for (int j = 0; j < embedding.get_n_vertices (); j++) {
      int u = embedding.get_vertex (j);
      
      for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
        int v = csr->get_edges () [e];
        
        if (embedding.has (v) == false) {//TODO:
          
          extension.add(v);
          
          if (clique_filter_vector (csr, &extension)) {

            ((VectorVertexEmbedding<embedding_size+1>*)output_ptr)[*n_output] = extension;
            (*n_output)++;

            new_embeddings[*n_next_step] = extension;
            (*n_next_step)++;
          }
          
          extension.remove_last ();
        }
      }
    }
  }
  
  printf ("step done\n");
}

void print_embedding_bitvector (BitVectorVertexEmbedding embedding, std::ostream& os)
{
  os << "[";
  for (int u = 0; u < N; u++) {
    if (embedding.test(u)) {
      os << u << ", ";
    }
  }
  os << "]";
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
  
  csr_from_graph (csr, graph);
  //csr->print (std::cout);
  
  std::vector<VectorVertexEmbedding<0>> initial_embeddings = get_initial_embedding_vector (csr);
  std::vector<VectorVertexEmbedding<1>> output_1;
  std::vector<VectorVertexEmbedding<2>> output_2;
  std::vector<VectorVertexEmbedding<3>> output_3;
  std::vector<VectorVertexEmbedding<4>> output_4;
  std::vector<VectorVertexEmbedding<5>> output_5;
  std::vector<VectorVertexEmbedding<6>> output_6;
  std::vector<VectorVertexEmbedding<7>> output_7;
  std::vector<VectorVertexEmbedding<8>> output_8;
  
  void* embeddings;
  //filter = clique_filter;
  //process = clique_process;
  int iter = 0;
  size_t new_embeddings_size;
  {
    std::vector<VectorVertexEmbedding<1>> new_embeddings;
    run_single_step_initial_vector (&initial_embeddings[0], 1, csr, output_1, new_embeddings);
    
    embeddings = &new_embeddings[0];
    new_embeddings_size = new_embeddings.size ();
  }

  iter = 1;
  
  std::cout << "n_edges " << n_edges << std::endl;
  for (iter; iter < 10 && new_embeddings_size > 0; iter++) {
    std::cout << "iter " << iter << std::endl;
    
    char* global_mem_ptr = new char[3*1024*1024*1024UL];
    int n_embeddings = new_embeddings_size;
    size_t embedding_size = 0;
    size_t new_embedding_size = 0;
    switch (iter) {
      case 1: {
        embedding_size = sizeof (VectorVertexEmbedding<1>);
        new_embedding_size = sizeof (VectorVertexEmbedding<2>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<1>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<1>*) embeddings)[i];
        }
        break;
      }      
      case 2: {
        embedding_size = sizeof (VectorVertexEmbedding<2>);
        new_embedding_size = sizeof (VectorVertexEmbedding<3>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<2>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<2>*)embeddings)[i];
        }
        break;
      }
      
      case 3: {
        embedding_size = sizeof (VectorVertexEmbedding<3>);
        new_embedding_size = sizeof (VectorVertexEmbedding<4>);
        
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<3>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<3>*)embeddings)[i];
        }
        break;
      }
      
      case 4: {
          embedding_size = sizeof (VectorVertexEmbedding<4>);
          new_embedding_size = sizeof (VectorVertexEmbedding<5>);
          for (int i = 0; i < n_embeddings; i++) {
            ((VectorVertexEmbedding<4>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<4>*)embeddings)[i];
        }
        break;
      }
      case 5: {
        embedding_size = sizeof (VectorVertexEmbedding<5>);
        new_embedding_size = sizeof (VectorVertexEmbedding<6>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<5>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<5>*)embeddings)[i];
        }
        break;
      }
      case 6: {
        embedding_size = sizeof (VectorVertexEmbedding<6>);
        new_embedding_size = sizeof (VectorVertexEmbedding<7>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<6>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<6>*)embeddings)[i];
        }
        break;
      }
      case 7: {
        embedding_size = sizeof (VectorVertexEmbedding<7>);
        new_embedding_size = sizeof (VectorVertexEmbedding<8>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<7>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<7>*)embeddings)[i];
        }
        break;
      }
      case 8: {
        embedding_size = sizeof (VectorVertexEmbedding<8>);
        new_embedding_size = sizeof (VectorVertexEmbedding<9>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<8>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<8>*)embeddings)[i];
        }
        break;
      }
    }
    
    void* embeddings_ptr = global_mem_ptr;
    
    int n_new_embeddings = 0;
    void* new_embeddings_ptr = (char*)embeddings_ptr + (n_embeddings)*(embedding_size);
    int max_embeddings = 1000000;
    void* output_ptr = (char*)new_embeddings_ptr + (max_embeddings)*new_embedding_size;
    int n_output = 0;
    double t1 = convertTimeValToDouble (getTimeOfDay ());
    switch (iter) {
      case 1:
      {
        run_single_step_vector<1> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 2:
      {
        run_single_step_vector<2> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 3:
      {
        run_single_step_vector<3> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 4:
      {
        run_single_step_vector<4> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 5:
      {
        run_single_step_vector<5> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 6:
      {
        run_single_step_vector<6> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 7:
      {
        run_single_step_vector<7> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 8:
      {
        run_single_step_vector<8> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
    }
    
    double t2 = convertTimeValToDouble (getTimeOfDay ());
    std::cout << "step execution time " << (t2-t1) << " secs" << std::endl;
    
    new_embeddings_size = n_new_embeddings;
    
    switch (iter) {
      case 1: {
        VectorVertexEmbedding<2>* new_embeddings = new VectorVertexEmbedding<2>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<2> embedding = ((VectorVertexEmbedding<2>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_2.push_back (((VectorVertexEmbedding<2>*)output_ptr)[i]);
        }
        
        break;
      }
      
      case 2: {
        VectorVertexEmbedding<3>* new_embeddings = new VectorVertexEmbedding<3>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<3> embedding = ((VectorVertexEmbedding<3>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %ld vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_3.push_back (((VectorVertexEmbedding<3>*)output_ptr)[i]);
        }
        break;
      }
      
      case 3: {
        VectorVertexEmbedding<4>* new_embeddings = new VectorVertexEmbedding<4>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<4> embedding = ((VectorVertexEmbedding<4>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_4.push_back (((VectorVertexEmbedding<4>*)output_ptr)[i]);
        }
        break;
      }
      
      case 4: {
        VectorVertexEmbedding<5>* new_embeddings = new VectorVertexEmbedding<5>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<5> embedding = ((VectorVertexEmbedding<5>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_5.push_back (((VectorVertexEmbedding<5>*)output_ptr)[i]);
        }
        break;
      }
      
      case 5: {
        VectorVertexEmbedding<6>* new_embeddings = new VectorVertexEmbedding<6>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<6> embedding = ((VectorVertexEmbedding<6>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_6.push_back (((VectorVertexEmbedding<6>*)output_ptr)[i]);
        }
        break;
      }
      
      case 6: {
        VectorVertexEmbedding<7>* new_embeddings = new VectorVertexEmbedding<7>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<7> embedding = ((VectorVertexEmbedding<7>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_7.push_back (((VectorVertexEmbedding<7>*)output_ptr)[i]);
        }
        break;
      }
      
      case 7: {
        VectorVertexEmbedding<8>* new_embeddings = new VectorVertexEmbedding<8>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<8> embedding = ((VectorVertexEmbedding<8>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_8.push_back (((VectorVertexEmbedding<8>*)output_ptr)[i]);
        }
        break;
      }
    }

    delete[] global_mem_ptr;
    
  }
  
  std::cout << "Number of embeddings found "<< output_1.size () + output_2.size () + output_3.size () + output_4.size () + output_5.size () + output_6.size () + output_7.size () + output_8.size ()<< std::endl;
  
  /*for (auto embedding : output) {
    print_embedding (embedding, std::cout);
    std::cout << std::endl;
  }*/
}
