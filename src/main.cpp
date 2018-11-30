#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <unordered_set>

#include <string.h>
#include <assert.h>

#define __global__
#define __host__
#define __device__

#define LINE_SIZE 1024*1024
const int N = 3312;
const int N_EDGES = 9079;

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
  
//template <size_t N> using VertexEmbedding = std::bitset<N>;

class VertexEmbedding
{
private:
  unsigned char array[((N/8)+1)];

public:
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
    return ((n/8)+1)*8;
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
bool clique_filter (CSR* csr, VertexEmbedding& embedding)
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
      if (clique_filter (csr, extension)) {
        clique_process (output, extension);
        next_step.push_back (extension);
      }
    }
  }
}


__device__ 
void printf_embedding (VertexEmbedding& embedding) 
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
void run_single_step (void* input, int n_embeddings, CSR* csr,
                      void* output_ptr, 
                      int* n_output,
                      void* next_step, int* n_next_step)
{
  printf ("run_single_step: start, n_embeddings %d\n", n_embeddings);  
  VertexEmbedding* embeddings = (VertexEmbedding*)input;
  VertexEmbedding* new_embeddings = (VertexEmbedding*)next_step;
  
  //int i = blockIdx.x*blockDim.x + threadIdx.x;
  //if (i >= n_embeddings)
  //  return;
  for (int i = 0; i < n_embeddings; i++) {
    //printf ("i %d ", i);
    VertexEmbedding& embedding = embeddings[i];
    //std::vector<VertexEmbedding> extensions = get_extensions (embedding, csr);
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
            VertexEmbedding extension = VertexEmbedding(embedding);
            extension.set(v);
            /*if (i == 1500) {
              printf ("Extension at 1500");
              printf_embedding (extension);
            }*/
            
            if (clique_filter (csr, extension)) {
              /*if (i == 1500) printf ("extension passed\n");
              if (i >= 1000 && i < 2000) q[0]++;
              if (i >= 2000 && i < 3000) q[1]++;
              if (i == 1500) q[2]++;*/
              //clique_process (output, extension);
              //(*n_output)++;
              ((VertexEmbedding*)output_ptr)[*n_output] = VertexEmbedding(extension);
              (*n_output)++;
              //(*n_next_step)++; //make it atomic
              new_embeddings[*n_next_step] = VertexEmbedding(extension);
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
  
  std::cout << "n_edges " << n_edges << std::endl;
  for (iter; iter < 10 && embeddings.size () > 0; iter++) {
    std::cout << "iter " << iter << std::endl;
    char* global_mem_ptr = new char[3*1024*1024*1024UL];
    int n_embeddings = embeddings.size ();
    for (int i = 0; i < n_embeddings; i++) {
      ((VertexEmbedding*)global_mem_ptr)[i] = embeddings[i];
    }
    
    void* embeddings_ptr = global_mem_ptr;
    
    int n_new_embeddings = 0;
    void* new_embeddings_ptr = (char*)embeddings_ptr + (n_embeddings)*sizeof(VertexEmbedding);
    int max_embeddings = 1000000;
    void* output_ptr = (char*)new_embeddings_ptr + (max_embeddings)*sizeof(VertexEmbedding);
    int n_output = 0;
    run_single_step (embeddings_ptr, n_embeddings, csr, 
                     output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
    
    std::cout << "n_new_embeddings "<<n_new_embeddings<<std::endl;
    std::vector<VertexEmbedding> new_embeddings;
    for (int i = 0; i < n_new_embeddings; i++) {
      new_embeddings.push_back (((VertexEmbedding*)new_embeddings_ptr)[i]);
    }
    for (int i = 0; i < n_output; i++) {
      output.push_back (((VertexEmbedding*)output_ptr)[i]);
    }
    embeddings = new_embeddings;
    delete[] global_mem_ptr;
  }
  
  std::cout << "Number of embeddings found "<< output.size () << std::endl;
  
  /*for (auto embedding : output) {
    print_embedding (embedding, std::cout);
    std::cout << std::endl;
  }*/
}
