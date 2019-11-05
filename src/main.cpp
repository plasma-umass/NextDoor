#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <vector>
#include <bitset>
#include <unordered_set>
#include <time.h>
#include <sys/time.h>
#include <random>
#include <sys/types.h>

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

  void print (std::ostream& os)
  {
    for (int i = 0; i < filled_size; i++){
      os << get_vertex (i) << " ";
    }
  }
};


// C++ program for Huffman Coding 
#include <bits/stdc++.h> 
using namespace std; 
  
// A Huffman tree node 
struct MinHeapNode { 
  
    // One of the input characters 
    int32_t data; 
  
    // Frequency of the character 
    unsigned freq; 
  
    // Left and right child 
    MinHeapNode *left, *right; 
  
    MinHeapNode(int32_t data, unsigned freq) 
  
    { 
  
        left = right = NULL; 
        this->data = data; 
        this->freq = freq; 
    } 
}; 
  
// For comparison of 
// two heap nodes (needed in min heap) 
struct compare { 
  
    bool operator()(MinHeapNode* l, MinHeapNode* r) 
  
    { 
        return (l->freq > r->freq); 
    } 
}; 
  
// Prints huffman codes from 
// the root of Huffman Tree. 
void printCodes(std::unordered_map<int, std::string>& codes, struct MinHeapNode* root, string str) 
{ 
  if (!root) 
    return; 

  if (root->data != -1) {
    assert (root->data >= 0 && root->data < N);
    codes[root->data] = str; 
  }

  printCodes(codes, root->left, str + "0"); 
  printCodes(codes, root->right, str + "1"); 
}
  
// The main function that builds a Huffman Tree and 
// print codes by traversing the built Huffman Tree 
void HuffmanCodes(std::vector<int>& data, std::unordered_map<int, int>& freq,
                  std::unordered_map<int, std::string>& codes)

{ 
    struct MinHeapNode *left, *right, *top; 
  
    // Create a min heap & inserts all characters of data[] 
    priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap; 
  
    for (size_t i = 0; i < data.size (); ++i) {
      assert (data[i]>= 0 && data[i] < N);
      //std::cout << "d " << data[i] << " " << freq[data[i]] << std::endl;
      minHeap.push(new MinHeapNode(data[i], freq[data[i]])); 
    }
  
    // Iterate while size of heap doesn't become 1 
    while (minHeap.size() != 1) { 
  
        // Extract the two minimum 
        // freq items from min heap 
        left = minHeap.top(); 
        minHeap.pop(); 
  
        right = minHeap.top(); 
        minHeap.pop(); 
  
        // Create a new internal node with 
        // frequency equal to the sum of the 
        // two nodes frequencies. Make the 
        // two extracted node as left and right children 
        // of this new node. Add this node 
        // to the min heap '$' is a special value 
        // for internal nodes, not used 
        top = new MinHeapNode(-1, left->freq + right->freq); 
  
        top->left = left; 
        top->right = right; 
  
        minHeap.push(top); 
    } 
  
    // Print Huffman codes using 
    // the Huffman tree built above 
    printCodes(codes, minHeap.top(), ""); 
} 

template <size_t size>
size_t get_vertex_frequencies (std::unordered_map<int, int>& vertex_freq, 
                               VectorVertexEmbedding<size>* embeddings, size_t n_embeddings)
{
  for (size_t i = 0; i < n_embeddings; i++) {
    VectorVertexEmbedding<size>& embedding = embeddings[i];
    for (size_t j = 0; j < embedding.get_n_vertices (); j++) {
      int v = embedding.get_vertex (j);
      if (vertex_freq.find (v) == vertex_freq.end ()) {
        vertex_freq[v] = 0;
      }

      vertex_freq[v] += 1;
    }
  }
}

template<size_t size>
std::string encode_embedding (VectorVertexEmbedding<size>& embedding, std::unordered_map<int, std::string>& codes)
{
  std::string encoding = "";
  for (int i = 0; i < embedding.get_n_vertices (); i++) {
    int v = embedding.get_vertex (i);
    assert (codes.find(v) != codes.end ());
    encoding += codes[v];
  }
  return encoding;
}

template<size_t size>
void perform_huffman_encoding (VectorVertexEmbedding<size>* embeddings, size_t n_embeddings)
{
  return;
  size_t encoded_size = 0;
  size_t lookup_table_size = 0;
  //For all vertices
  for (int v = 0; v < N; v++) {
    //if (v != 3193)
    //  continue;
    size_t start_v = -1;
    size_t end_v = -1;
    for (int i=0; i < n_embeddings; i++) {
      if (embeddings[i].get_vertex(0) == v) {
        start_v = i;
        break;
      }
    }

    if (start_v == -1)
      continue;

    for (int i = start_v; i < n_embeddings; i++) {
      if (embeddings[i].get_vertex (0) != v) {
        end_v = i;
        break;
      }
    }    
    
    if (end_v == -1) {
      end_v = n_embeddings-1;
    }
    //end_2984 = end_2984 - 10;
    //std::cout << "start_2984 " << start_2984 << " end_2984 " << end_2984 << std::endl;
    int parts = 4;
    size_t _n_embeddings = end_v+1 - start_v;
    size_t each_part_size = _n_embeddings/parts; //std::min (_n_embeddings, 10UL); 
    //parts = _n_embeddings/each_part_size;
    size_t total_size = 0;

    for (int part = 0; part < parts; part++) {
      std::unordered_map<int, int> vertex_freq;
      std::unordered_map<int, std::string> codes;
      vertex_freq.clear ();
      codes.clear ();
      int start = start_v + part*each_part_size;
      VectorVertexEmbedding<size>* _embeddings = &embeddings[start];
      get_vertex_frequencies (vertex_freq, _embeddings, each_part_size);
      std::vector <int> data;
      for (size_t i = 0; i < each_part_size; i++) {
        VectorVertexEmbedding<size>& e = _embeddings[i];
        assert (e.get_n_vertices() == size);
        for (int j = 0; j < e.get_n_vertices (); j++) {
          int v = e.get_vertex (j);
          //std::cout << "v : " << v << std::endl;
          data.push_back (v);
        }
      }
    
      //TODO: We can optimize data here too. No need to create another data array.
      
      HuffmanCodes (data, vertex_freq, codes);
      size_t max_code_size = 0;
      for (auto iter : codes) {
        max_code_size = std::max (max_code_size, iter.second.length ());
        //std::cout << iter.first << " : " << iter.second << std::endl;
      }
    #if 1
      if (max_code_size < 8) max_code_size = 8;
      else if (max_code_size < 16) max_code_size = 16;
      //else if (max_code_size < 24) max_code_size = 24;
      //else if (max_code_size < 32) max_code_size = 32;
      else assert (false);
    #endif

      for (auto iter : vertex_freq) {
        //std::cout << iter.first << " : " << iter.second << std::endl;
      }

      std::cout << "distinct codes " << codes.size () << std::endl;
      std::cout << "vertex_freqs " << vertex_freq.size () << std::endl;
      assert (vertex_freq.size () == codes.size ());
      
      //lookup_table_size += powl (2, max_code_size);
      lookup_table_size += codes.size()*max_code_size/8;
      double_t avg_encoding_len = 0;
      size_t max_encoding_len = 0;
      for (size_t i = 0; i < each_part_size; i++) {
        std::string encoding = encode_embedding (_embeddings[i], codes);
        avg_encoding_len += encoding.length();
        max_encoding_len = std::max(max_encoding_len, encoding.length ());
        if (false and encoding.length () == 53) {
          std::cout << encoding << std::endl;
          _embeddings[i].print (std::cout);
          std::cout << std::endl;
        }

        if (false and _embeddings[i].has (3154)) {
          std::cout << "embedding has 3154 "<< std::endl;
        }
      }

      avg_encoding_len = avg_encoding_len/each_part_size;
      std::cout << "v: " << v << " part_size " << each_part_size << " max_code_size = " 
        <<  max_code_size << "bits avg_encoding_len = " << avg_encoding_len << " max_encoding_len = " << max_encoding_len << std::endl;      
      total_size += max_code_size*size*each_part_size/8;
    }

    encoded_size += total_size;
    std::cout << "v: " << v << " encoding_size: " << total_size << " bytes original_size: " << _n_embeddings*size*4 << " bytes"<<std::endl;
  }

  std::cout << "encoded_size: " << encoded_size << " bytes " << " uncompressed_size: " << n_embeddings*size*4 << " lookup_table_size " << lookup_table_size << std::endl;
}

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
bool is_embedding_canonical (CSR* csr, VectorVertexEmbedding<embedding_size>& embedding, int v)
{
  if (embedding.get_vertex (0) > v)
    return false;

  bool found_neighbor = false;
  for (int j = 0; j < embedding.get_n_vertices (); j++) {
    int v_j = embedding.get_vertex (j);
    if (found_neighbor == false && csr->has_edge (v_j, v)) {
      found_neighbor = true;
    } else if (found_neighbor == true && v_j > v) {
      return false;
    }
  }

  return true;
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
    if (false) {
      VectorVertexEmbedding<embedding_size+1> extension;
      vector_embedding_from_one_less_size (embedding, extension);
      for (int j = 0; j < embedding.get_n_vertices (); j++) {
        int u = embedding.get_vertex (j);
        
        for (int e = csr->get_start_edge_idx(u); e <= csr->get_end_edge_idx(u); e++) {
          int v = csr->get_edges () [e];

          if (embedding.has (v) == false && 
              is_embedding_canonical (csr, embedding, v)) {
            
            if (embedding_size == 2) {
              if (embedding.get_vertex (0) == 1372 && embedding.get_vertex (1) == 2171) {
                std::cout << "For 1372 2171 " << u << " " << v << std::endl;
              }
            }
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
    } else {
      VectorVertexEmbedding<embedding_size+1> extension;
      vector_embedding_from_one_less_size (embedding, extension);
      int u = embedding.get_vertex (embedding.get_n_vertices () - 1);
      if (csr->get_end_edge_idx(u) == -1)
        continue;

      int n_edges = csr->get_end_edge_idx(u) - csr->get_start_edge_idx(u) + 1;
      
      int v = csr->get_edges () [csr->get_start_edge_idx(u) + rand() % n_edges];

      extension.add(v);
      
      if (true) {

        ((VectorVertexEmbedding<embedding_size+1>*)output_ptr)[*n_output] = extension;
        (*n_output)++;

        new_embeddings[*n_next_step] = extension;
        (*n_next_step)++;
      }
      
      extension.remove_last ();
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

#define HOST_TO_DEVICE_CASE(X) case X: {\
        embedding_size = sizeof (VectorVertexEmbedding<X>);\
        new_embedding_size = sizeof (VectorVertexEmbedding<(X+1)>);\
        for (int i = 0; i < n_embeddings; i++) {\
          ((VectorVertexEmbedding<X>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<X>*)embeddings)[i];}\
          break;}

#define RUN_KERNEL_CASE(X) case X:{\
        run_single_step_vector<X> (embeddings_ptr, n_embeddings, csr, \
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);\
        break;}

#define DEVICE_TO_HOST_CASE(X,X1) case X: {\
        VectorVertexEmbedding<X+1>* new_embeddings = new VectorVertexEmbedding<X+1>[n_new_embeddings];\
        for (int i = 0; i < n_new_embeddings; i++) {\
          VectorVertexEmbedding<X+1> embedding = ((VectorVertexEmbedding<X+1>*)new_embeddings_ptr)[i];\
          new_embeddings [i] = embedding;}\
        embeddings = &new_embeddings[0];\
        for (int i = 0; i < n_output; i++) {output_##X1.push_back (((VectorVertexEmbedding<X+1>*)output_ptr)[i]);} break; }
#define DECLARE_OUTPUT(X)   std::vector<VectorVertexEmbedding<X>> output_##X;

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
  float total_time = 0.0;
  std::vector<VectorVertexEmbedding<0>> initial_embeddings = get_initial_embedding_vector (csr);
  std::vector<VectorVertexEmbedding<1>> output_1;
  std::vector<VectorVertexEmbedding<2>> output_2;
  std::vector<VectorVertexEmbedding<3>> output_3;
  std::vector<VectorVertexEmbedding<4>> output_4;
  std::vector<VectorVertexEmbedding<5>> output_5;
  std::vector<VectorVertexEmbedding<6>> output_6;
  std::vector<VectorVertexEmbedding<7>> output_7;
  std::vector<VectorVertexEmbedding<8>> output_8;
  std::vector<VectorVertexEmbedding<9>> output_9;
  std::vector<VectorVertexEmbedding<10>> output_10;
  std::vector<VectorVertexEmbedding<11>> output_11;
  std::vector<VectorVertexEmbedding<12>> output_12;
  std::vector<VectorVertexEmbedding<13>> output_13;
  std::vector<VectorVertexEmbedding<14>> output_14;
  std::vector<VectorVertexEmbedding<15>> output_15;
  std::vector<VectorVertexEmbedding<16>> output_16;
  std::vector<VectorVertexEmbedding<17>> output_17;
  std::vector<VectorVertexEmbedding<18>> output_18;
  std::vector<VectorVertexEmbedding<19>> output_19;
  std::vector<VectorVertexEmbedding<20>> output_20;
  DECLARE_OUTPUT(21)
  DECLARE_OUTPUT(22)
  DECLARE_OUTPUT(23)
  DECLARE_OUTPUT(24)
  DECLARE_OUTPUT(25)
  DECLARE_OUTPUT(26)
  DECLARE_OUTPUT(27)
  DECLARE_OUTPUT(28)
  DECLARE_OUTPUT(29)
  DECLARE_OUTPUT(30)
DECLARE_OUTPUT(31)
DECLARE_OUTPUT(32)
DECLARE_OUTPUT(33)
DECLARE_OUTPUT(34)
DECLARE_OUTPUT(35)
DECLARE_OUTPUT(36)
DECLARE_OUTPUT(37)
DECLARE_OUTPUT(38)
DECLARE_OUTPUT(39)
DECLARE_OUTPUT(40)
DECLARE_OUTPUT(41)
DECLARE_OUTPUT(42)
DECLARE_OUTPUT(43)
DECLARE_OUTPUT(44)
DECLARE_OUTPUT(45)
DECLARE_OUTPUT(46)
DECLARE_OUTPUT(47)
DECLARE_OUTPUT(48)
DECLARE_OUTPUT(49)
DECLARE_OUTPUT(50)
DECLARE_OUTPUT(51)
DECLARE_OUTPUT(52)
DECLARE_OUTPUT(53)
DECLARE_OUTPUT(54)
DECLARE_OUTPUT(55)
DECLARE_OUTPUT(56)
DECLARE_OUTPUT(57)
DECLARE_OUTPUT(58)
DECLARE_OUTPUT(59)
DECLARE_OUTPUT(60)
DECLARE_OUTPUT(61)
DECLARE_OUTPUT(62)
DECLARE_OUTPUT(63)
DECLARE_OUTPUT(64)
DECLARE_OUTPUT(65)
DECLARE_OUTPUT(66)
DECLARE_OUTPUT(67)
DECLARE_OUTPUT(68)
DECLARE_OUTPUT(69)
DECLARE_OUTPUT(70)
DECLARE_OUTPUT(71)
DECLARE_OUTPUT(72)
DECLARE_OUTPUT(73)
DECLARE_OUTPUT(74)
DECLARE_OUTPUT(75)
DECLARE_OUTPUT(76)
DECLARE_OUTPUT(77)
DECLARE_OUTPUT(78)
DECLARE_OUTPUT(79)
DECLARE_OUTPUT(80)
DECLARE_OUTPUT(81)
DECLARE_OUTPUT(82)
DECLARE_OUTPUT(83)
DECLARE_OUTPUT(84)
DECLARE_OUTPUT(85)
DECLARE_OUTPUT(86)
DECLARE_OUTPUT(87)
DECLARE_OUTPUT(88)
DECLARE_OUTPUT(89)
DECLARE_OUTPUT(90)
DECLARE_OUTPUT(91)
DECLARE_OUTPUT(92)
DECLARE_OUTPUT(93)
DECLARE_OUTPUT(94)
DECLARE_OUTPUT(95)
DECLARE_OUTPUT(96)
DECLARE_OUTPUT(97)
DECLARE_OUTPUT(98)
DECLARE_OUTPUT(99)
DECLARE_OUTPUT(100)
/*DECLARE_OUTPUT(101)
DECLARE_OUTPUT(102)
DECLARE_OUTPUT(103)
DECLARE_OUTPUT(104)
DECLARE_OUTPUT(105)
DECLARE_OUTPUT(106)
DECLARE_OUTPUT(107)
DECLARE_OUTPUT(108)
DECLARE_OUTPUT(109)
DECLARE_OUTPUT(110)
DECLARE_OUTPUT(111)
DECLARE_OUTPUT(112)
DECLARE_OUTPUT(113)
DECLARE_OUTPUT(114)
DECLARE_OUTPUT(115)
DECLARE_OUTPUT(116)
DECLARE_OUTPUT(117)
DECLARE_OUTPUT(118)
DECLARE_OUTPUT(119)
DECLARE_OUTPUT(120)
DECLARE_OUTPUT(121)
DECLARE_OUTPUT(122)
DECLARE_OUTPUT(123)
DECLARE_OUTPUT(124)
DECLARE_OUTPUT(125)
DECLARE_OUTPUT(126)
DECLARE_OUTPUT(127)
DECLARE_OUTPUT(128)
DECLARE_OUTPUT(129)
DECLARE_OUTPUT(130)
DECLARE_OUTPUT(131)
DECLARE_OUTPUT(132)
DECLARE_OUTPUT(133)
DECLARE_OUTPUT(134)
DECLARE_OUTPUT(135)
DECLARE_OUTPUT(136)
DECLARE_OUTPUT(137)
DECLARE_OUTPUT(138)
DECLARE_OUTPUT(139)
DECLARE_OUTPUT(140)
DECLARE_OUTPUT(141)
DECLARE_OUTPUT(142)
DECLARE_OUTPUT(143)
DECLARE_OUTPUT(144)
DECLARE_OUTPUT(145)
DECLARE_OUTPUT(146)
DECLARE_OUTPUT(147)
DECLARE_OUTPUT(148)
DECLARE_OUTPUT(149)
DECLARE_OUTPUT(150)
DECLARE_OUTPUT(151)
DECLARE_OUTPUT(152)
DECLARE_OUTPUT(153)
DECLARE_OUTPUT(154)
DECLARE_OUTPUT(155)
DECLARE_OUTPUT(156)
DECLARE_OUTPUT(157)
DECLARE_OUTPUT(158)
DECLARE_OUTPUT(159)
DECLARE_OUTPUT(160)
DECLARE_OUTPUT(161)
DECLARE_OUTPUT(162)
DECLARE_OUTPUT(163)
DECLARE_OUTPUT(164)
DECLARE_OUTPUT(165)
DECLARE_OUTPUT(166)
DECLARE_OUTPUT(167)
DECLARE_OUTPUT(168)
DECLARE_OUTPUT(169)
DECLARE_OUTPUT(170)
DECLARE_OUTPUT(171)
DECLARE_OUTPUT(172)
DECLARE_OUTPUT(173)
DECLARE_OUTPUT(174)
DECLARE_OUTPUT(175)
DECLARE_OUTPUT(176)
DECLARE_OUTPUT(177)
DECLARE_OUTPUT(178)
DECLARE_OUTPUT(179)
DECLARE_OUTPUT(180)
DECLARE_OUTPUT(181)
DECLARE_OUTPUT(182)
DECLARE_OUTPUT(183)
DECLARE_OUTPUT(184)
DECLARE_OUTPUT(185)
DECLARE_OUTPUT(186)
DECLARE_OUTPUT(187)
DECLARE_OUTPUT(188)
DECLARE_OUTPUT(189)
DECLARE_OUTPUT(190)
DECLARE_OUTPUT(191)
DECLARE_OUTPUT(192)
DECLARE_OUTPUT(193)
DECLARE_OUTPUT(194)
DECLARE_OUTPUT(195)
DECLARE_OUTPUT(196)
DECLARE_OUTPUT(197)
DECLARE_OUTPUT(198)
DECLARE_OUTPUT(199)
DECLARE_OUTPUT(200)
DECLARE_OUTPUT(201)
DECLARE_OUTPUT(202)
DECLARE_OUTPUT(203)
DECLARE_OUTPUT(204)
DECLARE_OUTPUT(205)
DECLARE_OUTPUT(206)
DECLARE_OUTPUT(207)
DECLARE_OUTPUT(208)
DECLARE_OUTPUT(209)
DECLARE_OUTPUT(210)
DECLARE_OUTPUT(211)
DECLARE_OUTPUT(212)
DECLARE_OUTPUT(213)
DECLARE_OUTPUT(214)
DECLARE_OUTPUT(215)
DECLARE_OUTPUT(216)
DECLARE_OUTPUT(217)
DECLARE_OUTPUT(218)
DECLARE_OUTPUT(219)
DECLARE_OUTPUT(220)
DECLARE_OUTPUT(221)
DECLARE_OUTPUT(222)
DECLARE_OUTPUT(223)
DECLARE_OUTPUT(224)
DECLARE_OUTPUT(225)
DECLARE_OUTPUT(226)
DECLARE_OUTPUT(227)
DECLARE_OUTPUT(228)
DECLARE_OUTPUT(229)
DECLARE_OUTPUT(230)
DECLARE_OUTPUT(231)
DECLARE_OUTPUT(232)
DECLARE_OUTPUT(233)
DECLARE_OUTPUT(234)
DECLARE_OUTPUT(235)
DECLARE_OUTPUT(236)
DECLARE_OUTPUT(237)
DECLARE_OUTPUT(238)
DECLARE_OUTPUT(239)
DECLARE_OUTPUT(240)
DECLARE_OUTPUT(241)
DECLARE_OUTPUT(242)
DECLARE_OUTPUT(243)
DECLARE_OUTPUT(244)
DECLARE_OUTPUT(245)
DECLARE_OUTPUT(246)
DECLARE_OUTPUT(247)
DECLARE_OUTPUT(248)
DECLARE_OUTPUT(249)
DECLARE_OUTPUT(250)
DECLARE_OUTPUT(251)
DECLARE_OUTPUT(252)
DECLARE_OUTPUT(253)
DECLARE_OUTPUT(254)
DECLARE_OUTPUT(255)
DECLARE_OUTPUT(256)
DECLARE_OUTPUT(257)
DECLARE_OUTPUT(258)
DECLARE_OUTPUT(259)
DECLARE_OUTPUT(260)
DECLARE_OUTPUT(261)
DECLARE_OUTPUT(262)
DECLARE_OUTPUT(263)
DECLARE_OUTPUT(264)
DECLARE_OUTPUT(265)
DECLARE_OUTPUT(266)
DECLARE_OUTPUT(267)
DECLARE_OUTPUT(268)
DECLARE_OUTPUT(269)
DECLARE_OUTPUT(270)
DECLARE_OUTPUT(271)
DECLARE_OUTPUT(272)
DECLARE_OUTPUT(273)
DECLARE_OUTPUT(274)
DECLARE_OUTPUT(275)
DECLARE_OUTPUT(276)
DECLARE_OUTPUT(277)
DECLARE_OUTPUT(278)
DECLARE_OUTPUT(279)
DECLARE_OUTPUT(280)
DECLARE_OUTPUT(281)
DECLARE_OUTPUT(282)
DECLARE_OUTPUT(283)
DECLARE_OUTPUT(284)
DECLARE_OUTPUT(285)
DECLARE_OUTPUT(286)
DECLARE_OUTPUT(287)
DECLARE_OUTPUT(288)
DECLARE_OUTPUT(289)
DECLARE_OUTPUT(290)
DECLARE_OUTPUT(291)
DECLARE_OUTPUT(292)
DECLARE_OUTPUT(293)
DECLARE_OUTPUT(294)
DECLARE_OUTPUT(295)
DECLARE_OUTPUT(296)
DECLARE_OUTPUT(297)
DECLARE_OUTPUT(298)
DECLARE_OUTPUT(299)
DECLARE_OUTPUT(300)
DECLARE_OUTPUT(301)
DECLARE_OUTPUT(302)
DECLARE_OUTPUT(303)
DECLARE_OUTPUT(304)
DECLARE_OUTPUT(305)
DECLARE_OUTPUT(306)
DECLARE_OUTPUT(307)
DECLARE_OUTPUT(308)
DECLARE_OUTPUT(309)
DECLARE_OUTPUT(310)
DECLARE_OUTPUT(311)
DECLARE_OUTPUT(312)
DECLARE_OUTPUT(313)
DECLARE_OUTPUT(314)
DECLARE_OUTPUT(315)
DECLARE_OUTPUT(316)
DECLARE_OUTPUT(317)
DECLARE_OUTPUT(318)
DECLARE_OUTPUT(319)
DECLARE_OUTPUT(320)
DECLARE_OUTPUT(321)
DECLARE_OUTPUT(322)
DECLARE_OUTPUT(323)
DECLARE_OUTPUT(324)
DECLARE_OUTPUT(325)
DECLARE_OUTPUT(326)
DECLARE_OUTPUT(327)
DECLARE_OUTPUT(328)
DECLARE_OUTPUT(329)
DECLARE_OUTPUT(330)
DECLARE_OUTPUT(331)
DECLARE_OUTPUT(332)
DECLARE_OUTPUT(333)
DECLARE_OUTPUT(334)
DECLARE_OUTPUT(335)
DECLARE_OUTPUT(336)
DECLARE_OUTPUT(337)
DECLARE_OUTPUT(338)
DECLARE_OUTPUT(339)
DECLARE_OUTPUT(340)
DECLARE_OUTPUT(341)
DECLARE_OUTPUT(342)
DECLARE_OUTPUT(343)
DECLARE_OUTPUT(344)
DECLARE_OUTPUT(345)
DECLARE_OUTPUT(346)
DECLARE_OUTPUT(347)
DECLARE_OUTPUT(348)
DECLARE_OUTPUT(349)
DECLARE_OUTPUT(350)
DECLARE_OUTPUT(351)
DECLARE_OUTPUT(352)
DECLARE_OUTPUT(353)
DECLARE_OUTPUT(354)
DECLARE_OUTPUT(355)
DECLARE_OUTPUT(356)
DECLARE_OUTPUT(357)
DECLARE_OUTPUT(358)
DECLARE_OUTPUT(359)
DECLARE_OUTPUT(360)
DECLARE_OUTPUT(361)
DECLARE_OUTPUT(362)
DECLARE_OUTPUT(363)
DECLARE_OUTPUT(364)
DECLARE_OUTPUT(365)
DECLARE_OUTPUT(366)
DECLARE_OUTPUT(367)
DECLARE_OUTPUT(368)
DECLARE_OUTPUT(369)
DECLARE_OUTPUT(370)
DECLARE_OUTPUT(371)
DECLARE_OUTPUT(372)
DECLARE_OUTPUT(373)
DECLARE_OUTPUT(374)
DECLARE_OUTPUT(375)
DECLARE_OUTPUT(376)
DECLARE_OUTPUT(377)
DECLARE_OUTPUT(378)
DECLARE_OUTPUT(379)
DECLARE_OUTPUT(380)
DECLARE_OUTPUT(381)
DECLARE_OUTPUT(382)
DECLARE_OUTPUT(383)
DECLARE_OUTPUT(384)
DECLARE_OUTPUT(385)
DECLARE_OUTPUT(386)
DECLARE_OUTPUT(387)
DECLARE_OUTPUT(388)
DECLARE_OUTPUT(389)
DECLARE_OUTPUT(390)
DECLARE_OUTPUT(391)
DECLARE_OUTPUT(392)
DECLARE_OUTPUT(393)
DECLARE_OUTPUT(394)
DECLARE_OUTPUT(395)
DECLARE_OUTPUT(396)
DECLARE_OUTPUT(397)
DECLARE_OUTPUT(398)
DECLARE_OUTPUT(399)
DECLARE_OUTPUT(400)
DECLARE_OUTPUT(401)
DECLARE_OUTPUT(402)
DECLARE_OUTPUT(403)
DECLARE_OUTPUT(404)
DECLARE_OUTPUT(405)
DECLARE_OUTPUT(406)
DECLARE_OUTPUT(407)
DECLARE_OUTPUT(408)
DECLARE_OUTPUT(409)
DECLARE_OUTPUT(410)
DECLARE_OUTPUT(411)
DECLARE_OUTPUT(412)
DECLARE_OUTPUT(413)
DECLARE_OUTPUT(414)
DECLARE_OUTPUT(415)
DECLARE_OUTPUT(416)
DECLARE_OUTPUT(417)
DECLARE_OUTPUT(418)
DECLARE_OUTPUT(419)
DECLARE_OUTPUT(420)
DECLARE_OUTPUT(421)
DECLARE_OUTPUT(422)
DECLARE_OUTPUT(423)
DECLARE_OUTPUT(424)
DECLARE_OUTPUT(425)
DECLARE_OUTPUT(426)
DECLARE_OUTPUT(427)
DECLARE_OUTPUT(428)
DECLARE_OUTPUT(429)
DECLARE_OUTPUT(430)
DECLARE_OUTPUT(431)
DECLARE_OUTPUT(432)
DECLARE_OUTPUT(433)
DECLARE_OUTPUT(434)
DECLARE_OUTPUT(435)
DECLARE_OUTPUT(436)
DECLARE_OUTPUT(437)
DECLARE_OUTPUT(438)
DECLARE_OUTPUT(439)
DECLARE_OUTPUT(440)
DECLARE_OUTPUT(441)
DECLARE_OUTPUT(442)
DECLARE_OUTPUT(443)
DECLARE_OUTPUT(444)
DECLARE_OUTPUT(445)
DECLARE_OUTPUT(446)
DECLARE_OUTPUT(447)
DECLARE_OUTPUT(448)
DECLARE_OUTPUT(449)
DECLARE_OUTPUT(450)
DECLARE_OUTPUT(451)
DECLARE_OUTPUT(452)
DECLARE_OUTPUT(453)
DECLARE_OUTPUT(454)
DECLARE_OUTPUT(455)
DECLARE_OUTPUT(456)
DECLARE_OUTPUT(457)
DECLARE_OUTPUT(458)
DECLARE_OUTPUT(459)
DECLARE_OUTPUT(460)
DECLARE_OUTPUT(461)
DECLARE_OUTPUT(462)
DECLARE_OUTPUT(463)
DECLARE_OUTPUT(464)
DECLARE_OUTPUT(465)
DECLARE_OUTPUT(466)
DECLARE_OUTPUT(467)
DECLARE_OUTPUT(468)
DECLARE_OUTPUT(469)
DECLARE_OUTPUT(470)
DECLARE_OUTPUT(471)
DECLARE_OUTPUT(472)
DECLARE_OUTPUT(473)
DECLARE_OUTPUT(474)
DECLARE_OUTPUT(475)
DECLARE_OUTPUT(476)
DECLARE_OUTPUT(477)
DECLARE_OUTPUT(478)
DECLARE_OUTPUT(479)
DECLARE_OUTPUT(480)
DECLARE_OUTPUT(481)
DECLARE_OUTPUT(482)
DECLARE_OUTPUT(483)
DECLARE_OUTPUT(484)
DECLARE_OUTPUT(485)
DECLARE_OUTPUT(486)
DECLARE_OUTPUT(487)
DECLARE_OUTPUT(488)
DECLARE_OUTPUT(489)
DECLARE_OUTPUT(490)
DECLARE_OUTPUT(491)
DECLARE_OUTPUT(492)
DECLARE_OUTPUT(493)
DECLARE_OUTPUT(494)
DECLARE_OUTPUT(495)
DECLARE_OUTPUT(496)
DECLARE_OUTPUT(497)
DECLARE_OUTPUT(498)
DECLARE_OUTPUT(499)
DECLARE_OUTPUT(500)
DECLARE_OUTPUT(501)
DECLARE_OUTPUT(502)
DECLARE_OUTPUT(503)
DECLARE_OUTPUT(504)
DECLARE_OUTPUT(505)
DECLARE_OUTPUT(506)
DECLARE_OUTPUT(507)
DECLARE_OUTPUT(508)
DECLARE_OUTPUT(509)
DECLARE_OUTPUT(510)
DECLARE_OUTPUT(511)
DECLARE_OUTPUT(512)
DECLARE_OUTPUT(513)
DECLARE_OUTPUT(514)
DECLARE_OUTPUT(515)
DECLARE_OUTPUT(516)
DECLARE_OUTPUT(517)
DECLARE_OUTPUT(518)
DECLARE_OUTPUT(519)
DECLARE_OUTPUT(520)
DECLARE_OUTPUT(521)
DECLARE_OUTPUT(522)
DECLARE_OUTPUT(523)
DECLARE_OUTPUT(524)
DECLARE_OUTPUT(525)
DECLARE_OUTPUT(526)
DECLARE_OUTPUT(527)
DECLARE_OUTPUT(528)
DECLARE_OUTPUT(529)
DECLARE_OUTPUT(530)
DECLARE_OUTPUT(531)
DECLARE_OUTPUT(532)
DECLARE_OUTPUT(533)
DECLARE_OUTPUT(534)
DECLARE_OUTPUT(535)
DECLARE_OUTPUT(536)
DECLARE_OUTPUT(537)
DECLARE_OUTPUT(538)
DECLARE_OUTPUT(539)
DECLARE_OUTPUT(540)
DECLARE_OUTPUT(541)
DECLARE_OUTPUT(542)
DECLARE_OUTPUT(543)
DECLARE_OUTPUT(544)
DECLARE_OUTPUT(545)
DECLARE_OUTPUT(546)
DECLARE_OUTPUT(547)
DECLARE_OUTPUT(548)
DECLARE_OUTPUT(549)
DECLARE_OUTPUT(550)
DECLARE_OUTPUT(551)
DECLARE_OUTPUT(552)
DECLARE_OUTPUT(553)
DECLARE_OUTPUT(554)
DECLARE_OUTPUT(555)
DECLARE_OUTPUT(556)
DECLARE_OUTPUT(557)
DECLARE_OUTPUT(558)
DECLARE_OUTPUT(559)
DECLARE_OUTPUT(560)
DECLARE_OUTPUT(561)
DECLARE_OUTPUT(562)
DECLARE_OUTPUT(563)
DECLARE_OUTPUT(564)
DECLARE_OUTPUT(565)
DECLARE_OUTPUT(566)
DECLARE_OUTPUT(567)
DECLARE_OUTPUT(568)
DECLARE_OUTPUT(569)
DECLARE_OUTPUT(570)
DECLARE_OUTPUT(571)
DECLARE_OUTPUT(572)
DECLARE_OUTPUT(573)
DECLARE_OUTPUT(574)
DECLARE_OUTPUT(575)
DECLARE_OUTPUT(576)
DECLARE_OUTPUT(577)
DECLARE_OUTPUT(578)
DECLARE_OUTPUT(579)
DECLARE_OUTPUT(580)
DECLARE_OUTPUT(581)
DECLARE_OUTPUT(582)
DECLARE_OUTPUT(583)
DECLARE_OUTPUT(584)
DECLARE_OUTPUT(585)
DECLARE_OUTPUT(586)
DECLARE_OUTPUT(587)
DECLARE_OUTPUT(588)
DECLARE_OUTPUT(589)
DECLARE_OUTPUT(590)
DECLARE_OUTPUT(591)
DECLARE_OUTPUT(592)
DECLARE_OUTPUT(593)
DECLARE_OUTPUT(594)
DECLARE_OUTPUT(595)
DECLARE_OUTPUT(596)
DECLARE_OUTPUT(597)
DECLARE_OUTPUT(598)
DECLARE_OUTPUT(599)
DECLARE_OUTPUT(600)
DECLARE_OUTPUT(601)
DECLARE_OUTPUT(602)
DECLARE_OUTPUT(603)
DECLARE_OUTPUT(604)
DECLARE_OUTPUT(605)
DECLARE_OUTPUT(606)
DECLARE_OUTPUT(607)
DECLARE_OUTPUT(608)
DECLARE_OUTPUT(609)
DECLARE_OUTPUT(610)
DECLARE_OUTPUT(611)
DECLARE_OUTPUT(612)
DECLARE_OUTPUT(613)
DECLARE_OUTPUT(614)
DECLARE_OUTPUT(615)
DECLARE_OUTPUT(616)
DECLARE_OUTPUT(617)
DECLARE_OUTPUT(618)
DECLARE_OUTPUT(619)
DECLARE_OUTPUT(620)
DECLARE_OUTPUT(621)
DECLARE_OUTPUT(622)
DECLARE_OUTPUT(623)
DECLARE_OUTPUT(624)
DECLARE_OUTPUT(625)
DECLARE_OUTPUT(626)
DECLARE_OUTPUT(627)
DECLARE_OUTPUT(628)
DECLARE_OUTPUT(629)
DECLARE_OUTPUT(630)
DECLARE_OUTPUT(631)
DECLARE_OUTPUT(632)
DECLARE_OUTPUT(633)
DECLARE_OUTPUT(634)
DECLARE_OUTPUT(635)
DECLARE_OUTPUT(636)
DECLARE_OUTPUT(637)
DECLARE_OUTPUT(638)
DECLARE_OUTPUT(639)
DECLARE_OUTPUT(640)
DECLARE_OUTPUT(641)
DECLARE_OUTPUT(642)
DECLARE_OUTPUT(643)
DECLARE_OUTPUT(644)
DECLARE_OUTPUT(645)
DECLARE_OUTPUT(646)
DECLARE_OUTPUT(647)
DECLARE_OUTPUT(648)
DECLARE_OUTPUT(649)
DECLARE_OUTPUT(650)
DECLARE_OUTPUT(651)
DECLARE_OUTPUT(652)
DECLARE_OUTPUT(653)
DECLARE_OUTPUT(654)
DECLARE_OUTPUT(655)
DECLARE_OUTPUT(656)
DECLARE_OUTPUT(657)
DECLARE_OUTPUT(658)
DECLARE_OUTPUT(659)
DECLARE_OUTPUT(660)
DECLARE_OUTPUT(661)
DECLARE_OUTPUT(662)
DECLARE_OUTPUT(663)
DECLARE_OUTPUT(664)
DECLARE_OUTPUT(665)
DECLARE_OUTPUT(666)
DECLARE_OUTPUT(667)
DECLARE_OUTPUT(668)
DECLARE_OUTPUT(669)
DECLARE_OUTPUT(670)
DECLARE_OUTPUT(671)
DECLARE_OUTPUT(672)
DECLARE_OUTPUT(673)
DECLARE_OUTPUT(674)
DECLARE_OUTPUT(675)
DECLARE_OUTPUT(676)
DECLARE_OUTPUT(677)
DECLARE_OUTPUT(678)
DECLARE_OUTPUT(679)
DECLARE_OUTPUT(680)
DECLARE_OUTPUT(681)
DECLARE_OUTPUT(682)
DECLARE_OUTPUT(683)
DECLARE_OUTPUT(684)
DECLARE_OUTPUT(685)
DECLARE_OUTPUT(686)
DECLARE_OUTPUT(687)
DECLARE_OUTPUT(688)
DECLARE_OUTPUT(689)
DECLARE_OUTPUT(690)
DECLARE_OUTPUT(691)
DECLARE_OUTPUT(692)
DECLARE_OUTPUT(693)
DECLARE_OUTPUT(694)
DECLARE_OUTPUT(695)
DECLARE_OUTPUT(696)
DECLARE_OUTPUT(697)
DECLARE_OUTPUT(698)
DECLARE_OUTPUT(699)
DECLARE_OUTPUT(700)
DECLARE_OUTPUT(701)
DECLARE_OUTPUT(702)
DECLARE_OUTPUT(703)
DECLARE_OUTPUT(704)
DECLARE_OUTPUT(705)
DECLARE_OUTPUT(706)
DECLARE_OUTPUT(707)
DECLARE_OUTPUT(708)
DECLARE_OUTPUT(709)
DECLARE_OUTPUT(710)
DECLARE_OUTPUT(711)
DECLARE_OUTPUT(712)
DECLARE_OUTPUT(713)
DECLARE_OUTPUT(714)
DECLARE_OUTPUT(715)
DECLARE_OUTPUT(716)
DECLARE_OUTPUT(717)
DECLARE_OUTPUT(718)
DECLARE_OUTPUT(719)
DECLARE_OUTPUT(720)
DECLARE_OUTPUT(721)
DECLARE_OUTPUT(722)
DECLARE_OUTPUT(723)
DECLARE_OUTPUT(724)
DECLARE_OUTPUT(725)
DECLARE_OUTPUT(726)
DECLARE_OUTPUT(727)
DECLARE_OUTPUT(728)
DECLARE_OUTPUT(729)
DECLARE_OUTPUT(730)
DECLARE_OUTPUT(731)
DECLARE_OUTPUT(732)
DECLARE_OUTPUT(733)
DECLARE_OUTPUT(734)
DECLARE_OUTPUT(735)
DECLARE_OUTPUT(736)
DECLARE_OUTPUT(737)
DECLARE_OUTPUT(738)
DECLARE_OUTPUT(739)
DECLARE_OUTPUT(740)
DECLARE_OUTPUT(741)
DECLARE_OUTPUT(742)
DECLARE_OUTPUT(743)
DECLARE_OUTPUT(744)
DECLARE_OUTPUT(745)
DECLARE_OUTPUT(746)
DECLARE_OUTPUT(747)
DECLARE_OUTPUT(748)
DECLARE_OUTPUT(749)
DECLARE_OUTPUT(750)
DECLARE_OUTPUT(751)
DECLARE_OUTPUT(752)
DECLARE_OUTPUT(753)
DECLARE_OUTPUT(754)
DECLARE_OUTPUT(755)
DECLARE_OUTPUT(756)
DECLARE_OUTPUT(757)
DECLARE_OUTPUT(758)
DECLARE_OUTPUT(759)
DECLARE_OUTPUT(760)
DECLARE_OUTPUT(761)
DECLARE_OUTPUT(762)
DECLARE_OUTPUT(763)
DECLARE_OUTPUT(764)
DECLARE_OUTPUT(765)
DECLARE_OUTPUT(766)
DECLARE_OUTPUT(767)
DECLARE_OUTPUT(768)
DECLARE_OUTPUT(769)
DECLARE_OUTPUT(770)
DECLARE_OUTPUT(771)
DECLARE_OUTPUT(772)
DECLARE_OUTPUT(773)
DECLARE_OUTPUT(774)
DECLARE_OUTPUT(775)
DECLARE_OUTPUT(776)
DECLARE_OUTPUT(777)
DECLARE_OUTPUT(778)
DECLARE_OUTPUT(779)
DECLARE_OUTPUT(780)
DECLARE_OUTPUT(781)
DECLARE_OUTPUT(782)
DECLARE_OUTPUT(783)
DECLARE_OUTPUT(784)
DECLARE_OUTPUT(785)
DECLARE_OUTPUT(786)
DECLARE_OUTPUT(787)
DECLARE_OUTPUT(788)
DECLARE_OUTPUT(789)
DECLARE_OUTPUT(790)
DECLARE_OUTPUT(791)
DECLARE_OUTPUT(792)
DECLARE_OUTPUT(793)
DECLARE_OUTPUT(794)
DECLARE_OUTPUT(795)
DECLARE_OUTPUT(796)
DECLARE_OUTPUT(797)
DECLARE_OUTPUT(798)
DECLARE_OUTPUT(799)
DECLARE_OUTPUT(800)
DECLARE_OUTPUT(801)
DECLARE_OUTPUT(802)
DECLARE_OUTPUT(803)
DECLARE_OUTPUT(804)
DECLARE_OUTPUT(805)
DECLARE_OUTPUT(806)
DECLARE_OUTPUT(807)
DECLARE_OUTPUT(808)
DECLARE_OUTPUT(809)
DECLARE_OUTPUT(810)
DECLARE_OUTPUT(811)
DECLARE_OUTPUT(812)
DECLARE_OUTPUT(813)
DECLARE_OUTPUT(814)
DECLARE_OUTPUT(815)
DECLARE_OUTPUT(816)
DECLARE_OUTPUT(817)
DECLARE_OUTPUT(818)
DECLARE_OUTPUT(819)
DECLARE_OUTPUT(820)
DECLARE_OUTPUT(821)
DECLARE_OUTPUT(822)
DECLARE_OUTPUT(823)
DECLARE_OUTPUT(824)
DECLARE_OUTPUT(825)
DECLARE_OUTPUT(826)
DECLARE_OUTPUT(827)
DECLARE_OUTPUT(828)
DECLARE_OUTPUT(829)
DECLARE_OUTPUT(830)
DECLARE_OUTPUT(831)
DECLARE_OUTPUT(832)
DECLARE_OUTPUT(833)
DECLARE_OUTPUT(834)
DECLARE_OUTPUT(835)
DECLARE_OUTPUT(836)
DECLARE_OUTPUT(837)
DECLARE_OUTPUT(838)
DECLARE_OUTPUT(839)
DECLARE_OUTPUT(840)
DECLARE_OUTPUT(841)
DECLARE_OUTPUT(842)
DECLARE_OUTPUT(843)
DECLARE_OUTPUT(844)
DECLARE_OUTPUT(845)
DECLARE_OUTPUT(846)
DECLARE_OUTPUT(847)
DECLARE_OUTPUT(848)
DECLARE_OUTPUT(849)
DECLARE_OUTPUT(850)
DECLARE_OUTPUT(851)
DECLARE_OUTPUT(852)
DECLARE_OUTPUT(853)
DECLARE_OUTPUT(854)
DECLARE_OUTPUT(855)
DECLARE_OUTPUT(856)
DECLARE_OUTPUT(857)
DECLARE_OUTPUT(858)
DECLARE_OUTPUT(859)
DECLARE_OUTPUT(860)
DECLARE_OUTPUT(861)
DECLARE_OUTPUT(862)
DECLARE_OUTPUT(863)
DECLARE_OUTPUT(864)
DECLARE_OUTPUT(865)
DECLARE_OUTPUT(866)
DECLARE_OUTPUT(867)
DECLARE_OUTPUT(868)
DECLARE_OUTPUT(869)
DECLARE_OUTPUT(870)
DECLARE_OUTPUT(871)
DECLARE_OUTPUT(872)
DECLARE_OUTPUT(873)
DECLARE_OUTPUT(874)
DECLARE_OUTPUT(875)
DECLARE_OUTPUT(876)
DECLARE_OUTPUT(877)
DECLARE_OUTPUT(878)
DECLARE_OUTPUT(879)
DECLARE_OUTPUT(880)
DECLARE_OUTPUT(881)
DECLARE_OUTPUT(882)
DECLARE_OUTPUT(883)
DECLARE_OUTPUT(884)
DECLARE_OUTPUT(885)
DECLARE_OUTPUT(886)
DECLARE_OUTPUT(887)
DECLARE_OUTPUT(888)
DECLARE_OUTPUT(889)
DECLARE_OUTPUT(890)
DECLARE_OUTPUT(891)
DECLARE_OUTPUT(892)
DECLARE_OUTPUT(893)
DECLARE_OUTPUT(894)
DECLARE_OUTPUT(895)
DECLARE_OUTPUT(896)
DECLARE_OUTPUT(897)
DECLARE_OUTPUT(898)
DECLARE_OUTPUT(899)
DECLARE_OUTPUT(900)
DECLARE_OUTPUT(901)
DECLARE_OUTPUT(902)
DECLARE_OUTPUT(903)
DECLARE_OUTPUT(904)
DECLARE_OUTPUT(905)
DECLARE_OUTPUT(906)
DECLARE_OUTPUT(907)
DECLARE_OUTPUT(908)
DECLARE_OUTPUT(909)
DECLARE_OUTPUT(910)
DECLARE_OUTPUT(911)
DECLARE_OUTPUT(912)
DECLARE_OUTPUT(913)
DECLARE_OUTPUT(914)
DECLARE_OUTPUT(915)
DECLARE_OUTPUT(916)
DECLARE_OUTPUT(917)
DECLARE_OUTPUT(918)
DECLARE_OUTPUT(919)
DECLARE_OUTPUT(920)
DECLARE_OUTPUT(921)
DECLARE_OUTPUT(922)
DECLARE_OUTPUT(923)
DECLARE_OUTPUT(924)
DECLARE_OUTPUT(925)
DECLARE_OUTPUT(926)
DECLARE_OUTPUT(927)
DECLARE_OUTPUT(928)
DECLARE_OUTPUT(929)
DECLARE_OUTPUT(930)
DECLARE_OUTPUT(931)
DECLARE_OUTPUT(932)
DECLARE_OUTPUT(933)
DECLARE_OUTPUT(934)
DECLARE_OUTPUT(935)
DECLARE_OUTPUT(936)
DECLARE_OUTPUT(937)
DECLARE_OUTPUT(938)
DECLARE_OUTPUT(939)
DECLARE_OUTPUT(940)
DECLARE_OUTPUT(941)
DECLARE_OUTPUT(942)
DECLARE_OUTPUT(943)
DECLARE_OUTPUT(944)
DECLARE_OUTPUT(945)
DECLARE_OUTPUT(946)
DECLARE_OUTPUT(947)
DECLARE_OUTPUT(948)
DECLARE_OUTPUT(949)
DECLARE_OUTPUT(950)
DECLARE_OUTPUT(951)
DECLARE_OUTPUT(952)
DECLARE_OUTPUT(953)
DECLARE_OUTPUT(954)
DECLARE_OUTPUT(955)
DECLARE_OUTPUT(956)
DECLARE_OUTPUT(957)
DECLARE_OUTPUT(958)
DECLARE_OUTPUT(959)
DECLARE_OUTPUT(960)
DECLARE_OUTPUT(961)
DECLARE_OUTPUT(962)
DECLARE_OUTPUT(963)
DECLARE_OUTPUT(964)
DECLARE_OUTPUT(965)
DECLARE_OUTPUT(966)
DECLARE_OUTPUT(967)
DECLARE_OUTPUT(968)
DECLARE_OUTPUT(969)
DECLARE_OUTPUT(970)
DECLARE_OUTPUT(971)
DECLARE_OUTPUT(972)
DECLARE_OUTPUT(973)
DECLARE_OUTPUT(974)
DECLARE_OUTPUT(975)
DECLARE_OUTPUT(976)
DECLARE_OUTPUT(977)
DECLARE_OUTPUT(978)
DECLARE_OUTPUT(979)
DECLARE_OUTPUT(980)
DECLARE_OUTPUT(981)
DECLARE_OUTPUT(982)
DECLARE_OUTPUT(983)
DECLARE_OUTPUT(984)
DECLARE_OUTPUT(985)
DECLARE_OUTPUT(986)
DECLARE_OUTPUT(987)
DECLARE_OUTPUT(988)
DECLARE_OUTPUT(989)
DECLARE_OUTPUT(990)
DECLARE_OUTPUT(991)
DECLARE_OUTPUT(992)
DECLARE_OUTPUT(993)
DECLARE_OUTPUT(994)
DECLARE_OUTPUT(995)
DECLARE_OUTPUT(996)
DECLARE_OUTPUT(997)
DECLARE_OUTPUT(998)
DECLARE_OUTPUT(999)
DECLARE_OUTPUT(1000)
*/



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
    
    char* global_mem_ptr = new char[10*1024*1024*1024UL];
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
        //perform_huffman_encoding (((VectorVertexEmbedding<3>*) embeddings), n_embeddings);
        break;
      }
      
      case 4: {
          embedding_size = sizeof (VectorVertexEmbedding<4>);
          new_embedding_size = sizeof (VectorVertexEmbedding<5>);
          for (int i = 0; i < n_embeddings; i++) {
            ((VectorVertexEmbedding<4>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<4>*)embeddings)[i];
        }
        //perform_huffman_encoding (((VectorVertexEmbedding<4>*) embeddings), n_embeddings);
        break;
      }
      case 5: {
        embedding_size = sizeof (VectorVertexEmbedding<5>);
        new_embedding_size = sizeof (VectorVertexEmbedding<6>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<5>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<5>*)embeddings)[i];
        }
        //perform_huffman_encoding (((VectorVertexEmbedding<5>*) embeddings), n_embeddings);
        break;
      }
      case 6: {
        embedding_size = sizeof (VectorVertexEmbedding<6>);
        new_embedding_size = sizeof (VectorVertexEmbedding<7>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<6>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<6>*)embeddings)[i];
        }
        //perform_huffman_encoding (((VectorVertexEmbedding<6>*) embeddings), n_embeddings);
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
      case 9: {
        embedding_size = sizeof (VectorVertexEmbedding<9>);
        new_embedding_size = sizeof (VectorVertexEmbedding<10>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<9>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<9>*)embeddings)[i];
        }
        break;
      }
      case 10: {
        embedding_size = sizeof (VectorVertexEmbedding<10>);
        new_embedding_size = sizeof (VectorVertexEmbedding<11>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<10>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<10>*)embeddings)[i];
        }
        break;
      }
      case 11: {
        embedding_size = sizeof (VectorVertexEmbedding<11>);
        new_embedding_size = sizeof (VectorVertexEmbedding<12>);
        for (int i = 0; i < n_embeddings; i++) {
          ((VectorVertexEmbedding<10>*)global_mem_ptr)[i] = ((VectorVertexEmbedding<10>*)embeddings)[i];
        }
        break;
      }
      HOST_TO_DEVICE_CASE(12)
      HOST_TO_DEVICE_CASE(13)
      HOST_TO_DEVICE_CASE(14)
      HOST_TO_DEVICE_CASE(15)
      HOST_TO_DEVICE_CASE(16)
      HOST_TO_DEVICE_CASE(17)
      HOST_TO_DEVICE_CASE(18)
      HOST_TO_DEVICE_CASE(19)
      HOST_TO_DEVICE_CASE(20)
      HOST_TO_DEVICE_CASE(21)
      HOST_TO_DEVICE_CASE(22)
      HOST_TO_DEVICE_CASE(23)
      HOST_TO_DEVICE_CASE(24)
      HOST_TO_DEVICE_CASE(25)
      HOST_TO_DEVICE_CASE(26)
      HOST_TO_DEVICE_CASE(27)
      HOST_TO_DEVICE_CASE(28)
      HOST_TO_DEVICE_CASE(29)
      HOST_TO_DEVICE_CASE(30)
      HOST_TO_DEVICE_CASE(31)
HOST_TO_DEVICE_CASE(32)
HOST_TO_DEVICE_CASE(33)
HOST_TO_DEVICE_CASE(34)
HOST_TO_DEVICE_CASE(35)
HOST_TO_DEVICE_CASE(36)
HOST_TO_DEVICE_CASE(37)
HOST_TO_DEVICE_CASE(38)
HOST_TO_DEVICE_CASE(39)
HOST_TO_DEVICE_CASE(40)
HOST_TO_DEVICE_CASE(41)
HOST_TO_DEVICE_CASE(42)
HOST_TO_DEVICE_CASE(43)
HOST_TO_DEVICE_CASE(44)
HOST_TO_DEVICE_CASE(45)
HOST_TO_DEVICE_CASE(46)
HOST_TO_DEVICE_CASE(47)
HOST_TO_DEVICE_CASE(48)
HOST_TO_DEVICE_CASE(49)
HOST_TO_DEVICE_CASE(50)
HOST_TO_DEVICE_CASE(51)
HOST_TO_DEVICE_CASE(52)
HOST_TO_DEVICE_CASE(53)
HOST_TO_DEVICE_CASE(54)
HOST_TO_DEVICE_CASE(55)
HOST_TO_DEVICE_CASE(56)
HOST_TO_DEVICE_CASE(57)
HOST_TO_DEVICE_CASE(58)
HOST_TO_DEVICE_CASE(59)
HOST_TO_DEVICE_CASE(60)
HOST_TO_DEVICE_CASE(61)
HOST_TO_DEVICE_CASE(62)
HOST_TO_DEVICE_CASE(63)
HOST_TO_DEVICE_CASE(64)
HOST_TO_DEVICE_CASE(65)
HOST_TO_DEVICE_CASE(66)
HOST_TO_DEVICE_CASE(67)
HOST_TO_DEVICE_CASE(68)
HOST_TO_DEVICE_CASE(69)
HOST_TO_DEVICE_CASE(70)
HOST_TO_DEVICE_CASE(71)
HOST_TO_DEVICE_CASE(72)
HOST_TO_DEVICE_CASE(73)
HOST_TO_DEVICE_CASE(74)
HOST_TO_DEVICE_CASE(75)
HOST_TO_DEVICE_CASE(76)
HOST_TO_DEVICE_CASE(77)
HOST_TO_DEVICE_CASE(78)
HOST_TO_DEVICE_CASE(79)
HOST_TO_DEVICE_CASE(80)
HOST_TO_DEVICE_CASE(81)
HOST_TO_DEVICE_CASE(82)
HOST_TO_DEVICE_CASE(83)
HOST_TO_DEVICE_CASE(84)
HOST_TO_DEVICE_CASE(85)
HOST_TO_DEVICE_CASE(86)
HOST_TO_DEVICE_CASE(87)
HOST_TO_DEVICE_CASE(88)
HOST_TO_DEVICE_CASE(89)
HOST_TO_DEVICE_CASE(90)
HOST_TO_DEVICE_CASE(91)
HOST_TO_DEVICE_CASE(92)
HOST_TO_DEVICE_CASE(93)
HOST_TO_DEVICE_CASE(94)
HOST_TO_DEVICE_CASE(95)
HOST_TO_DEVICE_CASE(96)
HOST_TO_DEVICE_CASE(97)
HOST_TO_DEVICE_CASE(98)
HOST_TO_DEVICE_CASE(99)
/*
HOST_TO_DEVICE_CASE(100)
HOST_TO_DEVICE_CASE(101)
HOST_TO_DEVICE_CASE(102)
HOST_TO_DEVICE_CASE(103)
HOST_TO_DEVICE_CASE(104)
HOST_TO_DEVICE_CASE(105)
HOST_TO_DEVICE_CASE(106)
HOST_TO_DEVICE_CASE(107)
HOST_TO_DEVICE_CASE(108)
HOST_TO_DEVICE_CASE(109)
HOST_TO_DEVICE_CASE(110)
HOST_TO_DEVICE_CASE(111)
HOST_TO_DEVICE_CASE(112)
HOST_TO_DEVICE_CASE(113)
HOST_TO_DEVICE_CASE(114)
HOST_TO_DEVICE_CASE(115)
HOST_TO_DEVICE_CASE(116)
HOST_TO_DEVICE_CASE(117)
HOST_TO_DEVICE_CASE(118)
HOST_TO_DEVICE_CASE(119)
HOST_TO_DEVICE_CASE(120)
HOST_TO_DEVICE_CASE(121)
HOST_TO_DEVICE_CASE(122)
HOST_TO_DEVICE_CASE(123)
HOST_TO_DEVICE_CASE(124)
HOST_TO_DEVICE_CASE(125)
HOST_TO_DEVICE_CASE(126)
HOST_TO_DEVICE_CASE(127)
HOST_TO_DEVICE_CASE(128)
HOST_TO_DEVICE_CASE(129)
HOST_TO_DEVICE_CASE(130)
HOST_TO_DEVICE_CASE(131)
HOST_TO_DEVICE_CASE(132)
HOST_TO_DEVICE_CASE(133)
HOST_TO_DEVICE_CASE(134)
HOST_TO_DEVICE_CASE(135)
HOST_TO_DEVICE_CASE(136)
HOST_TO_DEVICE_CASE(137)
HOST_TO_DEVICE_CASE(138)
HOST_TO_DEVICE_CASE(139)
HOST_TO_DEVICE_CASE(140)
HOST_TO_DEVICE_CASE(141)
HOST_TO_DEVICE_CASE(142)
HOST_TO_DEVICE_CASE(143)
HOST_TO_DEVICE_CASE(144)
HOST_TO_DEVICE_CASE(145)
HOST_TO_DEVICE_CASE(146)
HOST_TO_DEVICE_CASE(147)
HOST_TO_DEVICE_CASE(148)
HOST_TO_DEVICE_CASE(149)
HOST_TO_DEVICE_CASE(150)
HOST_TO_DEVICE_CASE(151)
HOST_TO_DEVICE_CASE(152)
HOST_TO_DEVICE_CASE(153)
HOST_TO_DEVICE_CASE(154)
HOST_TO_DEVICE_CASE(155)
HOST_TO_DEVICE_CASE(156)
HOST_TO_DEVICE_CASE(157)
HOST_TO_DEVICE_CASE(158)
HOST_TO_DEVICE_CASE(159)
HOST_TO_DEVICE_CASE(160)
HOST_TO_DEVICE_CASE(161)
HOST_TO_DEVICE_CASE(162)
HOST_TO_DEVICE_CASE(163)
HOST_TO_DEVICE_CASE(164)
HOST_TO_DEVICE_CASE(165)
HOST_TO_DEVICE_CASE(166)
HOST_TO_DEVICE_CASE(167)
HOST_TO_DEVICE_CASE(168)
HOST_TO_DEVICE_CASE(169)
HOST_TO_DEVICE_CASE(170)
HOST_TO_DEVICE_CASE(171)
HOST_TO_DEVICE_CASE(172)
HOST_TO_DEVICE_CASE(173)
HOST_TO_DEVICE_CASE(174)
HOST_TO_DEVICE_CASE(175)
HOST_TO_DEVICE_CASE(176)
HOST_TO_DEVICE_CASE(177)
HOST_TO_DEVICE_CASE(178)
HOST_TO_DEVICE_CASE(179)
HOST_TO_DEVICE_CASE(180)
HOST_TO_DEVICE_CASE(181)
HOST_TO_DEVICE_CASE(182)
HOST_TO_DEVICE_CASE(183)
HOST_TO_DEVICE_CASE(184)
HOST_TO_DEVICE_CASE(185)
HOST_TO_DEVICE_CASE(186)
HOST_TO_DEVICE_CASE(187)
HOST_TO_DEVICE_CASE(188)
HOST_TO_DEVICE_CASE(189)
HOST_TO_DEVICE_CASE(190)
HOST_TO_DEVICE_CASE(191)
HOST_TO_DEVICE_CASE(192)
HOST_TO_DEVICE_CASE(193)
HOST_TO_DEVICE_CASE(194)
HOST_TO_DEVICE_CASE(195)
HOST_TO_DEVICE_CASE(196)
HOST_TO_DEVICE_CASE(197)
HOST_TO_DEVICE_CASE(198)
HOST_TO_DEVICE_CASE(199)
HOST_TO_DEVICE_CASE(200)
HOST_TO_DEVICE_CASE(201)
HOST_TO_DEVICE_CASE(202)
HOST_TO_DEVICE_CASE(203)
HOST_TO_DEVICE_CASE(204)
HOST_TO_DEVICE_CASE(205)
HOST_TO_DEVICE_CASE(206)
HOST_TO_DEVICE_CASE(207)
HOST_TO_DEVICE_CASE(208)
HOST_TO_DEVICE_CASE(209)
HOST_TO_DEVICE_CASE(210)
HOST_TO_DEVICE_CASE(211)
HOST_TO_DEVICE_CASE(212)
HOST_TO_DEVICE_CASE(213)
HOST_TO_DEVICE_CASE(214)
HOST_TO_DEVICE_CASE(215)
HOST_TO_DEVICE_CASE(216)
HOST_TO_DEVICE_CASE(217)
HOST_TO_DEVICE_CASE(218)
HOST_TO_DEVICE_CASE(219)
HOST_TO_DEVICE_CASE(220)
HOST_TO_DEVICE_CASE(221)
HOST_TO_DEVICE_CASE(222)
HOST_TO_DEVICE_CASE(223)
HOST_TO_DEVICE_CASE(224)
HOST_TO_DEVICE_CASE(225)
HOST_TO_DEVICE_CASE(226)
HOST_TO_DEVICE_CASE(227)
HOST_TO_DEVICE_CASE(228)
HOST_TO_DEVICE_CASE(229)
HOST_TO_DEVICE_CASE(230)
HOST_TO_DEVICE_CASE(231)
HOST_TO_DEVICE_CASE(232)
HOST_TO_DEVICE_CASE(233)
HOST_TO_DEVICE_CASE(234)
HOST_TO_DEVICE_CASE(235)
HOST_TO_DEVICE_CASE(236)
HOST_TO_DEVICE_CASE(237)
HOST_TO_DEVICE_CASE(238)
HOST_TO_DEVICE_CASE(239)
HOST_TO_DEVICE_CASE(240)
HOST_TO_DEVICE_CASE(241)
HOST_TO_DEVICE_CASE(242)
HOST_TO_DEVICE_CASE(243)
HOST_TO_DEVICE_CASE(244)
HOST_TO_DEVICE_CASE(245)
HOST_TO_DEVICE_CASE(246)
HOST_TO_DEVICE_CASE(247)
HOST_TO_DEVICE_CASE(248)
HOST_TO_DEVICE_CASE(249)
HOST_TO_DEVICE_CASE(250)
HOST_TO_DEVICE_CASE(251)
HOST_TO_DEVICE_CASE(252)
HOST_TO_DEVICE_CASE(253)
HOST_TO_DEVICE_CASE(254)
HOST_TO_DEVICE_CASE(255)
HOST_TO_DEVICE_CASE(256)
HOST_TO_DEVICE_CASE(257)
HOST_TO_DEVICE_CASE(258)
HOST_TO_DEVICE_CASE(259)
HOST_TO_DEVICE_CASE(260)
HOST_TO_DEVICE_CASE(261)
HOST_TO_DEVICE_CASE(262)
HOST_TO_DEVICE_CASE(263)
HOST_TO_DEVICE_CASE(264)
HOST_TO_DEVICE_CASE(265)
HOST_TO_DEVICE_CASE(266)
HOST_TO_DEVICE_CASE(267)
HOST_TO_DEVICE_CASE(268)
HOST_TO_DEVICE_CASE(269)
HOST_TO_DEVICE_CASE(270)
HOST_TO_DEVICE_CASE(271)
HOST_TO_DEVICE_CASE(272)
HOST_TO_DEVICE_CASE(273)
HOST_TO_DEVICE_CASE(274)
HOST_TO_DEVICE_CASE(275)
HOST_TO_DEVICE_CASE(276)
HOST_TO_DEVICE_CASE(277)
HOST_TO_DEVICE_CASE(278)
HOST_TO_DEVICE_CASE(279)
HOST_TO_DEVICE_CASE(280)
HOST_TO_DEVICE_CASE(281)
HOST_TO_DEVICE_CASE(282)
HOST_TO_DEVICE_CASE(283)
HOST_TO_DEVICE_CASE(284)
HOST_TO_DEVICE_CASE(285)
HOST_TO_DEVICE_CASE(286)
HOST_TO_DEVICE_CASE(287)
HOST_TO_DEVICE_CASE(288)
HOST_TO_DEVICE_CASE(289)
HOST_TO_DEVICE_CASE(290)
HOST_TO_DEVICE_CASE(291)
HOST_TO_DEVICE_CASE(292)
HOST_TO_DEVICE_CASE(293)
HOST_TO_DEVICE_CASE(294)
HOST_TO_DEVICE_CASE(295)
HOST_TO_DEVICE_CASE(296)
HOST_TO_DEVICE_CASE(297)
HOST_TO_DEVICE_CASE(298)
HOST_TO_DEVICE_CASE(299)
HOST_TO_DEVICE_CASE(300)
HOST_TO_DEVICE_CASE(301)
HOST_TO_DEVICE_CASE(302)
HOST_TO_DEVICE_CASE(303)
HOST_TO_DEVICE_CASE(304)
HOST_TO_DEVICE_CASE(305)
HOST_TO_DEVICE_CASE(306)
HOST_TO_DEVICE_CASE(307)
HOST_TO_DEVICE_CASE(308)
HOST_TO_DEVICE_CASE(309)
HOST_TO_DEVICE_CASE(310)
HOST_TO_DEVICE_CASE(311)
HOST_TO_DEVICE_CASE(312)
HOST_TO_DEVICE_CASE(313)
HOST_TO_DEVICE_CASE(314)
HOST_TO_DEVICE_CASE(315)
HOST_TO_DEVICE_CASE(316)
HOST_TO_DEVICE_CASE(317)
HOST_TO_DEVICE_CASE(318)
HOST_TO_DEVICE_CASE(319)
HOST_TO_DEVICE_CASE(320)
HOST_TO_DEVICE_CASE(321)
HOST_TO_DEVICE_CASE(322)
HOST_TO_DEVICE_CASE(323)
HOST_TO_DEVICE_CASE(324)
HOST_TO_DEVICE_CASE(325)
HOST_TO_DEVICE_CASE(326)
HOST_TO_DEVICE_CASE(327)
HOST_TO_DEVICE_CASE(328)
HOST_TO_DEVICE_CASE(329)
HOST_TO_DEVICE_CASE(330)
HOST_TO_DEVICE_CASE(331)
HOST_TO_DEVICE_CASE(332)
HOST_TO_DEVICE_CASE(333)
HOST_TO_DEVICE_CASE(334)
HOST_TO_DEVICE_CASE(335)
HOST_TO_DEVICE_CASE(336)
HOST_TO_DEVICE_CASE(337)
HOST_TO_DEVICE_CASE(338)
HOST_TO_DEVICE_CASE(339)
HOST_TO_DEVICE_CASE(340)
HOST_TO_DEVICE_CASE(341)
HOST_TO_DEVICE_CASE(342)
HOST_TO_DEVICE_CASE(343)
HOST_TO_DEVICE_CASE(344)
HOST_TO_DEVICE_CASE(345)
HOST_TO_DEVICE_CASE(346)
HOST_TO_DEVICE_CASE(347)
HOST_TO_DEVICE_CASE(348)
HOST_TO_DEVICE_CASE(349)
HOST_TO_DEVICE_CASE(350)
HOST_TO_DEVICE_CASE(351)
HOST_TO_DEVICE_CASE(352)
HOST_TO_DEVICE_CASE(353)
HOST_TO_DEVICE_CASE(354)
HOST_TO_DEVICE_CASE(355)
HOST_TO_DEVICE_CASE(356)
HOST_TO_DEVICE_CASE(357)
HOST_TO_DEVICE_CASE(358)
HOST_TO_DEVICE_CASE(359)
HOST_TO_DEVICE_CASE(360)
HOST_TO_DEVICE_CASE(361)
HOST_TO_DEVICE_CASE(362)
HOST_TO_DEVICE_CASE(363)
HOST_TO_DEVICE_CASE(364)
HOST_TO_DEVICE_CASE(365)
HOST_TO_DEVICE_CASE(366)
HOST_TO_DEVICE_CASE(367)
HOST_TO_DEVICE_CASE(368)
HOST_TO_DEVICE_CASE(369)
HOST_TO_DEVICE_CASE(370)
HOST_TO_DEVICE_CASE(371)
HOST_TO_DEVICE_CASE(372)
HOST_TO_DEVICE_CASE(373)
HOST_TO_DEVICE_CASE(374)
HOST_TO_DEVICE_CASE(375)
HOST_TO_DEVICE_CASE(376)
HOST_TO_DEVICE_CASE(377)
HOST_TO_DEVICE_CASE(378)
HOST_TO_DEVICE_CASE(379)
HOST_TO_DEVICE_CASE(380)
HOST_TO_DEVICE_CASE(381)
HOST_TO_DEVICE_CASE(382)
HOST_TO_DEVICE_CASE(383)
HOST_TO_DEVICE_CASE(384)
HOST_TO_DEVICE_CASE(385)
HOST_TO_DEVICE_CASE(386)
HOST_TO_DEVICE_CASE(387)
HOST_TO_DEVICE_CASE(388)
HOST_TO_DEVICE_CASE(389)
HOST_TO_DEVICE_CASE(390)
HOST_TO_DEVICE_CASE(391)
HOST_TO_DEVICE_CASE(392)
HOST_TO_DEVICE_CASE(393)
HOST_TO_DEVICE_CASE(394)
HOST_TO_DEVICE_CASE(395)
HOST_TO_DEVICE_CASE(396)
HOST_TO_DEVICE_CASE(397)
HOST_TO_DEVICE_CASE(398)
HOST_TO_DEVICE_CASE(399)
HOST_TO_DEVICE_CASE(400)
HOST_TO_DEVICE_CASE(401)
HOST_TO_DEVICE_CASE(402)
HOST_TO_DEVICE_CASE(403)
HOST_TO_DEVICE_CASE(404)
HOST_TO_DEVICE_CASE(405)
HOST_TO_DEVICE_CASE(406)
HOST_TO_DEVICE_CASE(407)
HOST_TO_DEVICE_CASE(408)
HOST_TO_DEVICE_CASE(409)
HOST_TO_DEVICE_CASE(410)
HOST_TO_DEVICE_CASE(411)
HOST_TO_DEVICE_CASE(412)
HOST_TO_DEVICE_CASE(413)
HOST_TO_DEVICE_CASE(414)
HOST_TO_DEVICE_CASE(415)
HOST_TO_DEVICE_CASE(416)
HOST_TO_DEVICE_CASE(417)
HOST_TO_DEVICE_CASE(418)
HOST_TO_DEVICE_CASE(419)
HOST_TO_DEVICE_CASE(420)
HOST_TO_DEVICE_CASE(421)
HOST_TO_DEVICE_CASE(422)
HOST_TO_DEVICE_CASE(423)
HOST_TO_DEVICE_CASE(424)
HOST_TO_DEVICE_CASE(425)
HOST_TO_DEVICE_CASE(426)
HOST_TO_DEVICE_CASE(427)
HOST_TO_DEVICE_CASE(428)
HOST_TO_DEVICE_CASE(429)
HOST_TO_DEVICE_CASE(430)
HOST_TO_DEVICE_CASE(431)
HOST_TO_DEVICE_CASE(432)
HOST_TO_DEVICE_CASE(433)
HOST_TO_DEVICE_CASE(434)
HOST_TO_DEVICE_CASE(435)
HOST_TO_DEVICE_CASE(436)
HOST_TO_DEVICE_CASE(437)
HOST_TO_DEVICE_CASE(438)
HOST_TO_DEVICE_CASE(439)
HOST_TO_DEVICE_CASE(440)
HOST_TO_DEVICE_CASE(441)
HOST_TO_DEVICE_CASE(442)
HOST_TO_DEVICE_CASE(443)
HOST_TO_DEVICE_CASE(444)
HOST_TO_DEVICE_CASE(445)
HOST_TO_DEVICE_CASE(446)
HOST_TO_DEVICE_CASE(447)
HOST_TO_DEVICE_CASE(448)
HOST_TO_DEVICE_CASE(449)
HOST_TO_DEVICE_CASE(450)
HOST_TO_DEVICE_CASE(451)
HOST_TO_DEVICE_CASE(452)
HOST_TO_DEVICE_CASE(453)
HOST_TO_DEVICE_CASE(454)
HOST_TO_DEVICE_CASE(455)
HOST_TO_DEVICE_CASE(456)
HOST_TO_DEVICE_CASE(457)
HOST_TO_DEVICE_CASE(458)
HOST_TO_DEVICE_CASE(459)
HOST_TO_DEVICE_CASE(460)
HOST_TO_DEVICE_CASE(461)
HOST_TO_DEVICE_CASE(462)
HOST_TO_DEVICE_CASE(463)
HOST_TO_DEVICE_CASE(464)
HOST_TO_DEVICE_CASE(465)
HOST_TO_DEVICE_CASE(466)
HOST_TO_DEVICE_CASE(467)
HOST_TO_DEVICE_CASE(468)
HOST_TO_DEVICE_CASE(469)
HOST_TO_DEVICE_CASE(470)
HOST_TO_DEVICE_CASE(471)
HOST_TO_DEVICE_CASE(472)
HOST_TO_DEVICE_CASE(473)
HOST_TO_DEVICE_CASE(474)
HOST_TO_DEVICE_CASE(475)
HOST_TO_DEVICE_CASE(476)
HOST_TO_DEVICE_CASE(477)
HOST_TO_DEVICE_CASE(478)
HOST_TO_DEVICE_CASE(479)
HOST_TO_DEVICE_CASE(480)
HOST_TO_DEVICE_CASE(481)
HOST_TO_DEVICE_CASE(482)
HOST_TO_DEVICE_CASE(483)
HOST_TO_DEVICE_CASE(484)
HOST_TO_DEVICE_CASE(485)
HOST_TO_DEVICE_CASE(486)
HOST_TO_DEVICE_CASE(487)
HOST_TO_DEVICE_CASE(488)
HOST_TO_DEVICE_CASE(489)
HOST_TO_DEVICE_CASE(490)
HOST_TO_DEVICE_CASE(491)
HOST_TO_DEVICE_CASE(492)
HOST_TO_DEVICE_CASE(493)
HOST_TO_DEVICE_CASE(494)
HOST_TO_DEVICE_CASE(495)
HOST_TO_DEVICE_CASE(496)
HOST_TO_DEVICE_CASE(497)
HOST_TO_DEVICE_CASE(498)
HOST_TO_DEVICE_CASE(499)
HOST_TO_DEVICE_CASE(500)
HOST_TO_DEVICE_CASE(501)
HOST_TO_DEVICE_CASE(502)
HOST_TO_DEVICE_CASE(503)
HOST_TO_DEVICE_CASE(504)
HOST_TO_DEVICE_CASE(505)
HOST_TO_DEVICE_CASE(506)
HOST_TO_DEVICE_CASE(507)
HOST_TO_DEVICE_CASE(508)
HOST_TO_DEVICE_CASE(509)
HOST_TO_DEVICE_CASE(510)
HOST_TO_DEVICE_CASE(511)
HOST_TO_DEVICE_CASE(512)
HOST_TO_DEVICE_CASE(513)
HOST_TO_DEVICE_CASE(514)
HOST_TO_DEVICE_CASE(515)
HOST_TO_DEVICE_CASE(516)
HOST_TO_DEVICE_CASE(517)
HOST_TO_DEVICE_CASE(518)
HOST_TO_DEVICE_CASE(519)
HOST_TO_DEVICE_CASE(520)
HOST_TO_DEVICE_CASE(521)
HOST_TO_DEVICE_CASE(522)
HOST_TO_DEVICE_CASE(523)
HOST_TO_DEVICE_CASE(524)
HOST_TO_DEVICE_CASE(525)
HOST_TO_DEVICE_CASE(526)
HOST_TO_DEVICE_CASE(527)
HOST_TO_DEVICE_CASE(528)
HOST_TO_DEVICE_CASE(529)
HOST_TO_DEVICE_CASE(530)
HOST_TO_DEVICE_CASE(531)
HOST_TO_DEVICE_CASE(532)
HOST_TO_DEVICE_CASE(533)
HOST_TO_DEVICE_CASE(534)
HOST_TO_DEVICE_CASE(535)
HOST_TO_DEVICE_CASE(536)
HOST_TO_DEVICE_CASE(537)
HOST_TO_DEVICE_CASE(538)
HOST_TO_DEVICE_CASE(539)
HOST_TO_DEVICE_CASE(540)
HOST_TO_DEVICE_CASE(541)
HOST_TO_DEVICE_CASE(542)
HOST_TO_DEVICE_CASE(543)
HOST_TO_DEVICE_CASE(544)
HOST_TO_DEVICE_CASE(545)
HOST_TO_DEVICE_CASE(546)
HOST_TO_DEVICE_CASE(547)
HOST_TO_DEVICE_CASE(548)
HOST_TO_DEVICE_CASE(549)
HOST_TO_DEVICE_CASE(550)
HOST_TO_DEVICE_CASE(551)
HOST_TO_DEVICE_CASE(552)
HOST_TO_DEVICE_CASE(553)
HOST_TO_DEVICE_CASE(554)
HOST_TO_DEVICE_CASE(555)
HOST_TO_DEVICE_CASE(556)
HOST_TO_DEVICE_CASE(557)
HOST_TO_DEVICE_CASE(558)
HOST_TO_DEVICE_CASE(559)
HOST_TO_DEVICE_CASE(560)
HOST_TO_DEVICE_CASE(561)
HOST_TO_DEVICE_CASE(562)
HOST_TO_DEVICE_CASE(563)
HOST_TO_DEVICE_CASE(564)
HOST_TO_DEVICE_CASE(565)
HOST_TO_DEVICE_CASE(566)
HOST_TO_DEVICE_CASE(567)
HOST_TO_DEVICE_CASE(568)
HOST_TO_DEVICE_CASE(569)
HOST_TO_DEVICE_CASE(570)
HOST_TO_DEVICE_CASE(571)
HOST_TO_DEVICE_CASE(572)
HOST_TO_DEVICE_CASE(573)
HOST_TO_DEVICE_CASE(574)
HOST_TO_DEVICE_CASE(575)
HOST_TO_DEVICE_CASE(576)
HOST_TO_DEVICE_CASE(577)
HOST_TO_DEVICE_CASE(578)
HOST_TO_DEVICE_CASE(579)
HOST_TO_DEVICE_CASE(580)
HOST_TO_DEVICE_CASE(581)
HOST_TO_DEVICE_CASE(582)
HOST_TO_DEVICE_CASE(583)
HOST_TO_DEVICE_CASE(584)
HOST_TO_DEVICE_CASE(585)
HOST_TO_DEVICE_CASE(586)
HOST_TO_DEVICE_CASE(587)
HOST_TO_DEVICE_CASE(588)
HOST_TO_DEVICE_CASE(589)
HOST_TO_DEVICE_CASE(590)
HOST_TO_DEVICE_CASE(591)
HOST_TO_DEVICE_CASE(592)
HOST_TO_DEVICE_CASE(593)
HOST_TO_DEVICE_CASE(594)
HOST_TO_DEVICE_CASE(595)
HOST_TO_DEVICE_CASE(596)
HOST_TO_DEVICE_CASE(597)
HOST_TO_DEVICE_CASE(598)
HOST_TO_DEVICE_CASE(599)
HOST_TO_DEVICE_CASE(600)
HOST_TO_DEVICE_CASE(601)
HOST_TO_DEVICE_CASE(602)
HOST_TO_DEVICE_CASE(603)
HOST_TO_DEVICE_CASE(604)
HOST_TO_DEVICE_CASE(605)
HOST_TO_DEVICE_CASE(606)
HOST_TO_DEVICE_CASE(607)
HOST_TO_DEVICE_CASE(608)
HOST_TO_DEVICE_CASE(609)
HOST_TO_DEVICE_CASE(610)
HOST_TO_DEVICE_CASE(611)
HOST_TO_DEVICE_CASE(612)
HOST_TO_DEVICE_CASE(613)
HOST_TO_DEVICE_CASE(614)
HOST_TO_DEVICE_CASE(615)
HOST_TO_DEVICE_CASE(616)
HOST_TO_DEVICE_CASE(617)
HOST_TO_DEVICE_CASE(618)
HOST_TO_DEVICE_CASE(619)
HOST_TO_DEVICE_CASE(620)
HOST_TO_DEVICE_CASE(621)
HOST_TO_DEVICE_CASE(622)
HOST_TO_DEVICE_CASE(623)
HOST_TO_DEVICE_CASE(624)
HOST_TO_DEVICE_CASE(625)
HOST_TO_DEVICE_CASE(626)
HOST_TO_DEVICE_CASE(627)
HOST_TO_DEVICE_CASE(628)
HOST_TO_DEVICE_CASE(629)
HOST_TO_DEVICE_CASE(630)
HOST_TO_DEVICE_CASE(631)
HOST_TO_DEVICE_CASE(632)
HOST_TO_DEVICE_CASE(633)
HOST_TO_DEVICE_CASE(634)
HOST_TO_DEVICE_CASE(635)
HOST_TO_DEVICE_CASE(636)
HOST_TO_DEVICE_CASE(637)
HOST_TO_DEVICE_CASE(638)
HOST_TO_DEVICE_CASE(639)
HOST_TO_DEVICE_CASE(640)
HOST_TO_DEVICE_CASE(641)
HOST_TO_DEVICE_CASE(642)
HOST_TO_DEVICE_CASE(643)
HOST_TO_DEVICE_CASE(644)
HOST_TO_DEVICE_CASE(645)
HOST_TO_DEVICE_CASE(646)
HOST_TO_DEVICE_CASE(647)
HOST_TO_DEVICE_CASE(648)
HOST_TO_DEVICE_CASE(649)
HOST_TO_DEVICE_CASE(650)
HOST_TO_DEVICE_CASE(651)
HOST_TO_DEVICE_CASE(652)
HOST_TO_DEVICE_CASE(653)
HOST_TO_DEVICE_CASE(654)
HOST_TO_DEVICE_CASE(655)
HOST_TO_DEVICE_CASE(656)
HOST_TO_DEVICE_CASE(657)
HOST_TO_DEVICE_CASE(658)
HOST_TO_DEVICE_CASE(659)
HOST_TO_DEVICE_CASE(660)
HOST_TO_DEVICE_CASE(661)
HOST_TO_DEVICE_CASE(662)
HOST_TO_DEVICE_CASE(663)
HOST_TO_DEVICE_CASE(664)
HOST_TO_DEVICE_CASE(665)
HOST_TO_DEVICE_CASE(666)
HOST_TO_DEVICE_CASE(667)
HOST_TO_DEVICE_CASE(668)
HOST_TO_DEVICE_CASE(669)
HOST_TO_DEVICE_CASE(670)
HOST_TO_DEVICE_CASE(671)
HOST_TO_DEVICE_CASE(672)
HOST_TO_DEVICE_CASE(673)
HOST_TO_DEVICE_CASE(674)
HOST_TO_DEVICE_CASE(675)
HOST_TO_DEVICE_CASE(676)
HOST_TO_DEVICE_CASE(677)
HOST_TO_DEVICE_CASE(678)
HOST_TO_DEVICE_CASE(679)
HOST_TO_DEVICE_CASE(680)
HOST_TO_DEVICE_CASE(681)
HOST_TO_DEVICE_CASE(682)
HOST_TO_DEVICE_CASE(683)
HOST_TO_DEVICE_CASE(684)
HOST_TO_DEVICE_CASE(685)
HOST_TO_DEVICE_CASE(686)
HOST_TO_DEVICE_CASE(687)
HOST_TO_DEVICE_CASE(688)
HOST_TO_DEVICE_CASE(689)
HOST_TO_DEVICE_CASE(690)
HOST_TO_DEVICE_CASE(691)
HOST_TO_DEVICE_CASE(692)
HOST_TO_DEVICE_CASE(693)
HOST_TO_DEVICE_CASE(694)
HOST_TO_DEVICE_CASE(695)
HOST_TO_DEVICE_CASE(696)
HOST_TO_DEVICE_CASE(697)
HOST_TO_DEVICE_CASE(698)
HOST_TO_DEVICE_CASE(699)
HOST_TO_DEVICE_CASE(700)
HOST_TO_DEVICE_CASE(701)
HOST_TO_DEVICE_CASE(702)
HOST_TO_DEVICE_CASE(703)
HOST_TO_DEVICE_CASE(704)
HOST_TO_DEVICE_CASE(705)
HOST_TO_DEVICE_CASE(706)
HOST_TO_DEVICE_CASE(707)
HOST_TO_DEVICE_CASE(708)
HOST_TO_DEVICE_CASE(709)
HOST_TO_DEVICE_CASE(710)
HOST_TO_DEVICE_CASE(711)
HOST_TO_DEVICE_CASE(712)
HOST_TO_DEVICE_CASE(713)
HOST_TO_DEVICE_CASE(714)
HOST_TO_DEVICE_CASE(715)
HOST_TO_DEVICE_CASE(716)
HOST_TO_DEVICE_CASE(717)
HOST_TO_DEVICE_CASE(718)
HOST_TO_DEVICE_CASE(719)
HOST_TO_DEVICE_CASE(720)
HOST_TO_DEVICE_CASE(721)
HOST_TO_DEVICE_CASE(722)
HOST_TO_DEVICE_CASE(723)
HOST_TO_DEVICE_CASE(724)
HOST_TO_DEVICE_CASE(725)
HOST_TO_DEVICE_CASE(726)
HOST_TO_DEVICE_CASE(727)
HOST_TO_DEVICE_CASE(728)
HOST_TO_DEVICE_CASE(729)
HOST_TO_DEVICE_CASE(730)
HOST_TO_DEVICE_CASE(731)
HOST_TO_DEVICE_CASE(732)
HOST_TO_DEVICE_CASE(733)
HOST_TO_DEVICE_CASE(734)
HOST_TO_DEVICE_CASE(735)
HOST_TO_DEVICE_CASE(736)
HOST_TO_DEVICE_CASE(737)
HOST_TO_DEVICE_CASE(738)
HOST_TO_DEVICE_CASE(739)
HOST_TO_DEVICE_CASE(740)
HOST_TO_DEVICE_CASE(741)
HOST_TO_DEVICE_CASE(742)
HOST_TO_DEVICE_CASE(743)
HOST_TO_DEVICE_CASE(744)
HOST_TO_DEVICE_CASE(745)
HOST_TO_DEVICE_CASE(746)
HOST_TO_DEVICE_CASE(747)
HOST_TO_DEVICE_CASE(748)
HOST_TO_DEVICE_CASE(749)
HOST_TO_DEVICE_CASE(750)
HOST_TO_DEVICE_CASE(751)
HOST_TO_DEVICE_CASE(752)
HOST_TO_DEVICE_CASE(753)
HOST_TO_DEVICE_CASE(754)
HOST_TO_DEVICE_CASE(755)
HOST_TO_DEVICE_CASE(756)
HOST_TO_DEVICE_CASE(757)
HOST_TO_DEVICE_CASE(758)
HOST_TO_DEVICE_CASE(759)
HOST_TO_DEVICE_CASE(760)
HOST_TO_DEVICE_CASE(761)
HOST_TO_DEVICE_CASE(762)
HOST_TO_DEVICE_CASE(763)
HOST_TO_DEVICE_CASE(764)
HOST_TO_DEVICE_CASE(765)
HOST_TO_DEVICE_CASE(766)
HOST_TO_DEVICE_CASE(767)
HOST_TO_DEVICE_CASE(768)
HOST_TO_DEVICE_CASE(769)
HOST_TO_DEVICE_CASE(770)
HOST_TO_DEVICE_CASE(771)
HOST_TO_DEVICE_CASE(772)
HOST_TO_DEVICE_CASE(773)
HOST_TO_DEVICE_CASE(774)
HOST_TO_DEVICE_CASE(775)
HOST_TO_DEVICE_CASE(776)
HOST_TO_DEVICE_CASE(777)
HOST_TO_DEVICE_CASE(778)
HOST_TO_DEVICE_CASE(779)
HOST_TO_DEVICE_CASE(780)
HOST_TO_DEVICE_CASE(781)
HOST_TO_DEVICE_CASE(782)
HOST_TO_DEVICE_CASE(783)
HOST_TO_DEVICE_CASE(784)
HOST_TO_DEVICE_CASE(785)
HOST_TO_DEVICE_CASE(786)
HOST_TO_DEVICE_CASE(787)
HOST_TO_DEVICE_CASE(788)
HOST_TO_DEVICE_CASE(789)
HOST_TO_DEVICE_CASE(790)
HOST_TO_DEVICE_CASE(791)
HOST_TO_DEVICE_CASE(792)
HOST_TO_DEVICE_CASE(793)
HOST_TO_DEVICE_CASE(794)
HOST_TO_DEVICE_CASE(795)
HOST_TO_DEVICE_CASE(796)
HOST_TO_DEVICE_CASE(797)
HOST_TO_DEVICE_CASE(798)
HOST_TO_DEVICE_CASE(799)
HOST_TO_DEVICE_CASE(800)
HOST_TO_DEVICE_CASE(801)
HOST_TO_DEVICE_CASE(802)
HOST_TO_DEVICE_CASE(803)
HOST_TO_DEVICE_CASE(804)
HOST_TO_DEVICE_CASE(805)
HOST_TO_DEVICE_CASE(806)
HOST_TO_DEVICE_CASE(807)
HOST_TO_DEVICE_CASE(808)
HOST_TO_DEVICE_CASE(809)
HOST_TO_DEVICE_CASE(810)
HOST_TO_DEVICE_CASE(811)
HOST_TO_DEVICE_CASE(812)
HOST_TO_DEVICE_CASE(813)
HOST_TO_DEVICE_CASE(814)
HOST_TO_DEVICE_CASE(815)
HOST_TO_DEVICE_CASE(816)
HOST_TO_DEVICE_CASE(817)
HOST_TO_DEVICE_CASE(818)
HOST_TO_DEVICE_CASE(819)
HOST_TO_DEVICE_CASE(820)
HOST_TO_DEVICE_CASE(821)
HOST_TO_DEVICE_CASE(822)
HOST_TO_DEVICE_CASE(823)
HOST_TO_DEVICE_CASE(824)
HOST_TO_DEVICE_CASE(825)
HOST_TO_DEVICE_CASE(826)
HOST_TO_DEVICE_CASE(827)
HOST_TO_DEVICE_CASE(828)
HOST_TO_DEVICE_CASE(829)
HOST_TO_DEVICE_CASE(830)
HOST_TO_DEVICE_CASE(831)
HOST_TO_DEVICE_CASE(832)
HOST_TO_DEVICE_CASE(833)
HOST_TO_DEVICE_CASE(834)
HOST_TO_DEVICE_CASE(835)
HOST_TO_DEVICE_CASE(836)
HOST_TO_DEVICE_CASE(837)
HOST_TO_DEVICE_CASE(838)
HOST_TO_DEVICE_CASE(839)
HOST_TO_DEVICE_CASE(840)
HOST_TO_DEVICE_CASE(841)
HOST_TO_DEVICE_CASE(842)
HOST_TO_DEVICE_CASE(843)
HOST_TO_DEVICE_CASE(844)
HOST_TO_DEVICE_CASE(845)
HOST_TO_DEVICE_CASE(846)
HOST_TO_DEVICE_CASE(847)
HOST_TO_DEVICE_CASE(848)
HOST_TO_DEVICE_CASE(849)
HOST_TO_DEVICE_CASE(850)
HOST_TO_DEVICE_CASE(851)
HOST_TO_DEVICE_CASE(852)
HOST_TO_DEVICE_CASE(853)
HOST_TO_DEVICE_CASE(854)
HOST_TO_DEVICE_CASE(855)
HOST_TO_DEVICE_CASE(856)
HOST_TO_DEVICE_CASE(857)
HOST_TO_DEVICE_CASE(858)
HOST_TO_DEVICE_CASE(859)
HOST_TO_DEVICE_CASE(860)
HOST_TO_DEVICE_CASE(861)
HOST_TO_DEVICE_CASE(862)
HOST_TO_DEVICE_CASE(863)
HOST_TO_DEVICE_CASE(864)
HOST_TO_DEVICE_CASE(865)
HOST_TO_DEVICE_CASE(866)
HOST_TO_DEVICE_CASE(867)
HOST_TO_DEVICE_CASE(868)
HOST_TO_DEVICE_CASE(869)
HOST_TO_DEVICE_CASE(870)
HOST_TO_DEVICE_CASE(871)
HOST_TO_DEVICE_CASE(872)
HOST_TO_DEVICE_CASE(873)
HOST_TO_DEVICE_CASE(874)
HOST_TO_DEVICE_CASE(875)
HOST_TO_DEVICE_CASE(876)
HOST_TO_DEVICE_CASE(877)
HOST_TO_DEVICE_CASE(878)
HOST_TO_DEVICE_CASE(879)
HOST_TO_DEVICE_CASE(880)
HOST_TO_DEVICE_CASE(881)
HOST_TO_DEVICE_CASE(882)
HOST_TO_DEVICE_CASE(883)
HOST_TO_DEVICE_CASE(884)
HOST_TO_DEVICE_CASE(885)
HOST_TO_DEVICE_CASE(886)
HOST_TO_DEVICE_CASE(887)
HOST_TO_DEVICE_CASE(888)
HOST_TO_DEVICE_CASE(889)
HOST_TO_DEVICE_CASE(890)
HOST_TO_DEVICE_CASE(891)
HOST_TO_DEVICE_CASE(892)
HOST_TO_DEVICE_CASE(893)
HOST_TO_DEVICE_CASE(894)
HOST_TO_DEVICE_CASE(895)
HOST_TO_DEVICE_CASE(896)
HOST_TO_DEVICE_CASE(897)
HOST_TO_DEVICE_CASE(898)
HOST_TO_DEVICE_CASE(899)
HOST_TO_DEVICE_CASE(900)
HOST_TO_DEVICE_CASE(901)
HOST_TO_DEVICE_CASE(902)
HOST_TO_DEVICE_CASE(903)
HOST_TO_DEVICE_CASE(904)
HOST_TO_DEVICE_CASE(905)
HOST_TO_DEVICE_CASE(906)
HOST_TO_DEVICE_CASE(907)
HOST_TO_DEVICE_CASE(908)
HOST_TO_DEVICE_CASE(909)
HOST_TO_DEVICE_CASE(910)
HOST_TO_DEVICE_CASE(911)
HOST_TO_DEVICE_CASE(912)
HOST_TO_DEVICE_CASE(913)
HOST_TO_DEVICE_CASE(914)
HOST_TO_DEVICE_CASE(915)
HOST_TO_DEVICE_CASE(916)
HOST_TO_DEVICE_CASE(917)
HOST_TO_DEVICE_CASE(918)
HOST_TO_DEVICE_CASE(919)
HOST_TO_DEVICE_CASE(920)
HOST_TO_DEVICE_CASE(921)
HOST_TO_DEVICE_CASE(922)
HOST_TO_DEVICE_CASE(923)
HOST_TO_DEVICE_CASE(924)
HOST_TO_DEVICE_CASE(925)
HOST_TO_DEVICE_CASE(926)
HOST_TO_DEVICE_CASE(927)
HOST_TO_DEVICE_CASE(928)
HOST_TO_DEVICE_CASE(929)
HOST_TO_DEVICE_CASE(930)
HOST_TO_DEVICE_CASE(931)
HOST_TO_DEVICE_CASE(932)
HOST_TO_DEVICE_CASE(933)
HOST_TO_DEVICE_CASE(934)
HOST_TO_DEVICE_CASE(935)
HOST_TO_DEVICE_CASE(936)
HOST_TO_DEVICE_CASE(937)
HOST_TO_DEVICE_CASE(938)
HOST_TO_DEVICE_CASE(939)
HOST_TO_DEVICE_CASE(940)
HOST_TO_DEVICE_CASE(941)
HOST_TO_DEVICE_CASE(942)
HOST_TO_DEVICE_CASE(943)
HOST_TO_DEVICE_CASE(944)
HOST_TO_DEVICE_CASE(945)
HOST_TO_DEVICE_CASE(946)
HOST_TO_DEVICE_CASE(947)
HOST_TO_DEVICE_CASE(948)
HOST_TO_DEVICE_CASE(949)
HOST_TO_DEVICE_CASE(950)
HOST_TO_DEVICE_CASE(951)
HOST_TO_DEVICE_CASE(952)
HOST_TO_DEVICE_CASE(953)
HOST_TO_DEVICE_CASE(954)
HOST_TO_DEVICE_CASE(955)
HOST_TO_DEVICE_CASE(956)
HOST_TO_DEVICE_CASE(957)
HOST_TO_DEVICE_CASE(958)
HOST_TO_DEVICE_CASE(959)
HOST_TO_DEVICE_CASE(960)
HOST_TO_DEVICE_CASE(961)
HOST_TO_DEVICE_CASE(962)
HOST_TO_DEVICE_CASE(963)
HOST_TO_DEVICE_CASE(964)
HOST_TO_DEVICE_CASE(965)
HOST_TO_DEVICE_CASE(966)
HOST_TO_DEVICE_CASE(967)
HOST_TO_DEVICE_CASE(968)
HOST_TO_DEVICE_CASE(969)
HOST_TO_DEVICE_CASE(970)
HOST_TO_DEVICE_CASE(971)
HOST_TO_DEVICE_CASE(972)
HOST_TO_DEVICE_CASE(973)
HOST_TO_DEVICE_CASE(974)
HOST_TO_DEVICE_CASE(975)
HOST_TO_DEVICE_CASE(976)
HOST_TO_DEVICE_CASE(977)
HOST_TO_DEVICE_CASE(978)
HOST_TO_DEVICE_CASE(979)
HOST_TO_DEVICE_CASE(980)
HOST_TO_DEVICE_CASE(981)
HOST_TO_DEVICE_CASE(982)
HOST_TO_DEVICE_CASE(983)
HOST_TO_DEVICE_CASE(984)
HOST_TO_DEVICE_CASE(985)
HOST_TO_DEVICE_CASE(986)
HOST_TO_DEVICE_CASE(987)
HOST_TO_DEVICE_CASE(988)
HOST_TO_DEVICE_CASE(989)
HOST_TO_DEVICE_CASE(990)
HOST_TO_DEVICE_CASE(991)
HOST_TO_DEVICE_CASE(992)
HOST_TO_DEVICE_CASE(993)
HOST_TO_DEVICE_CASE(994)
HOST_TO_DEVICE_CASE(995)
HOST_TO_DEVICE_CASE(996)
HOST_TO_DEVICE_CASE(997)
HOST_TO_DEVICE_CASE(998)
HOST_TO_DEVICE_CASE(999)*/
    }
    
    void* embeddings_ptr = global_mem_ptr;
    
    int n_new_embeddings = 0;
    void* new_embeddings_ptr = (char*)embeddings_ptr + (n_embeddings)*(embedding_size);
    int max_embeddings = 2000000;
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
      case 9:
      {
        run_single_step_vector<9> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      case 10:
      {
        run_single_step_vector<10> (embeddings_ptr, n_embeddings, csr, 
                                  output_ptr, &n_output, new_embeddings_ptr, &n_new_embeddings);
        break;
      }
      RUN_KERNEL_CASE(11)
      RUN_KERNEL_CASE(12)
      RUN_KERNEL_CASE(13)
      RUN_KERNEL_CASE(14)
      RUN_KERNEL_CASE(15)
      RUN_KERNEL_CASE(16)
      RUN_KERNEL_CASE(17)
      RUN_KERNEL_CASE(18)
      RUN_KERNEL_CASE(19)
      RUN_KERNEL_CASE(20)
      RUN_KERNEL_CASE(21)
      RUN_KERNEL_CASE(22)
      RUN_KERNEL_CASE(23)
      RUN_KERNEL_CASE(24)
      RUN_KERNEL_CASE(25)
      RUN_KERNEL_CASE(26)
      RUN_KERNEL_CASE(27)
      RUN_KERNEL_CASE(28)
      RUN_KERNEL_CASE(29)
      RUN_KERNEL_CASE(30)
      RUN_KERNEL_CASE(31)
RUN_KERNEL_CASE(32)
RUN_KERNEL_CASE(33)
RUN_KERNEL_CASE(34)
RUN_KERNEL_CASE(35)
RUN_KERNEL_CASE(36)
RUN_KERNEL_CASE(37)
RUN_KERNEL_CASE(38)
RUN_KERNEL_CASE(39)
RUN_KERNEL_CASE(40)
RUN_KERNEL_CASE(41)
RUN_KERNEL_CASE(42)
RUN_KERNEL_CASE(43)
RUN_KERNEL_CASE(44)
RUN_KERNEL_CASE(45)
RUN_KERNEL_CASE(46)
RUN_KERNEL_CASE(47)
RUN_KERNEL_CASE(48)
RUN_KERNEL_CASE(49)
RUN_KERNEL_CASE(50)
RUN_KERNEL_CASE(51)
RUN_KERNEL_CASE(52)
RUN_KERNEL_CASE(53)
RUN_KERNEL_CASE(54)
RUN_KERNEL_CASE(55)
RUN_KERNEL_CASE(56)
RUN_KERNEL_CASE(57)
RUN_KERNEL_CASE(58)
RUN_KERNEL_CASE(59)
RUN_KERNEL_CASE(60)
RUN_KERNEL_CASE(61)
RUN_KERNEL_CASE(62)
RUN_KERNEL_CASE(63)
RUN_KERNEL_CASE(64)
RUN_KERNEL_CASE(65)
RUN_KERNEL_CASE(66)
RUN_KERNEL_CASE(67)
RUN_KERNEL_CASE(68)
RUN_KERNEL_CASE(69)
RUN_KERNEL_CASE(70)
RUN_KERNEL_CASE(71)
RUN_KERNEL_CASE(72)
RUN_KERNEL_CASE(73)
RUN_KERNEL_CASE(74)
RUN_KERNEL_CASE(75)
RUN_KERNEL_CASE(76)
RUN_KERNEL_CASE(77)
RUN_KERNEL_CASE(78)
RUN_KERNEL_CASE(79)
RUN_KERNEL_CASE(80)
RUN_KERNEL_CASE(81)
RUN_KERNEL_CASE(82)
RUN_KERNEL_CASE(83)
RUN_KERNEL_CASE(84)
RUN_KERNEL_CASE(85)
RUN_KERNEL_CASE(86)
RUN_KERNEL_CASE(87)
RUN_KERNEL_CASE(88)
RUN_KERNEL_CASE(89)
RUN_KERNEL_CASE(90)
RUN_KERNEL_CASE(91)
RUN_KERNEL_CASE(92)
RUN_KERNEL_CASE(93)
RUN_KERNEL_CASE(94)
RUN_KERNEL_CASE(95)
RUN_KERNEL_CASE(96)
RUN_KERNEL_CASE(97)
RUN_KERNEL_CASE(98)
RUN_KERNEL_CASE(99)
/*RUN_KERNEL_CASE(100)
RUN_KERNEL_CASE(101)
RUN_KERNEL_CASE(102)
RUN_KERNEL_CASE(103)
RUN_KERNEL_CASE(104)
RUN_KERNEL_CASE(105)
RUN_KERNEL_CASE(106)
RUN_KERNEL_CASE(107)
RUN_KERNEL_CASE(108)
RUN_KERNEL_CASE(109)
RUN_KERNEL_CASE(110)
RUN_KERNEL_CASE(111)
RUN_KERNEL_CASE(112)
RUN_KERNEL_CASE(113)
RUN_KERNEL_CASE(114)
RUN_KERNEL_CASE(115)
RUN_KERNEL_CASE(116)
RUN_KERNEL_CASE(117)
RUN_KERNEL_CASE(118)
RUN_KERNEL_CASE(119)
RUN_KERNEL_CASE(120)
RUN_KERNEL_CASE(121)
RUN_KERNEL_CASE(122)
RUN_KERNEL_CASE(123)
RUN_KERNEL_CASE(124)
RUN_KERNEL_CASE(125)
RUN_KERNEL_CASE(126)
RUN_KERNEL_CASE(127)
RUN_KERNEL_CASE(128)
RUN_KERNEL_CASE(129)
RUN_KERNEL_CASE(130)
RUN_KERNEL_CASE(131)
RUN_KERNEL_CASE(132)
RUN_KERNEL_CASE(133)
RUN_KERNEL_CASE(134)
RUN_KERNEL_CASE(135)
RUN_KERNEL_CASE(136)
RUN_KERNEL_CASE(137)
RUN_KERNEL_CASE(138)
RUN_KERNEL_CASE(139)
RUN_KERNEL_CASE(140)
RUN_KERNEL_CASE(141)
RUN_KERNEL_CASE(142)
RUN_KERNEL_CASE(143)
RUN_KERNEL_CASE(144)
RUN_KERNEL_CASE(145)
RUN_KERNEL_CASE(146)
RUN_KERNEL_CASE(147)
RUN_KERNEL_CASE(148)
RUN_KERNEL_CASE(149)
RUN_KERNEL_CASE(150)
RUN_KERNEL_CASE(151)
RUN_KERNEL_CASE(152)
RUN_KERNEL_CASE(153)
RUN_KERNEL_CASE(154)
RUN_KERNEL_CASE(155)
RUN_KERNEL_CASE(156)
RUN_KERNEL_CASE(157)
RUN_KERNEL_CASE(158)
RUN_KERNEL_CASE(159)
RUN_KERNEL_CASE(160)
RUN_KERNEL_CASE(161)
RUN_KERNEL_CASE(162)
RUN_KERNEL_CASE(163)
RUN_KERNEL_CASE(164)
RUN_KERNEL_CASE(165)
RUN_KERNEL_CASE(166)
RUN_KERNEL_CASE(167)
RUN_KERNEL_CASE(168)
RUN_KERNEL_CASE(169)
RUN_KERNEL_CASE(170)
RUN_KERNEL_CASE(171)
RUN_KERNEL_CASE(172)
RUN_KERNEL_CASE(173)
RUN_KERNEL_CASE(174)
RUN_KERNEL_CASE(175)
RUN_KERNEL_CASE(176)
RUN_KERNEL_CASE(177)
RUN_KERNEL_CASE(178)
RUN_KERNEL_CASE(179)
RUN_KERNEL_CASE(180)
RUN_KERNEL_CASE(181)
RUN_KERNEL_CASE(182)
RUN_KERNEL_CASE(183)
RUN_KERNEL_CASE(184)
RUN_KERNEL_CASE(185)
RUN_KERNEL_CASE(186)
RUN_KERNEL_CASE(187)
RUN_KERNEL_CASE(188)
RUN_KERNEL_CASE(189)
RUN_KERNEL_CASE(190)
RUN_KERNEL_CASE(191)
RUN_KERNEL_CASE(192)
RUN_KERNEL_CASE(193)
RUN_KERNEL_CASE(194)
RUN_KERNEL_CASE(195)
RUN_KERNEL_CASE(196)
RUN_KERNEL_CASE(197)
RUN_KERNEL_CASE(198)
RUN_KERNEL_CASE(199)
RUN_KERNEL_CASE(200)
RUN_KERNEL_CASE(201)
RUN_KERNEL_CASE(202)
RUN_KERNEL_CASE(203)
RUN_KERNEL_CASE(204)
RUN_KERNEL_CASE(205)
RUN_KERNEL_CASE(206)
RUN_KERNEL_CASE(207)
RUN_KERNEL_CASE(208)
RUN_KERNEL_CASE(209)
RUN_KERNEL_CASE(210)
RUN_KERNEL_CASE(211)
RUN_KERNEL_CASE(212)
RUN_KERNEL_CASE(213)
RUN_KERNEL_CASE(214)
RUN_KERNEL_CASE(215)
RUN_KERNEL_CASE(216)
RUN_KERNEL_CASE(217)
RUN_KERNEL_CASE(218)
RUN_KERNEL_CASE(219)
RUN_KERNEL_CASE(220)
RUN_KERNEL_CASE(221)
RUN_KERNEL_CASE(222)
RUN_KERNEL_CASE(223)
RUN_KERNEL_CASE(224)
RUN_KERNEL_CASE(225)
RUN_KERNEL_CASE(226)
RUN_KERNEL_CASE(227)
RUN_KERNEL_CASE(228)
RUN_KERNEL_CASE(229)
RUN_KERNEL_CASE(230)
RUN_KERNEL_CASE(231)
RUN_KERNEL_CASE(232)
RUN_KERNEL_CASE(233)
RUN_KERNEL_CASE(234)
RUN_KERNEL_CASE(235)
RUN_KERNEL_CASE(236)
RUN_KERNEL_CASE(237)
RUN_KERNEL_CASE(238)
RUN_KERNEL_CASE(239)
RUN_KERNEL_CASE(240)
RUN_KERNEL_CASE(241)
RUN_KERNEL_CASE(242)
RUN_KERNEL_CASE(243)
RUN_KERNEL_CASE(244)
RUN_KERNEL_CASE(245)
RUN_KERNEL_CASE(246)
RUN_KERNEL_CASE(247)
RUN_KERNEL_CASE(248)
RUN_KERNEL_CASE(249)
RUN_KERNEL_CASE(250)
RUN_KERNEL_CASE(251)
RUN_KERNEL_CASE(252)
RUN_KERNEL_CASE(253)
RUN_KERNEL_CASE(254)
RUN_KERNEL_CASE(255)
RUN_KERNEL_CASE(256)
RUN_KERNEL_CASE(257)
RUN_KERNEL_CASE(258)
RUN_KERNEL_CASE(259)
RUN_KERNEL_CASE(260)
RUN_KERNEL_CASE(261)
RUN_KERNEL_CASE(262)
RUN_KERNEL_CASE(263)
RUN_KERNEL_CASE(264)
RUN_KERNEL_CASE(265)
RUN_KERNEL_CASE(266)
RUN_KERNEL_CASE(267)
RUN_KERNEL_CASE(268)
RUN_KERNEL_CASE(269)
RUN_KERNEL_CASE(270)
RUN_KERNEL_CASE(271)
RUN_KERNEL_CASE(272)
RUN_KERNEL_CASE(273)
RUN_KERNEL_CASE(274)
RUN_KERNEL_CASE(275)
RUN_KERNEL_CASE(276)
RUN_KERNEL_CASE(277)
RUN_KERNEL_CASE(278)
RUN_KERNEL_CASE(279)
RUN_KERNEL_CASE(280)
RUN_KERNEL_CASE(281)
RUN_KERNEL_CASE(282)
RUN_KERNEL_CASE(283)
RUN_KERNEL_CASE(284)
RUN_KERNEL_CASE(285)
RUN_KERNEL_CASE(286)
RUN_KERNEL_CASE(287)
RUN_KERNEL_CASE(288)
RUN_KERNEL_CASE(289)
RUN_KERNEL_CASE(290)
RUN_KERNEL_CASE(291)
RUN_KERNEL_CASE(292)
RUN_KERNEL_CASE(293)
RUN_KERNEL_CASE(294)
RUN_KERNEL_CASE(295)
RUN_KERNEL_CASE(296)
RUN_KERNEL_CASE(297)
RUN_KERNEL_CASE(298)
RUN_KERNEL_CASE(299)
RUN_KERNEL_CASE(300)
RUN_KERNEL_CASE(301)
RUN_KERNEL_CASE(302)
RUN_KERNEL_CASE(303)
RUN_KERNEL_CASE(304)
RUN_KERNEL_CASE(305)
RUN_KERNEL_CASE(306)
RUN_KERNEL_CASE(307)
RUN_KERNEL_CASE(308)
RUN_KERNEL_CASE(309)
RUN_KERNEL_CASE(310)
RUN_KERNEL_CASE(311)
RUN_KERNEL_CASE(312)
RUN_KERNEL_CASE(313)
RUN_KERNEL_CASE(314)
RUN_KERNEL_CASE(315)
RUN_KERNEL_CASE(316)
RUN_KERNEL_CASE(317)
RUN_KERNEL_CASE(318)
RUN_KERNEL_CASE(319)
RUN_KERNEL_CASE(320)
RUN_KERNEL_CASE(321)
RUN_KERNEL_CASE(322)
RUN_KERNEL_CASE(323)
RUN_KERNEL_CASE(324)
RUN_KERNEL_CASE(325)
RUN_KERNEL_CASE(326)
RUN_KERNEL_CASE(327)
RUN_KERNEL_CASE(328)
RUN_KERNEL_CASE(329)
RUN_KERNEL_CASE(330)
RUN_KERNEL_CASE(331)
RUN_KERNEL_CASE(332)
RUN_KERNEL_CASE(333)
RUN_KERNEL_CASE(334)
RUN_KERNEL_CASE(335)
RUN_KERNEL_CASE(336)
RUN_KERNEL_CASE(337)
RUN_KERNEL_CASE(338)
RUN_KERNEL_CASE(339)
RUN_KERNEL_CASE(340)
RUN_KERNEL_CASE(341)
RUN_KERNEL_CASE(342)
RUN_KERNEL_CASE(343)
RUN_KERNEL_CASE(344)
RUN_KERNEL_CASE(345)
RUN_KERNEL_CASE(346)
RUN_KERNEL_CASE(347)
RUN_KERNEL_CASE(348)
RUN_KERNEL_CASE(349)
RUN_KERNEL_CASE(350)
RUN_KERNEL_CASE(351)
RUN_KERNEL_CASE(352)
RUN_KERNEL_CASE(353)
RUN_KERNEL_CASE(354)
RUN_KERNEL_CASE(355)
RUN_KERNEL_CASE(356)
RUN_KERNEL_CASE(357)
RUN_KERNEL_CASE(358)
RUN_KERNEL_CASE(359)
RUN_KERNEL_CASE(360)
RUN_KERNEL_CASE(361)
RUN_KERNEL_CASE(362)
RUN_KERNEL_CASE(363)
RUN_KERNEL_CASE(364)
RUN_KERNEL_CASE(365)
RUN_KERNEL_CASE(366)
RUN_KERNEL_CASE(367)
RUN_KERNEL_CASE(368)
RUN_KERNEL_CASE(369)
RUN_KERNEL_CASE(370)
RUN_KERNEL_CASE(371)
RUN_KERNEL_CASE(372)
RUN_KERNEL_CASE(373)
RUN_KERNEL_CASE(374)
RUN_KERNEL_CASE(375)
RUN_KERNEL_CASE(376)
RUN_KERNEL_CASE(377)
RUN_KERNEL_CASE(378)
RUN_KERNEL_CASE(379)
RUN_KERNEL_CASE(380)
RUN_KERNEL_CASE(381)
RUN_KERNEL_CASE(382)
RUN_KERNEL_CASE(383)
RUN_KERNEL_CASE(384)
RUN_KERNEL_CASE(385)
RUN_KERNEL_CASE(386)
RUN_KERNEL_CASE(387)
RUN_KERNEL_CASE(388)
RUN_KERNEL_CASE(389)
RUN_KERNEL_CASE(390)
RUN_KERNEL_CASE(391)
RUN_KERNEL_CASE(392)
RUN_KERNEL_CASE(393)
RUN_KERNEL_CASE(394)
RUN_KERNEL_CASE(395)
RUN_KERNEL_CASE(396)
RUN_KERNEL_CASE(397)
RUN_KERNEL_CASE(398)
RUN_KERNEL_CASE(399)
RUN_KERNEL_CASE(400)
RUN_KERNEL_CASE(401)
RUN_KERNEL_CASE(402)
RUN_KERNEL_CASE(403)
RUN_KERNEL_CASE(404)
RUN_KERNEL_CASE(405)
RUN_KERNEL_CASE(406)
RUN_KERNEL_CASE(407)
RUN_KERNEL_CASE(408)
RUN_KERNEL_CASE(409)
RUN_KERNEL_CASE(410)
RUN_KERNEL_CASE(411)
RUN_KERNEL_CASE(412)
RUN_KERNEL_CASE(413)
RUN_KERNEL_CASE(414)
RUN_KERNEL_CASE(415)
RUN_KERNEL_CASE(416)
RUN_KERNEL_CASE(417)
RUN_KERNEL_CASE(418)
RUN_KERNEL_CASE(419)
RUN_KERNEL_CASE(420)
RUN_KERNEL_CASE(421)
RUN_KERNEL_CASE(422)
RUN_KERNEL_CASE(423)
RUN_KERNEL_CASE(424)
RUN_KERNEL_CASE(425)
RUN_KERNEL_CASE(426)
RUN_KERNEL_CASE(427)
RUN_KERNEL_CASE(428)
RUN_KERNEL_CASE(429)
RUN_KERNEL_CASE(430)
RUN_KERNEL_CASE(431)
RUN_KERNEL_CASE(432)
RUN_KERNEL_CASE(433)
RUN_KERNEL_CASE(434)
RUN_KERNEL_CASE(435)
RUN_KERNEL_CASE(436)
RUN_KERNEL_CASE(437)
RUN_KERNEL_CASE(438)
RUN_KERNEL_CASE(439)
RUN_KERNEL_CASE(440)
RUN_KERNEL_CASE(441)
RUN_KERNEL_CASE(442)
RUN_KERNEL_CASE(443)
RUN_KERNEL_CASE(444)
RUN_KERNEL_CASE(445)
RUN_KERNEL_CASE(446)
RUN_KERNEL_CASE(447)
RUN_KERNEL_CASE(448)
RUN_KERNEL_CASE(449)
RUN_KERNEL_CASE(450)
RUN_KERNEL_CASE(451)
RUN_KERNEL_CASE(452)
RUN_KERNEL_CASE(453)
RUN_KERNEL_CASE(454)
RUN_KERNEL_CASE(455)
RUN_KERNEL_CASE(456)
RUN_KERNEL_CASE(457)
RUN_KERNEL_CASE(458)
RUN_KERNEL_CASE(459)
RUN_KERNEL_CASE(460)
RUN_KERNEL_CASE(461)
RUN_KERNEL_CASE(462)
RUN_KERNEL_CASE(463)
RUN_KERNEL_CASE(464)
RUN_KERNEL_CASE(465)
RUN_KERNEL_CASE(466)
RUN_KERNEL_CASE(467)
RUN_KERNEL_CASE(468)
RUN_KERNEL_CASE(469)
RUN_KERNEL_CASE(470)
RUN_KERNEL_CASE(471)
RUN_KERNEL_CASE(472)
RUN_KERNEL_CASE(473)
RUN_KERNEL_CASE(474)
RUN_KERNEL_CASE(475)
RUN_KERNEL_CASE(476)
RUN_KERNEL_CASE(477)
RUN_KERNEL_CASE(478)
RUN_KERNEL_CASE(479)
RUN_KERNEL_CASE(480)
RUN_KERNEL_CASE(481)
RUN_KERNEL_CASE(482)
RUN_KERNEL_CASE(483)
RUN_KERNEL_CASE(484)
RUN_KERNEL_CASE(485)
RUN_KERNEL_CASE(486)
RUN_KERNEL_CASE(487)
RUN_KERNEL_CASE(488)
RUN_KERNEL_CASE(489)
RUN_KERNEL_CASE(490)
RUN_KERNEL_CASE(491)
RUN_KERNEL_CASE(492)
RUN_KERNEL_CASE(493)
RUN_KERNEL_CASE(494)
RUN_KERNEL_CASE(495)
RUN_KERNEL_CASE(496)
RUN_KERNEL_CASE(497)
RUN_KERNEL_CASE(498)
RUN_KERNEL_CASE(499)
RUN_KERNEL_CASE(500)
RUN_KERNEL_CASE(501)
RUN_KERNEL_CASE(502)
RUN_KERNEL_CASE(503)
RUN_KERNEL_CASE(504)
RUN_KERNEL_CASE(505)
RUN_KERNEL_CASE(506)
RUN_KERNEL_CASE(507)
RUN_KERNEL_CASE(508)
RUN_KERNEL_CASE(509)
RUN_KERNEL_CASE(510)
RUN_KERNEL_CASE(511)
RUN_KERNEL_CASE(512)
RUN_KERNEL_CASE(513)
RUN_KERNEL_CASE(514)
RUN_KERNEL_CASE(515)
RUN_KERNEL_CASE(516)
RUN_KERNEL_CASE(517)
RUN_KERNEL_CASE(518)
RUN_KERNEL_CASE(519)
RUN_KERNEL_CASE(520)
RUN_KERNEL_CASE(521)
RUN_KERNEL_CASE(522)
RUN_KERNEL_CASE(523)
RUN_KERNEL_CASE(524)
RUN_KERNEL_CASE(525)
RUN_KERNEL_CASE(526)
RUN_KERNEL_CASE(527)
RUN_KERNEL_CASE(528)
RUN_KERNEL_CASE(529)
RUN_KERNEL_CASE(530)
RUN_KERNEL_CASE(531)
RUN_KERNEL_CASE(532)
RUN_KERNEL_CASE(533)
RUN_KERNEL_CASE(534)
RUN_KERNEL_CASE(535)
RUN_KERNEL_CASE(536)
RUN_KERNEL_CASE(537)
RUN_KERNEL_CASE(538)
RUN_KERNEL_CASE(539)
RUN_KERNEL_CASE(540)
RUN_KERNEL_CASE(541)
RUN_KERNEL_CASE(542)
RUN_KERNEL_CASE(543)
RUN_KERNEL_CASE(544)
RUN_KERNEL_CASE(545)
RUN_KERNEL_CASE(546)
RUN_KERNEL_CASE(547)
RUN_KERNEL_CASE(548)
RUN_KERNEL_CASE(549)
RUN_KERNEL_CASE(550)
RUN_KERNEL_CASE(551)
RUN_KERNEL_CASE(552)
RUN_KERNEL_CASE(553)
RUN_KERNEL_CASE(554)
RUN_KERNEL_CASE(555)
RUN_KERNEL_CASE(556)
RUN_KERNEL_CASE(557)
RUN_KERNEL_CASE(558)
RUN_KERNEL_CASE(559)
RUN_KERNEL_CASE(560)
RUN_KERNEL_CASE(561)
RUN_KERNEL_CASE(562)
RUN_KERNEL_CASE(563)
RUN_KERNEL_CASE(564)
RUN_KERNEL_CASE(565)
RUN_KERNEL_CASE(566)
RUN_KERNEL_CASE(567)
RUN_KERNEL_CASE(568)
RUN_KERNEL_CASE(569)
RUN_KERNEL_CASE(570)
RUN_KERNEL_CASE(571)
RUN_KERNEL_CASE(572)
RUN_KERNEL_CASE(573)
RUN_KERNEL_CASE(574)
RUN_KERNEL_CASE(575)
RUN_KERNEL_CASE(576)
RUN_KERNEL_CASE(577)
RUN_KERNEL_CASE(578)
RUN_KERNEL_CASE(579)
RUN_KERNEL_CASE(580)
RUN_KERNEL_CASE(581)
RUN_KERNEL_CASE(582)
RUN_KERNEL_CASE(583)
RUN_KERNEL_CASE(584)
RUN_KERNEL_CASE(585)
RUN_KERNEL_CASE(586)
RUN_KERNEL_CASE(587)
RUN_KERNEL_CASE(588)
RUN_KERNEL_CASE(589)
RUN_KERNEL_CASE(590)
RUN_KERNEL_CASE(591)
RUN_KERNEL_CASE(592)
RUN_KERNEL_CASE(593)
RUN_KERNEL_CASE(594)
RUN_KERNEL_CASE(595)
RUN_KERNEL_CASE(596)
RUN_KERNEL_CASE(597)
RUN_KERNEL_CASE(598)
RUN_KERNEL_CASE(599)
RUN_KERNEL_CASE(600)
RUN_KERNEL_CASE(601)
RUN_KERNEL_CASE(602)
RUN_KERNEL_CASE(603)
RUN_KERNEL_CASE(604)
RUN_KERNEL_CASE(605)
RUN_KERNEL_CASE(606)
RUN_KERNEL_CASE(607)
RUN_KERNEL_CASE(608)
RUN_KERNEL_CASE(609)
RUN_KERNEL_CASE(610)
RUN_KERNEL_CASE(611)
RUN_KERNEL_CASE(612)
RUN_KERNEL_CASE(613)
RUN_KERNEL_CASE(614)
RUN_KERNEL_CASE(615)
RUN_KERNEL_CASE(616)
RUN_KERNEL_CASE(617)
RUN_KERNEL_CASE(618)
RUN_KERNEL_CASE(619)
RUN_KERNEL_CASE(620)
RUN_KERNEL_CASE(621)
RUN_KERNEL_CASE(622)
RUN_KERNEL_CASE(623)
RUN_KERNEL_CASE(624)
RUN_KERNEL_CASE(625)
RUN_KERNEL_CASE(626)
RUN_KERNEL_CASE(627)
RUN_KERNEL_CASE(628)
RUN_KERNEL_CASE(629)
RUN_KERNEL_CASE(630)
RUN_KERNEL_CASE(631)
RUN_KERNEL_CASE(632)
RUN_KERNEL_CASE(633)
RUN_KERNEL_CASE(634)
RUN_KERNEL_CASE(635)
RUN_KERNEL_CASE(636)
RUN_KERNEL_CASE(637)
RUN_KERNEL_CASE(638)
RUN_KERNEL_CASE(639)
RUN_KERNEL_CASE(640)
RUN_KERNEL_CASE(641)
RUN_KERNEL_CASE(642)
RUN_KERNEL_CASE(643)
RUN_KERNEL_CASE(644)
RUN_KERNEL_CASE(645)
RUN_KERNEL_CASE(646)
RUN_KERNEL_CASE(647)
RUN_KERNEL_CASE(648)
RUN_KERNEL_CASE(649)
RUN_KERNEL_CASE(650)
RUN_KERNEL_CASE(651)
RUN_KERNEL_CASE(652)
RUN_KERNEL_CASE(653)
RUN_KERNEL_CASE(654)
RUN_KERNEL_CASE(655)
RUN_KERNEL_CASE(656)
RUN_KERNEL_CASE(657)
RUN_KERNEL_CASE(658)
RUN_KERNEL_CASE(659)
RUN_KERNEL_CASE(660)
RUN_KERNEL_CASE(661)
RUN_KERNEL_CASE(662)
RUN_KERNEL_CASE(663)
RUN_KERNEL_CASE(664)
RUN_KERNEL_CASE(665)
RUN_KERNEL_CASE(666)
RUN_KERNEL_CASE(667)
RUN_KERNEL_CASE(668)
RUN_KERNEL_CASE(669)
RUN_KERNEL_CASE(670)
RUN_KERNEL_CASE(671)
RUN_KERNEL_CASE(672)
RUN_KERNEL_CASE(673)
RUN_KERNEL_CASE(674)
RUN_KERNEL_CASE(675)
RUN_KERNEL_CASE(676)
RUN_KERNEL_CASE(677)
RUN_KERNEL_CASE(678)
RUN_KERNEL_CASE(679)
RUN_KERNEL_CASE(680)
RUN_KERNEL_CASE(681)
RUN_KERNEL_CASE(682)
RUN_KERNEL_CASE(683)
RUN_KERNEL_CASE(684)
RUN_KERNEL_CASE(685)
RUN_KERNEL_CASE(686)
RUN_KERNEL_CASE(687)
RUN_KERNEL_CASE(688)
RUN_KERNEL_CASE(689)
RUN_KERNEL_CASE(690)
RUN_KERNEL_CASE(691)
RUN_KERNEL_CASE(692)
RUN_KERNEL_CASE(693)
RUN_KERNEL_CASE(694)
RUN_KERNEL_CASE(695)
RUN_KERNEL_CASE(696)
RUN_KERNEL_CASE(697)
RUN_KERNEL_CASE(698)
RUN_KERNEL_CASE(699)
RUN_KERNEL_CASE(700)
RUN_KERNEL_CASE(701)
RUN_KERNEL_CASE(702)
RUN_KERNEL_CASE(703)
RUN_KERNEL_CASE(704)
RUN_KERNEL_CASE(705)
RUN_KERNEL_CASE(706)
RUN_KERNEL_CASE(707)
RUN_KERNEL_CASE(708)
RUN_KERNEL_CASE(709)
RUN_KERNEL_CASE(710)
RUN_KERNEL_CASE(711)
RUN_KERNEL_CASE(712)
RUN_KERNEL_CASE(713)
RUN_KERNEL_CASE(714)
RUN_KERNEL_CASE(715)
RUN_KERNEL_CASE(716)
RUN_KERNEL_CASE(717)
RUN_KERNEL_CASE(718)
RUN_KERNEL_CASE(719)
RUN_KERNEL_CASE(720)
RUN_KERNEL_CASE(721)
RUN_KERNEL_CASE(722)
RUN_KERNEL_CASE(723)
RUN_KERNEL_CASE(724)
RUN_KERNEL_CASE(725)
RUN_KERNEL_CASE(726)
RUN_KERNEL_CASE(727)
RUN_KERNEL_CASE(728)
RUN_KERNEL_CASE(729)
RUN_KERNEL_CASE(730)
RUN_KERNEL_CASE(731)
RUN_KERNEL_CASE(732)
RUN_KERNEL_CASE(733)
RUN_KERNEL_CASE(734)
RUN_KERNEL_CASE(735)
RUN_KERNEL_CASE(736)
RUN_KERNEL_CASE(737)
RUN_KERNEL_CASE(738)
RUN_KERNEL_CASE(739)
RUN_KERNEL_CASE(740)
RUN_KERNEL_CASE(741)
RUN_KERNEL_CASE(742)
RUN_KERNEL_CASE(743)
RUN_KERNEL_CASE(744)
RUN_KERNEL_CASE(745)
RUN_KERNEL_CASE(746)
RUN_KERNEL_CASE(747)
RUN_KERNEL_CASE(748)
RUN_KERNEL_CASE(749)
RUN_KERNEL_CASE(750)
RUN_KERNEL_CASE(751)
RUN_KERNEL_CASE(752)
RUN_KERNEL_CASE(753)
RUN_KERNEL_CASE(754)
RUN_KERNEL_CASE(755)
RUN_KERNEL_CASE(756)
RUN_KERNEL_CASE(757)
RUN_KERNEL_CASE(758)
RUN_KERNEL_CASE(759)
RUN_KERNEL_CASE(760)
RUN_KERNEL_CASE(761)
RUN_KERNEL_CASE(762)
RUN_KERNEL_CASE(763)
RUN_KERNEL_CASE(764)
RUN_KERNEL_CASE(765)
RUN_KERNEL_CASE(766)
RUN_KERNEL_CASE(767)
RUN_KERNEL_CASE(768)
RUN_KERNEL_CASE(769)
RUN_KERNEL_CASE(770)
RUN_KERNEL_CASE(771)
RUN_KERNEL_CASE(772)
RUN_KERNEL_CASE(773)
RUN_KERNEL_CASE(774)
RUN_KERNEL_CASE(775)
RUN_KERNEL_CASE(776)
RUN_KERNEL_CASE(777)
RUN_KERNEL_CASE(778)
RUN_KERNEL_CASE(779)
RUN_KERNEL_CASE(780)
RUN_KERNEL_CASE(781)
RUN_KERNEL_CASE(782)
RUN_KERNEL_CASE(783)
RUN_KERNEL_CASE(784)
RUN_KERNEL_CASE(785)
RUN_KERNEL_CASE(786)
RUN_KERNEL_CASE(787)
RUN_KERNEL_CASE(788)
RUN_KERNEL_CASE(789)
RUN_KERNEL_CASE(790)
RUN_KERNEL_CASE(791)
RUN_KERNEL_CASE(792)
RUN_KERNEL_CASE(793)
RUN_KERNEL_CASE(794)
RUN_KERNEL_CASE(795)
RUN_KERNEL_CASE(796)
RUN_KERNEL_CASE(797)
RUN_KERNEL_CASE(798)
RUN_KERNEL_CASE(799)
RUN_KERNEL_CASE(800)
RUN_KERNEL_CASE(801)
RUN_KERNEL_CASE(802)
RUN_KERNEL_CASE(803)
RUN_KERNEL_CASE(804)
RUN_KERNEL_CASE(805)
RUN_KERNEL_CASE(806)
RUN_KERNEL_CASE(807)
RUN_KERNEL_CASE(808)
RUN_KERNEL_CASE(809)
RUN_KERNEL_CASE(810)
RUN_KERNEL_CASE(811)
RUN_KERNEL_CASE(812)
RUN_KERNEL_CASE(813)
RUN_KERNEL_CASE(814)
RUN_KERNEL_CASE(815)
RUN_KERNEL_CASE(816)
RUN_KERNEL_CASE(817)
RUN_KERNEL_CASE(818)
RUN_KERNEL_CASE(819)
RUN_KERNEL_CASE(820)
RUN_KERNEL_CASE(821)
RUN_KERNEL_CASE(822)
RUN_KERNEL_CASE(823)
RUN_KERNEL_CASE(824)
RUN_KERNEL_CASE(825)
RUN_KERNEL_CASE(826)
RUN_KERNEL_CASE(827)
RUN_KERNEL_CASE(828)
RUN_KERNEL_CASE(829)
RUN_KERNEL_CASE(830)
RUN_KERNEL_CASE(831)
RUN_KERNEL_CASE(832)
RUN_KERNEL_CASE(833)
RUN_KERNEL_CASE(834)
RUN_KERNEL_CASE(835)
RUN_KERNEL_CASE(836)
RUN_KERNEL_CASE(837)
RUN_KERNEL_CASE(838)
RUN_KERNEL_CASE(839)
RUN_KERNEL_CASE(840)
RUN_KERNEL_CASE(841)
RUN_KERNEL_CASE(842)
RUN_KERNEL_CASE(843)
RUN_KERNEL_CASE(844)
RUN_KERNEL_CASE(845)
RUN_KERNEL_CASE(846)
RUN_KERNEL_CASE(847)
RUN_KERNEL_CASE(848)
RUN_KERNEL_CASE(849)
RUN_KERNEL_CASE(850)
RUN_KERNEL_CASE(851)
RUN_KERNEL_CASE(852)
RUN_KERNEL_CASE(853)
RUN_KERNEL_CASE(854)
RUN_KERNEL_CASE(855)
RUN_KERNEL_CASE(856)
RUN_KERNEL_CASE(857)
RUN_KERNEL_CASE(858)
RUN_KERNEL_CASE(859)
RUN_KERNEL_CASE(860)
RUN_KERNEL_CASE(861)
RUN_KERNEL_CASE(862)
RUN_KERNEL_CASE(863)
RUN_KERNEL_CASE(864)
RUN_KERNEL_CASE(865)
RUN_KERNEL_CASE(866)
RUN_KERNEL_CASE(867)
RUN_KERNEL_CASE(868)
RUN_KERNEL_CASE(869)
RUN_KERNEL_CASE(870)
RUN_KERNEL_CASE(871)
RUN_KERNEL_CASE(872)
RUN_KERNEL_CASE(873)
RUN_KERNEL_CASE(874)
RUN_KERNEL_CASE(875)
RUN_KERNEL_CASE(876)
RUN_KERNEL_CASE(877)
RUN_KERNEL_CASE(878)
RUN_KERNEL_CASE(879)
RUN_KERNEL_CASE(880)
RUN_KERNEL_CASE(881)
RUN_KERNEL_CASE(882)
RUN_KERNEL_CASE(883)
RUN_KERNEL_CASE(884)
RUN_KERNEL_CASE(885)
RUN_KERNEL_CASE(886)
RUN_KERNEL_CASE(887)
RUN_KERNEL_CASE(888)
RUN_KERNEL_CASE(889)
RUN_KERNEL_CASE(890)
RUN_KERNEL_CASE(891)
RUN_KERNEL_CASE(892)
RUN_KERNEL_CASE(893)
RUN_KERNEL_CASE(894)
RUN_KERNEL_CASE(895)
RUN_KERNEL_CASE(896)
RUN_KERNEL_CASE(897)
RUN_KERNEL_CASE(898)
RUN_KERNEL_CASE(899)
RUN_KERNEL_CASE(900)
RUN_KERNEL_CASE(901)
RUN_KERNEL_CASE(902)
RUN_KERNEL_CASE(903)
RUN_KERNEL_CASE(904)
RUN_KERNEL_CASE(905)
RUN_KERNEL_CASE(906)
RUN_KERNEL_CASE(907)
RUN_KERNEL_CASE(908)
RUN_KERNEL_CASE(909)
RUN_KERNEL_CASE(910)
RUN_KERNEL_CASE(911)
RUN_KERNEL_CASE(912)
RUN_KERNEL_CASE(913)
RUN_KERNEL_CASE(914)
RUN_KERNEL_CASE(915)
RUN_KERNEL_CASE(916)
RUN_KERNEL_CASE(917)
RUN_KERNEL_CASE(918)
RUN_KERNEL_CASE(919)
RUN_KERNEL_CASE(920)
RUN_KERNEL_CASE(921)
RUN_KERNEL_CASE(922)
RUN_KERNEL_CASE(923)
RUN_KERNEL_CASE(924)
RUN_KERNEL_CASE(925)
RUN_KERNEL_CASE(926)
RUN_KERNEL_CASE(927)
RUN_KERNEL_CASE(928)
RUN_KERNEL_CASE(929)
RUN_KERNEL_CASE(930)
RUN_KERNEL_CASE(931)
RUN_KERNEL_CASE(932)
RUN_KERNEL_CASE(933)
RUN_KERNEL_CASE(934)
RUN_KERNEL_CASE(935)
RUN_KERNEL_CASE(936)
RUN_KERNEL_CASE(937)
RUN_KERNEL_CASE(938)
RUN_KERNEL_CASE(939)
RUN_KERNEL_CASE(940)
RUN_KERNEL_CASE(941)
RUN_KERNEL_CASE(942)
RUN_KERNEL_CASE(943)
RUN_KERNEL_CASE(944)
RUN_KERNEL_CASE(945)
RUN_KERNEL_CASE(946)
RUN_KERNEL_CASE(947)
RUN_KERNEL_CASE(948)
RUN_KERNEL_CASE(949)
RUN_KERNEL_CASE(950)
RUN_KERNEL_CASE(951)
RUN_KERNEL_CASE(952)
RUN_KERNEL_CASE(953)
RUN_KERNEL_CASE(954)
RUN_KERNEL_CASE(955)
RUN_KERNEL_CASE(956)
RUN_KERNEL_CASE(957)
RUN_KERNEL_CASE(958)
RUN_KERNEL_CASE(959)
RUN_KERNEL_CASE(960)
RUN_KERNEL_CASE(961)
RUN_KERNEL_CASE(962)
RUN_KERNEL_CASE(963)
RUN_KERNEL_CASE(964)
RUN_KERNEL_CASE(965)
RUN_KERNEL_CASE(966)
RUN_KERNEL_CASE(967)
RUN_KERNEL_CASE(968)
RUN_KERNEL_CASE(969)
RUN_KERNEL_CASE(970)
RUN_KERNEL_CASE(971)
RUN_KERNEL_CASE(972)
RUN_KERNEL_CASE(973)
RUN_KERNEL_CASE(974)
RUN_KERNEL_CASE(975)
RUN_KERNEL_CASE(976)
RUN_KERNEL_CASE(977)
RUN_KERNEL_CASE(978)
RUN_KERNEL_CASE(979)
RUN_KERNEL_CASE(980)
RUN_KERNEL_CASE(981)
RUN_KERNEL_CASE(982)
RUN_KERNEL_CASE(983)
RUN_KERNEL_CASE(984)
RUN_KERNEL_CASE(985)
RUN_KERNEL_CASE(986)
RUN_KERNEL_CASE(987)
RUN_KERNEL_CASE(988)
RUN_KERNEL_CASE(989)
RUN_KERNEL_CASE(990)
RUN_KERNEL_CASE(991)
RUN_KERNEL_CASE(992)
RUN_KERNEL_CASE(993)
RUN_KERNEL_CASE(994)
RUN_KERNEL_CASE(995)
RUN_KERNEL_CASE(996)
RUN_KERNEL_CASE(997)
RUN_KERNEL_CASE(998)
RUN_KERNEL_CASE(999)*/

    }
    
    double t2 = convertTimeValToDouble (getTimeOfDay ());
    std::cout << "step execution time " << (t2-t1) << " secs" << std::endl;
    total_time += (t2-t1);
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
          //((VectorVertexEmbedding<2>*)output_ptr)[i].print (std::cout);
          //std::cout << std::endl;
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

      case 8: {
        VectorVertexEmbedding<9>* new_embeddings = new VectorVertexEmbedding<9>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<9> embedding = ((VectorVertexEmbedding<9>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_9.push_back (((VectorVertexEmbedding<9>*)output_ptr)[i]);
        }
        break;
      }

      case 9: {
        VectorVertexEmbedding<10>* new_embeddings = new VectorVertexEmbedding<10>[n_new_embeddings];
        
        for (int i = 0; i < n_new_embeddings; i++) {
          VectorVertexEmbedding<10> embedding = ((VectorVertexEmbedding<10>*)new_embeddings_ptr)[i];
          new_embeddings [i] = embedding;
          #ifdef DEBUG
          if (embedding.get_n_vertices () != (iter + 1)) {
            printf ("embedding has %d vertices\n", embedding.get_n_vertices ());
          }
          #endif
        }
        
        embeddings = &new_embeddings[0];
        for (int i = 0; i < n_output; i++) {
          output_10.push_back (((VectorVertexEmbedding<10>*)output_ptr)[i]);
        }
        break;
      }

      DEVICE_TO_HOST_CASE(10, 11)
      DEVICE_TO_HOST_CASE(11, 12)
DEVICE_TO_HOST_CASE(12,13)
DEVICE_TO_HOST_CASE(13,14)
DEVICE_TO_HOST_CASE(14,15)
DEVICE_TO_HOST_CASE(15,16)
DEVICE_TO_HOST_CASE(16,17)
DEVICE_TO_HOST_CASE(17,18)
DEVICE_TO_HOST_CASE(18,19)
DEVICE_TO_HOST_CASE(19,20)
DEVICE_TO_HOST_CASE(20,21)
DEVICE_TO_HOST_CASE(21,22)
DEVICE_TO_HOST_CASE(22,23)
DEVICE_TO_HOST_CASE(23,24)
DEVICE_TO_HOST_CASE(24,25)
DEVICE_TO_HOST_CASE(25,26)
DEVICE_TO_HOST_CASE(26,27)
DEVICE_TO_HOST_CASE(27,28)
DEVICE_TO_HOST_CASE(28,29)
DEVICE_TO_HOST_CASE(29,30)
DEVICE_TO_HOST_CASE(30,31)
DEVICE_TO_HOST_CASE(31,32)
DEVICE_TO_HOST_CASE(32,33)
DEVICE_TO_HOST_CASE(33,34)
DEVICE_TO_HOST_CASE(34,35)
DEVICE_TO_HOST_CASE(35,36)
DEVICE_TO_HOST_CASE(36,37)
DEVICE_TO_HOST_CASE(37,38)
DEVICE_TO_HOST_CASE(38,39)
DEVICE_TO_HOST_CASE(39,40)
DEVICE_TO_HOST_CASE(40,41)
DEVICE_TO_HOST_CASE(41,42)
DEVICE_TO_HOST_CASE(42,43)
DEVICE_TO_HOST_CASE(43,44)
DEVICE_TO_HOST_CASE(44,45)
DEVICE_TO_HOST_CASE(45,46)
DEVICE_TO_HOST_CASE(46,47)
DEVICE_TO_HOST_CASE(47,48)
DEVICE_TO_HOST_CASE(48,49)
DEVICE_TO_HOST_CASE(49,50)
DEVICE_TO_HOST_CASE(50,51)
DEVICE_TO_HOST_CASE(51,52)
DEVICE_TO_HOST_CASE(52,53)
DEVICE_TO_HOST_CASE(53,54)
DEVICE_TO_HOST_CASE(54,55)
DEVICE_TO_HOST_CASE(55,56)
DEVICE_TO_HOST_CASE(56,57)
DEVICE_TO_HOST_CASE(57,58)
DEVICE_TO_HOST_CASE(58,59)
DEVICE_TO_HOST_CASE(59,60)
DEVICE_TO_HOST_CASE(60,61)
DEVICE_TO_HOST_CASE(61,62)
DEVICE_TO_HOST_CASE(62,63)
DEVICE_TO_HOST_CASE(63,64)
DEVICE_TO_HOST_CASE(64,65)
DEVICE_TO_HOST_CASE(65,66)
DEVICE_TO_HOST_CASE(66,67)
DEVICE_TO_HOST_CASE(67,68)
DEVICE_TO_HOST_CASE(68,69)
DEVICE_TO_HOST_CASE(69,70)
DEVICE_TO_HOST_CASE(70,71)
DEVICE_TO_HOST_CASE(71,72)
DEVICE_TO_HOST_CASE(72,73)
DEVICE_TO_HOST_CASE(73,74)
DEVICE_TO_HOST_CASE(74,75)
DEVICE_TO_HOST_CASE(75,76)
DEVICE_TO_HOST_CASE(76,77)
DEVICE_TO_HOST_CASE(77,78)
DEVICE_TO_HOST_CASE(78,79)
DEVICE_TO_HOST_CASE(79,80)
DEVICE_TO_HOST_CASE(80,81)
DEVICE_TO_HOST_CASE(81,82)
DEVICE_TO_HOST_CASE(82,83)
DEVICE_TO_HOST_CASE(83,84)
DEVICE_TO_HOST_CASE(84,85)
DEVICE_TO_HOST_CASE(85,86)
DEVICE_TO_HOST_CASE(86,87)
DEVICE_TO_HOST_CASE(87,88)
DEVICE_TO_HOST_CASE(88,89)
DEVICE_TO_HOST_CASE(89,90)
DEVICE_TO_HOST_CASE(90,91)
DEVICE_TO_HOST_CASE(91,92)
DEVICE_TO_HOST_CASE(92,93)
DEVICE_TO_HOST_CASE(93,94)
DEVICE_TO_HOST_CASE(94,95)
DEVICE_TO_HOST_CASE(95,96)
DEVICE_TO_HOST_CASE(96,97)
DEVICE_TO_HOST_CASE(97,98)
DEVICE_TO_HOST_CASE(98,99)
DEVICE_TO_HOST_CASE(99,100)
/*DEVICE_TO_HOST_CASE(100,101)
DEVICE_TO_HOST_CASE(101,102)
DEVICE_TO_HOST_CASE(102,103)
DEVICE_TO_HOST_CASE(103,104)
DEVICE_TO_HOST_CASE(104,105)
DEVICE_TO_HOST_CASE(105,106)
DEVICE_TO_HOST_CASE(106,107)
DEVICE_TO_HOST_CASE(107,108)
DEVICE_TO_HOST_CASE(108,109)
DEVICE_TO_HOST_CASE(109,110)
DEVICE_TO_HOST_CASE(110,111)
DEVICE_TO_HOST_CASE(111,112)
DEVICE_TO_HOST_CASE(112,113)
DEVICE_TO_HOST_CASE(113,114)
DEVICE_TO_HOST_CASE(114,115)
DEVICE_TO_HOST_CASE(115,116)
DEVICE_TO_HOST_CASE(116,117)
DEVICE_TO_HOST_CASE(117,118)
DEVICE_TO_HOST_CASE(118,119)
DEVICE_TO_HOST_CASE(119,120)
DEVICE_TO_HOST_CASE(120,121)
DEVICE_TO_HOST_CASE(121,122)
DEVICE_TO_HOST_CASE(122,123)
DEVICE_TO_HOST_CASE(123,124)
DEVICE_TO_HOST_CASE(124,125)
DEVICE_TO_HOST_CASE(125,126)
DEVICE_TO_HOST_CASE(126,127)
DEVICE_TO_HOST_CASE(127,128)
DEVICE_TO_HOST_CASE(128,129)
DEVICE_TO_HOST_CASE(129,130)
DEVICE_TO_HOST_CASE(130,131)
DEVICE_TO_HOST_CASE(131,132)
DEVICE_TO_HOST_CASE(132,133)
DEVICE_TO_HOST_CASE(133,134)
DEVICE_TO_HOST_CASE(134,135)
DEVICE_TO_HOST_CASE(135,136)
DEVICE_TO_HOST_CASE(136,137)
DEVICE_TO_HOST_CASE(137,138)
DEVICE_TO_HOST_CASE(138,139)
DEVICE_TO_HOST_CASE(139,140)
DEVICE_TO_HOST_CASE(140,141)
DEVICE_TO_HOST_CASE(141,142)
DEVICE_TO_HOST_CASE(142,143)
DEVICE_TO_HOST_CASE(143,144)
DEVICE_TO_HOST_CASE(144,145)
DEVICE_TO_HOST_CASE(145,146)
DEVICE_TO_HOST_CASE(146,147)
DEVICE_TO_HOST_CASE(147,148)
DEVICE_TO_HOST_CASE(148,149)
DEVICE_TO_HOST_CASE(149,150)
DEVICE_TO_HOST_CASE(150,151)
DEVICE_TO_HOST_CASE(151,152)
DEVICE_TO_HOST_CASE(152,153)
DEVICE_TO_HOST_CASE(153,154)
DEVICE_TO_HOST_CASE(154,155)
DEVICE_TO_HOST_CASE(155,156)
DEVICE_TO_HOST_CASE(156,157)
DEVICE_TO_HOST_CASE(157,158)
DEVICE_TO_HOST_CASE(158,159)
DEVICE_TO_HOST_CASE(159,160)
DEVICE_TO_HOST_CASE(160,161)
DEVICE_TO_HOST_CASE(161,162)
DEVICE_TO_HOST_CASE(162,163)
DEVICE_TO_HOST_CASE(163,164)
DEVICE_TO_HOST_CASE(164,165)
DEVICE_TO_HOST_CASE(165,166)
DEVICE_TO_HOST_CASE(166,167)
DEVICE_TO_HOST_CASE(167,168)
DEVICE_TO_HOST_CASE(168,169)
DEVICE_TO_HOST_CASE(169,170)
DEVICE_TO_HOST_CASE(170,171)
DEVICE_TO_HOST_CASE(171,172)
DEVICE_TO_HOST_CASE(172,173)
DEVICE_TO_HOST_CASE(173,174)
DEVICE_TO_HOST_CASE(174,175)
DEVICE_TO_HOST_CASE(175,176)
DEVICE_TO_HOST_CASE(176,177)
DEVICE_TO_HOST_CASE(177,178)
DEVICE_TO_HOST_CASE(178,179)
DEVICE_TO_HOST_CASE(179,180)
DEVICE_TO_HOST_CASE(180,181)
DEVICE_TO_HOST_CASE(181,182)
DEVICE_TO_HOST_CASE(182,183)
DEVICE_TO_HOST_CASE(183,184)
DEVICE_TO_HOST_CASE(184,185)
DEVICE_TO_HOST_CASE(185,186)
DEVICE_TO_HOST_CASE(186,187)
DEVICE_TO_HOST_CASE(187,188)
DEVICE_TO_HOST_CASE(188,189)
DEVICE_TO_HOST_CASE(189,190)
DEVICE_TO_HOST_CASE(190,191)
DEVICE_TO_HOST_CASE(191,192)
DEVICE_TO_HOST_CASE(192,193)
DEVICE_TO_HOST_CASE(193,194)
DEVICE_TO_HOST_CASE(194,195)
DEVICE_TO_HOST_CASE(195,196)
DEVICE_TO_HOST_CASE(196,197)
DEVICE_TO_HOST_CASE(197,198)
DEVICE_TO_HOST_CASE(198,199)
DEVICE_TO_HOST_CASE(199,200)
DEVICE_TO_HOST_CASE(200,201)
DEVICE_TO_HOST_CASE(201,202)
DEVICE_TO_HOST_CASE(202,203)
DEVICE_TO_HOST_CASE(203,204)
DEVICE_TO_HOST_CASE(204,205)
DEVICE_TO_HOST_CASE(205,206)
DEVICE_TO_HOST_CASE(206,207)
DEVICE_TO_HOST_CASE(207,208)
DEVICE_TO_HOST_CASE(208,209)
DEVICE_TO_HOST_CASE(209,210)
DEVICE_TO_HOST_CASE(210,211)
DEVICE_TO_HOST_CASE(211,212)
DEVICE_TO_HOST_CASE(212,213)
DEVICE_TO_HOST_CASE(213,214)
DEVICE_TO_HOST_CASE(214,215)
DEVICE_TO_HOST_CASE(215,216)
DEVICE_TO_HOST_CASE(216,217)
DEVICE_TO_HOST_CASE(217,218)
DEVICE_TO_HOST_CASE(218,219)
DEVICE_TO_HOST_CASE(219,220)
DEVICE_TO_HOST_CASE(220,221)
DEVICE_TO_HOST_CASE(221,222)
DEVICE_TO_HOST_CASE(222,223)
DEVICE_TO_HOST_CASE(223,224)
DEVICE_TO_HOST_CASE(224,225)
DEVICE_TO_HOST_CASE(225,226)
DEVICE_TO_HOST_CASE(226,227)
DEVICE_TO_HOST_CASE(227,228)
DEVICE_TO_HOST_CASE(228,229)
DEVICE_TO_HOST_CASE(229,230)
DEVICE_TO_HOST_CASE(230,231)
DEVICE_TO_HOST_CASE(231,232)
DEVICE_TO_HOST_CASE(232,233)
DEVICE_TO_HOST_CASE(233,234)
DEVICE_TO_HOST_CASE(234,235)
DEVICE_TO_HOST_CASE(235,236)
DEVICE_TO_HOST_CASE(236,237)
DEVICE_TO_HOST_CASE(237,238)
DEVICE_TO_HOST_CASE(238,239)
DEVICE_TO_HOST_CASE(239,240)
DEVICE_TO_HOST_CASE(240,241)
DEVICE_TO_HOST_CASE(241,242)
DEVICE_TO_HOST_CASE(242,243)
DEVICE_TO_HOST_CASE(243,244)
DEVICE_TO_HOST_CASE(244,245)
DEVICE_TO_HOST_CASE(245,246)
DEVICE_TO_HOST_CASE(246,247)
DEVICE_TO_HOST_CASE(247,248)
DEVICE_TO_HOST_CASE(248,249)
DEVICE_TO_HOST_CASE(249,250)
DEVICE_TO_HOST_CASE(250,251)
DEVICE_TO_HOST_CASE(251,252)
DEVICE_TO_HOST_CASE(252,253)
DEVICE_TO_HOST_CASE(253,254)
DEVICE_TO_HOST_CASE(254,255)
DEVICE_TO_HOST_CASE(255,256)
DEVICE_TO_HOST_CASE(256,257)
DEVICE_TO_HOST_CASE(257,258)
DEVICE_TO_HOST_CASE(258,259)
DEVICE_TO_HOST_CASE(259,260)
DEVICE_TO_HOST_CASE(260,261)
DEVICE_TO_HOST_CASE(261,262)
DEVICE_TO_HOST_CASE(262,263)
DEVICE_TO_HOST_CASE(263,264)
DEVICE_TO_HOST_CASE(264,265)
DEVICE_TO_HOST_CASE(265,266)
DEVICE_TO_HOST_CASE(266,267)
DEVICE_TO_HOST_CASE(267,268)
DEVICE_TO_HOST_CASE(268,269)
DEVICE_TO_HOST_CASE(269,270)
DEVICE_TO_HOST_CASE(270,271)
DEVICE_TO_HOST_CASE(271,272)
DEVICE_TO_HOST_CASE(272,273)
DEVICE_TO_HOST_CASE(273,274)
DEVICE_TO_HOST_CASE(274,275)
DEVICE_TO_HOST_CASE(275,276)
DEVICE_TO_HOST_CASE(276,277)
DEVICE_TO_HOST_CASE(277,278)
DEVICE_TO_HOST_CASE(278,279)
DEVICE_TO_HOST_CASE(279,280)
DEVICE_TO_HOST_CASE(280,281)
DEVICE_TO_HOST_CASE(281,282)
DEVICE_TO_HOST_CASE(282,283)
DEVICE_TO_HOST_CASE(283,284)
DEVICE_TO_HOST_CASE(284,285)
DEVICE_TO_HOST_CASE(285,286)
DEVICE_TO_HOST_CASE(286,287)
DEVICE_TO_HOST_CASE(287,288)
DEVICE_TO_HOST_CASE(288,289)
DEVICE_TO_HOST_CASE(289,290)
DEVICE_TO_HOST_CASE(290,291)
DEVICE_TO_HOST_CASE(291,292)
DEVICE_TO_HOST_CASE(292,293)
DEVICE_TO_HOST_CASE(293,294)
DEVICE_TO_HOST_CASE(294,295)
DEVICE_TO_HOST_CASE(295,296)
DEVICE_TO_HOST_CASE(296,297)
DEVICE_TO_HOST_CASE(297,298)
DEVICE_TO_HOST_CASE(298,299)
DEVICE_TO_HOST_CASE(299,300)
DEVICE_TO_HOST_CASE(300,301)
DEVICE_TO_HOST_CASE(301,302)
DEVICE_TO_HOST_CASE(302,303)
DEVICE_TO_HOST_CASE(303,304)
DEVICE_TO_HOST_CASE(304,305)
DEVICE_TO_HOST_CASE(305,306)
DEVICE_TO_HOST_CASE(306,307)
DEVICE_TO_HOST_CASE(307,308)
DEVICE_TO_HOST_CASE(308,309)
DEVICE_TO_HOST_CASE(309,310)
DEVICE_TO_HOST_CASE(310,311)
DEVICE_TO_HOST_CASE(311,312)
DEVICE_TO_HOST_CASE(312,313)
DEVICE_TO_HOST_CASE(313,314)
DEVICE_TO_HOST_CASE(314,315)
DEVICE_TO_HOST_CASE(315,316)
DEVICE_TO_HOST_CASE(316,317)
DEVICE_TO_HOST_CASE(317,318)
DEVICE_TO_HOST_CASE(318,319)
DEVICE_TO_HOST_CASE(319,320)
DEVICE_TO_HOST_CASE(320,321)
DEVICE_TO_HOST_CASE(321,322)
DEVICE_TO_HOST_CASE(322,323)
DEVICE_TO_HOST_CASE(323,324)
DEVICE_TO_HOST_CASE(324,325)
DEVICE_TO_HOST_CASE(325,326)
DEVICE_TO_HOST_CASE(326,327)
DEVICE_TO_HOST_CASE(327,328)
DEVICE_TO_HOST_CASE(328,329)
DEVICE_TO_HOST_CASE(329,330)
DEVICE_TO_HOST_CASE(330,331)
DEVICE_TO_HOST_CASE(331,332)
DEVICE_TO_HOST_CASE(332,333)
DEVICE_TO_HOST_CASE(333,334)
DEVICE_TO_HOST_CASE(334,335)
DEVICE_TO_HOST_CASE(335,336)
DEVICE_TO_HOST_CASE(336,337)
DEVICE_TO_HOST_CASE(337,338)
DEVICE_TO_HOST_CASE(338,339)
DEVICE_TO_HOST_CASE(339,340)
DEVICE_TO_HOST_CASE(340,341)
DEVICE_TO_HOST_CASE(341,342)
DEVICE_TO_HOST_CASE(342,343)
DEVICE_TO_HOST_CASE(343,344)
DEVICE_TO_HOST_CASE(344,345)
DEVICE_TO_HOST_CASE(345,346)
DEVICE_TO_HOST_CASE(346,347)
DEVICE_TO_HOST_CASE(347,348)
DEVICE_TO_HOST_CASE(348,349)
DEVICE_TO_HOST_CASE(349,350)
DEVICE_TO_HOST_CASE(350,351)
DEVICE_TO_HOST_CASE(351,352)
DEVICE_TO_HOST_CASE(352,353)
DEVICE_TO_HOST_CASE(353,354)
DEVICE_TO_HOST_CASE(354,355)
DEVICE_TO_HOST_CASE(355,356)
DEVICE_TO_HOST_CASE(356,357)
DEVICE_TO_HOST_CASE(357,358)
DEVICE_TO_HOST_CASE(358,359)
DEVICE_TO_HOST_CASE(359,360)
DEVICE_TO_HOST_CASE(360,361)
DEVICE_TO_HOST_CASE(361,362)
DEVICE_TO_HOST_CASE(362,363)
DEVICE_TO_HOST_CASE(363,364)
DEVICE_TO_HOST_CASE(364,365)
DEVICE_TO_HOST_CASE(365,366)
DEVICE_TO_HOST_CASE(366,367)
DEVICE_TO_HOST_CASE(367,368)
DEVICE_TO_HOST_CASE(368,369)
DEVICE_TO_HOST_CASE(369,370)
DEVICE_TO_HOST_CASE(370,371)
DEVICE_TO_HOST_CASE(371,372)
DEVICE_TO_HOST_CASE(372,373)
DEVICE_TO_HOST_CASE(373,374)
DEVICE_TO_HOST_CASE(374,375)
DEVICE_TO_HOST_CASE(375,376)
DEVICE_TO_HOST_CASE(376,377)
DEVICE_TO_HOST_CASE(377,378)
DEVICE_TO_HOST_CASE(378,379)
DEVICE_TO_HOST_CASE(379,380)
DEVICE_TO_HOST_CASE(380,381)
DEVICE_TO_HOST_CASE(381,382)
DEVICE_TO_HOST_CASE(382,383)
DEVICE_TO_HOST_CASE(383,384)
DEVICE_TO_HOST_CASE(384,385)
DEVICE_TO_HOST_CASE(385,386)
DEVICE_TO_HOST_CASE(386,387)
DEVICE_TO_HOST_CASE(387,388)
DEVICE_TO_HOST_CASE(388,389)
DEVICE_TO_HOST_CASE(389,390)
DEVICE_TO_HOST_CASE(390,391)
DEVICE_TO_HOST_CASE(391,392)
DEVICE_TO_HOST_CASE(392,393)
DEVICE_TO_HOST_CASE(393,394)
DEVICE_TO_HOST_CASE(394,395)
DEVICE_TO_HOST_CASE(395,396)
DEVICE_TO_HOST_CASE(396,397)
DEVICE_TO_HOST_CASE(397,398)
DEVICE_TO_HOST_CASE(398,399)
DEVICE_TO_HOST_CASE(399,400)
DEVICE_TO_HOST_CASE(400,401)
DEVICE_TO_HOST_CASE(401,402)
DEVICE_TO_HOST_CASE(402,403)
DEVICE_TO_HOST_CASE(403,404)
DEVICE_TO_HOST_CASE(404,405)
DEVICE_TO_HOST_CASE(405,406)
DEVICE_TO_HOST_CASE(406,407)
DEVICE_TO_HOST_CASE(407,408)
DEVICE_TO_HOST_CASE(408,409)
DEVICE_TO_HOST_CASE(409,410)
DEVICE_TO_HOST_CASE(410,411)
DEVICE_TO_HOST_CASE(411,412)
DEVICE_TO_HOST_CASE(412,413)
DEVICE_TO_HOST_CASE(413,414)
DEVICE_TO_HOST_CASE(414,415)
DEVICE_TO_HOST_CASE(415,416)
DEVICE_TO_HOST_CASE(416,417)
DEVICE_TO_HOST_CASE(417,418)
DEVICE_TO_HOST_CASE(418,419)
DEVICE_TO_HOST_CASE(419,420)
DEVICE_TO_HOST_CASE(420,421)
DEVICE_TO_HOST_CASE(421,422)
DEVICE_TO_HOST_CASE(422,423)
DEVICE_TO_HOST_CASE(423,424)
DEVICE_TO_HOST_CASE(424,425)
DEVICE_TO_HOST_CASE(425,426)
DEVICE_TO_HOST_CASE(426,427)
DEVICE_TO_HOST_CASE(427,428)
DEVICE_TO_HOST_CASE(428,429)
DEVICE_TO_HOST_CASE(429,430)
DEVICE_TO_HOST_CASE(430,431)
DEVICE_TO_HOST_CASE(431,432)
DEVICE_TO_HOST_CASE(432,433)
DEVICE_TO_HOST_CASE(433,434)
DEVICE_TO_HOST_CASE(434,435)
DEVICE_TO_HOST_CASE(435,436)
DEVICE_TO_HOST_CASE(436,437)
DEVICE_TO_HOST_CASE(437,438)
DEVICE_TO_HOST_CASE(438,439)
DEVICE_TO_HOST_CASE(439,440)
DEVICE_TO_HOST_CASE(440,441)
DEVICE_TO_HOST_CASE(441,442)
DEVICE_TO_HOST_CASE(442,443)
DEVICE_TO_HOST_CASE(443,444)
DEVICE_TO_HOST_CASE(444,445)
DEVICE_TO_HOST_CASE(445,446)
DEVICE_TO_HOST_CASE(446,447)
DEVICE_TO_HOST_CASE(447,448)
DEVICE_TO_HOST_CASE(448,449)
DEVICE_TO_HOST_CASE(449,450)
DEVICE_TO_HOST_CASE(450,451)
DEVICE_TO_HOST_CASE(451,452)
DEVICE_TO_HOST_CASE(452,453)
DEVICE_TO_HOST_CASE(453,454)
DEVICE_TO_HOST_CASE(454,455)
DEVICE_TO_HOST_CASE(455,456)
DEVICE_TO_HOST_CASE(456,457)
DEVICE_TO_HOST_CASE(457,458)
DEVICE_TO_HOST_CASE(458,459)
DEVICE_TO_HOST_CASE(459,460)
DEVICE_TO_HOST_CASE(460,461)
DEVICE_TO_HOST_CASE(461,462)
DEVICE_TO_HOST_CASE(462,463)
DEVICE_TO_HOST_CASE(463,464)
DEVICE_TO_HOST_CASE(464,465)
DEVICE_TO_HOST_CASE(465,466)
DEVICE_TO_HOST_CASE(466,467)
DEVICE_TO_HOST_CASE(467,468)
DEVICE_TO_HOST_CASE(468,469)
DEVICE_TO_HOST_CASE(469,470)
DEVICE_TO_HOST_CASE(470,471)
DEVICE_TO_HOST_CASE(471,472)
DEVICE_TO_HOST_CASE(472,473)
DEVICE_TO_HOST_CASE(473,474)
DEVICE_TO_HOST_CASE(474,475)
DEVICE_TO_HOST_CASE(475,476)
DEVICE_TO_HOST_CASE(476,477)
DEVICE_TO_HOST_CASE(477,478)
DEVICE_TO_HOST_CASE(478,479)
DEVICE_TO_HOST_CASE(479,480)
DEVICE_TO_HOST_CASE(480,481)
DEVICE_TO_HOST_CASE(481,482)
DEVICE_TO_HOST_CASE(482,483)
DEVICE_TO_HOST_CASE(483,484)
DEVICE_TO_HOST_CASE(484,485)
DEVICE_TO_HOST_CASE(485,486)
DEVICE_TO_HOST_CASE(486,487)
DEVICE_TO_HOST_CASE(487,488)
DEVICE_TO_HOST_CASE(488,489)
DEVICE_TO_HOST_CASE(489,490)
DEVICE_TO_HOST_CASE(490,491)
DEVICE_TO_HOST_CASE(491,492)
DEVICE_TO_HOST_CASE(492,493)
DEVICE_TO_HOST_CASE(493,494)
DEVICE_TO_HOST_CASE(494,495)
DEVICE_TO_HOST_CASE(495,496)
DEVICE_TO_HOST_CASE(496,497)
DEVICE_TO_HOST_CASE(497,498)
DEVICE_TO_HOST_CASE(498,499)
DEVICE_TO_HOST_CASE(499,500)
DEVICE_TO_HOST_CASE(500,501)
DEVICE_TO_HOST_CASE(501,502)
DEVICE_TO_HOST_CASE(502,503)
DEVICE_TO_HOST_CASE(503,504)
DEVICE_TO_HOST_CASE(504,505)
DEVICE_TO_HOST_CASE(505,506)
DEVICE_TO_HOST_CASE(506,507)
DEVICE_TO_HOST_CASE(507,508)
DEVICE_TO_HOST_CASE(508,509)
DEVICE_TO_HOST_CASE(509,510)
DEVICE_TO_HOST_CASE(510,511)
DEVICE_TO_HOST_CASE(511,512)
DEVICE_TO_HOST_CASE(512,513)
DEVICE_TO_HOST_CASE(513,514)
DEVICE_TO_HOST_CASE(514,515)
DEVICE_TO_HOST_CASE(515,516)
DEVICE_TO_HOST_CASE(516,517)
DEVICE_TO_HOST_CASE(517,518)
DEVICE_TO_HOST_CASE(518,519)
DEVICE_TO_HOST_CASE(519,520)
DEVICE_TO_HOST_CASE(520,521)
DEVICE_TO_HOST_CASE(521,522)
DEVICE_TO_HOST_CASE(522,523)
DEVICE_TO_HOST_CASE(523,524)
DEVICE_TO_HOST_CASE(524,525)
DEVICE_TO_HOST_CASE(525,526)
DEVICE_TO_HOST_CASE(526,527)
DEVICE_TO_HOST_CASE(527,528)
DEVICE_TO_HOST_CASE(528,529)
DEVICE_TO_HOST_CASE(529,530)
DEVICE_TO_HOST_CASE(530,531)
DEVICE_TO_HOST_CASE(531,532)
DEVICE_TO_HOST_CASE(532,533)
DEVICE_TO_HOST_CASE(533,534)
DEVICE_TO_HOST_CASE(534,535)
DEVICE_TO_HOST_CASE(535,536)
DEVICE_TO_HOST_CASE(536,537)
DEVICE_TO_HOST_CASE(537,538)
DEVICE_TO_HOST_CASE(538,539)
DEVICE_TO_HOST_CASE(539,540)
DEVICE_TO_HOST_CASE(540,541)
DEVICE_TO_HOST_CASE(541,542)
DEVICE_TO_HOST_CASE(542,543)
DEVICE_TO_HOST_CASE(543,544)
DEVICE_TO_HOST_CASE(544,545)
DEVICE_TO_HOST_CASE(545,546)
DEVICE_TO_HOST_CASE(546,547)
DEVICE_TO_HOST_CASE(547,548)
DEVICE_TO_HOST_CASE(548,549)
DEVICE_TO_HOST_CASE(549,550)
DEVICE_TO_HOST_CASE(550,551)
DEVICE_TO_HOST_CASE(551,552)
DEVICE_TO_HOST_CASE(552,553)
DEVICE_TO_HOST_CASE(553,554)
DEVICE_TO_HOST_CASE(554,555)
DEVICE_TO_HOST_CASE(555,556)
DEVICE_TO_HOST_CASE(556,557)
DEVICE_TO_HOST_CASE(557,558)
DEVICE_TO_HOST_CASE(558,559)
DEVICE_TO_HOST_CASE(559,560)
DEVICE_TO_HOST_CASE(560,561)
DEVICE_TO_HOST_CASE(561,562)
DEVICE_TO_HOST_CASE(562,563)
DEVICE_TO_HOST_CASE(563,564)
DEVICE_TO_HOST_CASE(564,565)
DEVICE_TO_HOST_CASE(565,566)
DEVICE_TO_HOST_CASE(566,567)
DEVICE_TO_HOST_CASE(567,568)
DEVICE_TO_HOST_CASE(568,569)
DEVICE_TO_HOST_CASE(569,570)
DEVICE_TO_HOST_CASE(570,571)
DEVICE_TO_HOST_CASE(571,572)
DEVICE_TO_HOST_CASE(572,573)
DEVICE_TO_HOST_CASE(573,574)
DEVICE_TO_HOST_CASE(574,575)
DEVICE_TO_HOST_CASE(575,576)
DEVICE_TO_HOST_CASE(576,577)
DEVICE_TO_HOST_CASE(577,578)
DEVICE_TO_HOST_CASE(578,579)
DEVICE_TO_HOST_CASE(579,580)
DEVICE_TO_HOST_CASE(580,581)
DEVICE_TO_HOST_CASE(581,582)
DEVICE_TO_HOST_CASE(582,583)
DEVICE_TO_HOST_CASE(583,584)
DEVICE_TO_HOST_CASE(584,585)
DEVICE_TO_HOST_CASE(585,586)
DEVICE_TO_HOST_CASE(586,587)
DEVICE_TO_HOST_CASE(587,588)
DEVICE_TO_HOST_CASE(588,589)
DEVICE_TO_HOST_CASE(589,590)
DEVICE_TO_HOST_CASE(590,591)
DEVICE_TO_HOST_CASE(591,592)
DEVICE_TO_HOST_CASE(592,593)
DEVICE_TO_HOST_CASE(593,594)
DEVICE_TO_HOST_CASE(594,595)
DEVICE_TO_HOST_CASE(595,596)
DEVICE_TO_HOST_CASE(596,597)
DEVICE_TO_HOST_CASE(597,598)
DEVICE_TO_HOST_CASE(598,599)
DEVICE_TO_HOST_CASE(599,600)
DEVICE_TO_HOST_CASE(600,601)
DEVICE_TO_HOST_CASE(601,602)
DEVICE_TO_HOST_CASE(602,603)
DEVICE_TO_HOST_CASE(603,604)
DEVICE_TO_HOST_CASE(604,605)
DEVICE_TO_HOST_CASE(605,606)
DEVICE_TO_HOST_CASE(606,607)
DEVICE_TO_HOST_CASE(607,608)
DEVICE_TO_HOST_CASE(608,609)
DEVICE_TO_HOST_CASE(609,610)
DEVICE_TO_HOST_CASE(610,611)
DEVICE_TO_HOST_CASE(611,612)
DEVICE_TO_HOST_CASE(612,613)
DEVICE_TO_HOST_CASE(613,614)
DEVICE_TO_HOST_CASE(614,615)
DEVICE_TO_HOST_CASE(615,616)
DEVICE_TO_HOST_CASE(616,617)
DEVICE_TO_HOST_CASE(617,618)
DEVICE_TO_HOST_CASE(618,619)
DEVICE_TO_HOST_CASE(619,620)
DEVICE_TO_HOST_CASE(620,621)
DEVICE_TO_HOST_CASE(621,622)
DEVICE_TO_HOST_CASE(622,623)
DEVICE_TO_HOST_CASE(623,624)
DEVICE_TO_HOST_CASE(624,625)
DEVICE_TO_HOST_CASE(625,626)
DEVICE_TO_HOST_CASE(626,627)
DEVICE_TO_HOST_CASE(627,628)
DEVICE_TO_HOST_CASE(628,629)
DEVICE_TO_HOST_CASE(629,630)
DEVICE_TO_HOST_CASE(630,631)
DEVICE_TO_HOST_CASE(631,632)
DEVICE_TO_HOST_CASE(632,633)
DEVICE_TO_HOST_CASE(633,634)
DEVICE_TO_HOST_CASE(634,635)
DEVICE_TO_HOST_CASE(635,636)
DEVICE_TO_HOST_CASE(636,637)
DEVICE_TO_HOST_CASE(637,638)
DEVICE_TO_HOST_CASE(638,639)
DEVICE_TO_HOST_CASE(639,640)
DEVICE_TO_HOST_CASE(640,641)
DEVICE_TO_HOST_CASE(641,642)
DEVICE_TO_HOST_CASE(642,643)
DEVICE_TO_HOST_CASE(643,644)
DEVICE_TO_HOST_CASE(644,645)
DEVICE_TO_HOST_CASE(645,646)
DEVICE_TO_HOST_CASE(646,647)
DEVICE_TO_HOST_CASE(647,648)
DEVICE_TO_HOST_CASE(648,649)
DEVICE_TO_HOST_CASE(649,650)
DEVICE_TO_HOST_CASE(650,651)
DEVICE_TO_HOST_CASE(651,652)
DEVICE_TO_HOST_CASE(652,653)
DEVICE_TO_HOST_CASE(653,654)
DEVICE_TO_HOST_CASE(654,655)
DEVICE_TO_HOST_CASE(655,656)
DEVICE_TO_HOST_CASE(656,657)
DEVICE_TO_HOST_CASE(657,658)
DEVICE_TO_HOST_CASE(658,659)
DEVICE_TO_HOST_CASE(659,660)
DEVICE_TO_HOST_CASE(660,661)
DEVICE_TO_HOST_CASE(661,662)
DEVICE_TO_HOST_CASE(662,663)
DEVICE_TO_HOST_CASE(663,664)
DEVICE_TO_HOST_CASE(664,665)
DEVICE_TO_HOST_CASE(665,666)
DEVICE_TO_HOST_CASE(666,667)
DEVICE_TO_HOST_CASE(667,668)
DEVICE_TO_HOST_CASE(668,669)
DEVICE_TO_HOST_CASE(669,670)
DEVICE_TO_HOST_CASE(670,671)
DEVICE_TO_HOST_CASE(671,672)
DEVICE_TO_HOST_CASE(672,673)
DEVICE_TO_HOST_CASE(673,674)
DEVICE_TO_HOST_CASE(674,675)
DEVICE_TO_HOST_CASE(675,676)
DEVICE_TO_HOST_CASE(676,677)
DEVICE_TO_HOST_CASE(677,678)
DEVICE_TO_HOST_CASE(678,679)
DEVICE_TO_HOST_CASE(679,680)
DEVICE_TO_HOST_CASE(680,681)
DEVICE_TO_HOST_CASE(681,682)
DEVICE_TO_HOST_CASE(682,683)
DEVICE_TO_HOST_CASE(683,684)
DEVICE_TO_HOST_CASE(684,685)
DEVICE_TO_HOST_CASE(685,686)
DEVICE_TO_HOST_CASE(686,687)
DEVICE_TO_HOST_CASE(687,688)
DEVICE_TO_HOST_CASE(688,689)
DEVICE_TO_HOST_CASE(689,690)
DEVICE_TO_HOST_CASE(690,691)
DEVICE_TO_HOST_CASE(691,692)
DEVICE_TO_HOST_CASE(692,693)
DEVICE_TO_HOST_CASE(693,694)
DEVICE_TO_HOST_CASE(694,695)
DEVICE_TO_HOST_CASE(695,696)
DEVICE_TO_HOST_CASE(696,697)
DEVICE_TO_HOST_CASE(697,698)
DEVICE_TO_HOST_CASE(698,699)
DEVICE_TO_HOST_CASE(699,700)
DEVICE_TO_HOST_CASE(700,701)
DEVICE_TO_HOST_CASE(701,702)
DEVICE_TO_HOST_CASE(702,703)
DEVICE_TO_HOST_CASE(703,704)
DEVICE_TO_HOST_CASE(704,705)
DEVICE_TO_HOST_CASE(705,706)
DEVICE_TO_HOST_CASE(706,707)
DEVICE_TO_HOST_CASE(707,708)
DEVICE_TO_HOST_CASE(708,709)
DEVICE_TO_HOST_CASE(709,710)
DEVICE_TO_HOST_CASE(710,711)
DEVICE_TO_HOST_CASE(711,712)
DEVICE_TO_HOST_CASE(712,713)
DEVICE_TO_HOST_CASE(713,714)
DEVICE_TO_HOST_CASE(714,715)
DEVICE_TO_HOST_CASE(715,716)
DEVICE_TO_HOST_CASE(716,717)
DEVICE_TO_HOST_CASE(717,718)
DEVICE_TO_HOST_CASE(718,719)
DEVICE_TO_HOST_CASE(719,720)
DEVICE_TO_HOST_CASE(720,721)
DEVICE_TO_HOST_CASE(721,722)
DEVICE_TO_HOST_CASE(722,723)
DEVICE_TO_HOST_CASE(723,724)
DEVICE_TO_HOST_CASE(724,725)
DEVICE_TO_HOST_CASE(725,726)
DEVICE_TO_HOST_CASE(726,727)
DEVICE_TO_HOST_CASE(727,728)
DEVICE_TO_HOST_CASE(728,729)
DEVICE_TO_HOST_CASE(729,730)
DEVICE_TO_HOST_CASE(730,731)
DEVICE_TO_HOST_CASE(731,732)
DEVICE_TO_HOST_CASE(732,733)
DEVICE_TO_HOST_CASE(733,734)
DEVICE_TO_HOST_CASE(734,735)
DEVICE_TO_HOST_CASE(735,736)
DEVICE_TO_HOST_CASE(736,737)
DEVICE_TO_HOST_CASE(737,738)
DEVICE_TO_HOST_CASE(738,739)
DEVICE_TO_HOST_CASE(739,740)
DEVICE_TO_HOST_CASE(740,741)
DEVICE_TO_HOST_CASE(741,742)
DEVICE_TO_HOST_CASE(742,743)
DEVICE_TO_HOST_CASE(743,744)
DEVICE_TO_HOST_CASE(744,745)
DEVICE_TO_HOST_CASE(745,746)
DEVICE_TO_HOST_CASE(746,747)
DEVICE_TO_HOST_CASE(747,748)
DEVICE_TO_HOST_CASE(748,749)
DEVICE_TO_HOST_CASE(749,750)
DEVICE_TO_HOST_CASE(750,751)
DEVICE_TO_HOST_CASE(751,752)
DEVICE_TO_HOST_CASE(752,753)
DEVICE_TO_HOST_CASE(753,754)
DEVICE_TO_HOST_CASE(754,755)
DEVICE_TO_HOST_CASE(755,756)
DEVICE_TO_HOST_CASE(756,757)
DEVICE_TO_HOST_CASE(757,758)
DEVICE_TO_HOST_CASE(758,759)
DEVICE_TO_HOST_CASE(759,760)
DEVICE_TO_HOST_CASE(760,761)
DEVICE_TO_HOST_CASE(761,762)
DEVICE_TO_HOST_CASE(762,763)
DEVICE_TO_HOST_CASE(763,764)
DEVICE_TO_HOST_CASE(764,765)
DEVICE_TO_HOST_CASE(765,766)
DEVICE_TO_HOST_CASE(766,767)
DEVICE_TO_HOST_CASE(767,768)
DEVICE_TO_HOST_CASE(768,769)
DEVICE_TO_HOST_CASE(769,770)
DEVICE_TO_HOST_CASE(770,771)
DEVICE_TO_HOST_CASE(771,772)
DEVICE_TO_HOST_CASE(772,773)
DEVICE_TO_HOST_CASE(773,774)
DEVICE_TO_HOST_CASE(774,775)
DEVICE_TO_HOST_CASE(775,776)
DEVICE_TO_HOST_CASE(776,777)
DEVICE_TO_HOST_CASE(777,778)
DEVICE_TO_HOST_CASE(778,779)
DEVICE_TO_HOST_CASE(779,780)
DEVICE_TO_HOST_CASE(780,781)
DEVICE_TO_HOST_CASE(781,782)
DEVICE_TO_HOST_CASE(782,783)
DEVICE_TO_HOST_CASE(783,784)
DEVICE_TO_HOST_CASE(784,785)
DEVICE_TO_HOST_CASE(785,786)
DEVICE_TO_HOST_CASE(786,787)
DEVICE_TO_HOST_CASE(787,788)
DEVICE_TO_HOST_CASE(788,789)
DEVICE_TO_HOST_CASE(789,790)
DEVICE_TO_HOST_CASE(790,791)
DEVICE_TO_HOST_CASE(791,792)
DEVICE_TO_HOST_CASE(792,793)
DEVICE_TO_HOST_CASE(793,794)
DEVICE_TO_HOST_CASE(794,795)
DEVICE_TO_HOST_CASE(795,796)
DEVICE_TO_HOST_CASE(796,797)
DEVICE_TO_HOST_CASE(797,798)
DEVICE_TO_HOST_CASE(798,799)
DEVICE_TO_HOST_CASE(799,800)
DEVICE_TO_HOST_CASE(800,801)
DEVICE_TO_HOST_CASE(801,802)
DEVICE_TO_HOST_CASE(802,803)
DEVICE_TO_HOST_CASE(803,804)
DEVICE_TO_HOST_CASE(804,805)
DEVICE_TO_HOST_CASE(805,806)
DEVICE_TO_HOST_CASE(806,807)
DEVICE_TO_HOST_CASE(807,808)
DEVICE_TO_HOST_CASE(808,809)
DEVICE_TO_HOST_CASE(809,810)
DEVICE_TO_HOST_CASE(810,811)
DEVICE_TO_HOST_CASE(811,812)
DEVICE_TO_HOST_CASE(812,813)
DEVICE_TO_HOST_CASE(813,814)
DEVICE_TO_HOST_CASE(814,815)
DEVICE_TO_HOST_CASE(815,816)
DEVICE_TO_HOST_CASE(816,817)
DEVICE_TO_HOST_CASE(817,818)
DEVICE_TO_HOST_CASE(818,819)
DEVICE_TO_HOST_CASE(819,820)
DEVICE_TO_HOST_CASE(820,821)
DEVICE_TO_HOST_CASE(821,822)
DEVICE_TO_HOST_CASE(822,823)
DEVICE_TO_HOST_CASE(823,824)
DEVICE_TO_HOST_CASE(824,825)
DEVICE_TO_HOST_CASE(825,826)
DEVICE_TO_HOST_CASE(826,827)
DEVICE_TO_HOST_CASE(827,828)
DEVICE_TO_HOST_CASE(828,829)
DEVICE_TO_HOST_CASE(829,830)
DEVICE_TO_HOST_CASE(830,831)
DEVICE_TO_HOST_CASE(831,832)
DEVICE_TO_HOST_CASE(832,833)
DEVICE_TO_HOST_CASE(833,834)
DEVICE_TO_HOST_CASE(834,835)
DEVICE_TO_HOST_CASE(835,836)
DEVICE_TO_HOST_CASE(836,837)
DEVICE_TO_HOST_CASE(837,838)
DEVICE_TO_HOST_CASE(838,839)
DEVICE_TO_HOST_CASE(839,840)
DEVICE_TO_HOST_CASE(840,841)
DEVICE_TO_HOST_CASE(841,842)
DEVICE_TO_HOST_CASE(842,843)
DEVICE_TO_HOST_CASE(843,844)
DEVICE_TO_HOST_CASE(844,845)
DEVICE_TO_HOST_CASE(845,846)
DEVICE_TO_HOST_CASE(846,847)
DEVICE_TO_HOST_CASE(847,848)
DEVICE_TO_HOST_CASE(848,849)
DEVICE_TO_HOST_CASE(849,850)
DEVICE_TO_HOST_CASE(850,851)
DEVICE_TO_HOST_CASE(851,852)
DEVICE_TO_HOST_CASE(852,853)
DEVICE_TO_HOST_CASE(853,854)
DEVICE_TO_HOST_CASE(854,855)
DEVICE_TO_HOST_CASE(855,856)
DEVICE_TO_HOST_CASE(856,857)
DEVICE_TO_HOST_CASE(857,858)
DEVICE_TO_HOST_CASE(858,859)
DEVICE_TO_HOST_CASE(859,860)
DEVICE_TO_HOST_CASE(860,861)
DEVICE_TO_HOST_CASE(861,862)
DEVICE_TO_HOST_CASE(862,863)
DEVICE_TO_HOST_CASE(863,864)
DEVICE_TO_HOST_CASE(864,865)
DEVICE_TO_HOST_CASE(865,866)
DEVICE_TO_HOST_CASE(866,867)
DEVICE_TO_HOST_CASE(867,868)
DEVICE_TO_HOST_CASE(868,869)
DEVICE_TO_HOST_CASE(869,870)
DEVICE_TO_HOST_CASE(870,871)
DEVICE_TO_HOST_CASE(871,872)
DEVICE_TO_HOST_CASE(872,873)
DEVICE_TO_HOST_CASE(873,874)
DEVICE_TO_HOST_CASE(874,875)
DEVICE_TO_HOST_CASE(875,876)
DEVICE_TO_HOST_CASE(876,877)
DEVICE_TO_HOST_CASE(877,878)
DEVICE_TO_HOST_CASE(878,879)
DEVICE_TO_HOST_CASE(879,880)
DEVICE_TO_HOST_CASE(880,881)
DEVICE_TO_HOST_CASE(881,882)
DEVICE_TO_HOST_CASE(882,883)
DEVICE_TO_HOST_CASE(883,884)
DEVICE_TO_HOST_CASE(884,885)
DEVICE_TO_HOST_CASE(885,886)
DEVICE_TO_HOST_CASE(886,887)
DEVICE_TO_HOST_CASE(887,888)
DEVICE_TO_HOST_CASE(888,889)
DEVICE_TO_HOST_CASE(889,890)
DEVICE_TO_HOST_CASE(890,891)
DEVICE_TO_HOST_CASE(891,892)
DEVICE_TO_HOST_CASE(892,893)
DEVICE_TO_HOST_CASE(893,894)
DEVICE_TO_HOST_CASE(894,895)
DEVICE_TO_HOST_CASE(895,896)
DEVICE_TO_HOST_CASE(896,897)
DEVICE_TO_HOST_CASE(897,898)
DEVICE_TO_HOST_CASE(898,899)
DEVICE_TO_HOST_CASE(899,900)
DEVICE_TO_HOST_CASE(900,901)
DEVICE_TO_HOST_CASE(901,902)
DEVICE_TO_HOST_CASE(902,903)
DEVICE_TO_HOST_CASE(903,904)
DEVICE_TO_HOST_CASE(904,905)
DEVICE_TO_HOST_CASE(905,906)
DEVICE_TO_HOST_CASE(906,907)
DEVICE_TO_HOST_CASE(907,908)
DEVICE_TO_HOST_CASE(908,909)
DEVICE_TO_HOST_CASE(909,910)
DEVICE_TO_HOST_CASE(910,911)
DEVICE_TO_HOST_CASE(911,912)
DEVICE_TO_HOST_CASE(912,913)
DEVICE_TO_HOST_CASE(913,914)
DEVICE_TO_HOST_CASE(914,915)
DEVICE_TO_HOST_CASE(915,916)
DEVICE_TO_HOST_CASE(916,917)
DEVICE_TO_HOST_CASE(917,918)
DEVICE_TO_HOST_CASE(918,919)
DEVICE_TO_HOST_CASE(919,920)
DEVICE_TO_HOST_CASE(920,921)
DEVICE_TO_HOST_CASE(921,922)
DEVICE_TO_HOST_CASE(922,923)
DEVICE_TO_HOST_CASE(923,924)
DEVICE_TO_HOST_CASE(924,925)
DEVICE_TO_HOST_CASE(925,926)
DEVICE_TO_HOST_CASE(926,927)
DEVICE_TO_HOST_CASE(927,928)
DEVICE_TO_HOST_CASE(928,929)
DEVICE_TO_HOST_CASE(929,930)
DEVICE_TO_HOST_CASE(930,931)
DEVICE_TO_HOST_CASE(931,932)
DEVICE_TO_HOST_CASE(932,933)
DEVICE_TO_HOST_CASE(933,934)
DEVICE_TO_HOST_CASE(934,935)
DEVICE_TO_HOST_CASE(935,936)
DEVICE_TO_HOST_CASE(936,937)
DEVICE_TO_HOST_CASE(937,938)
DEVICE_TO_HOST_CASE(938,939)
DEVICE_TO_HOST_CASE(939,940)
DEVICE_TO_HOST_CASE(940,941)
DEVICE_TO_HOST_CASE(941,942)
DEVICE_TO_HOST_CASE(942,943)
DEVICE_TO_HOST_CASE(943,944)
DEVICE_TO_HOST_CASE(944,945)
DEVICE_TO_HOST_CASE(945,946)
DEVICE_TO_HOST_CASE(946,947)
DEVICE_TO_HOST_CASE(947,948)
DEVICE_TO_HOST_CASE(948,949)
DEVICE_TO_HOST_CASE(949,950)
DEVICE_TO_HOST_CASE(950,951)
DEVICE_TO_HOST_CASE(951,952)
DEVICE_TO_HOST_CASE(952,953)
DEVICE_TO_HOST_CASE(953,954)
DEVICE_TO_HOST_CASE(954,955)
DEVICE_TO_HOST_CASE(955,956)
DEVICE_TO_HOST_CASE(956,957)
DEVICE_TO_HOST_CASE(957,958)
DEVICE_TO_HOST_CASE(958,959)
DEVICE_TO_HOST_CASE(959,960)
DEVICE_TO_HOST_CASE(960,961)
DEVICE_TO_HOST_CASE(961,962)
DEVICE_TO_HOST_CASE(962,963)
DEVICE_TO_HOST_CASE(963,964)
DEVICE_TO_HOST_CASE(964,965)
DEVICE_TO_HOST_CASE(965,966)
DEVICE_TO_HOST_CASE(966,967)
DEVICE_TO_HOST_CASE(967,968)
DEVICE_TO_HOST_CASE(968,969)
DEVICE_TO_HOST_CASE(969,970)
DEVICE_TO_HOST_CASE(970,971)
DEVICE_TO_HOST_CASE(971,972)
DEVICE_TO_HOST_CASE(972,973)
DEVICE_TO_HOST_CASE(973,974)
DEVICE_TO_HOST_CASE(974,975)
DEVICE_TO_HOST_CASE(975,976)
DEVICE_TO_HOST_CASE(976,977)
DEVICE_TO_HOST_CASE(977,978)
DEVICE_TO_HOST_CASE(978,979)
DEVICE_TO_HOST_CASE(979,980)
DEVICE_TO_HOST_CASE(980,981)
DEVICE_TO_HOST_CASE(981,982)
DEVICE_TO_HOST_CASE(982,983)
DEVICE_TO_HOST_CASE(983,984)
DEVICE_TO_HOST_CASE(984,985)
DEVICE_TO_HOST_CASE(985,986)
DEVICE_TO_HOST_CASE(986,987)
DEVICE_TO_HOST_CASE(987,988)
DEVICE_TO_HOST_CASE(988,989)
DEVICE_TO_HOST_CASE(989,990)
DEVICE_TO_HOST_CASE(990,991)
DEVICE_TO_HOST_CASE(991,992)
DEVICE_TO_HOST_CASE(992,993)
DEVICE_TO_HOST_CASE(993,994)
DEVICE_TO_HOST_CASE(994,995)
DEVICE_TO_HOST_CASE(995,996)
DEVICE_TO_HOST_CASE(996,997)
DEVICE_TO_HOST_CASE(997,998)
DEVICE_TO_HOST_CASE(998,999)
DEVICE_TO_HOST_CASE(999,1000)*/
    }

    delete[] global_mem_ptr;
    
  }
  std::cout<<"total_time " << total_time << std::endl;
  std::cout << "Number of embeddings found "<< output_1.size () + output_2.size () + output_3.size () + output_4.size () + output_5.size () + output_6.size () + output_7.size () + output_8.size ()<< std::endl;
  
  /*for (auto embedding : output) {
    print_embedding (embedding, std::cout);
    std::cout << std::endl;
  }*/
}
