#include "graph.hpp"

#ifndef __CSR_HPP__
#define __CSR_HPP__

typedef int32_t VertexID;
typedef int32_t EdgePos_t;
typedef VertexID VertexID_t;

class VertexRange 
{
  public:
    class vertex_iterator 
    {
    private:
      VertexID v;

    public: 
      vertex_iterator(VertexID _v) : v(_v) {}
      vertex_iterator operator++() {v++; return *this;}
      vertex_iterator operator--() {v--; return *this;}
      VertexID operator*() {return v;}

      bool operator==(const vertex_iterator& rhs) {return v == rhs.v;}
      bool operator!=(const vertex_iterator& rhs) {return v != rhs.v;}
    };

  private:
    VertexID first;
    VertexID last;

  public:
    VertexRange(VertexID _first, VertexID _last) : first(_first), last(_last) {}

    vertex_iterator begin() const
    {
      return vertex_iterator(first);
    }

    vertex_iterator end() const
    {
      return vertex_iterator(last+1);
    }
};

class CSR
{
public:
  struct Vertex
  {
    VertexID id;
    EdgePos_t start_edge_id;
    EdgePos_t end_edge_id;
    float max_weight;
    __host__ __device__
    Vertex ()
    {
      id = -1;
      start_edge_id = -1;
      end_edge_id = -1;
      max_weight = -1;
    }

    void set_from_graph_vertex (::Vertex& vertex)
    {
      id = vertex.get_id ();
    }

    __host__ __device__ EdgePos_t get_start_edge_idx () const {return start_edge_id;}
    __host__ __device__ EdgePos_t get_end_edge_idx () const {return end_edge_id;}
    __host__ __device__ float get_max_weight() {return max_weight;}
    __host__ __device__ VertexID get_id () {return id;}
    __host__ __device__ void set_start_edge_id (EdgePos_t start) {start_edge_id = start;}
    __host__ __device__ void set_end_edge_id (EdgePos_t end) {end_edge_id = end;}
    __host__ __device__ void set_max_weight(float w) {max_weight = w;}
    __host__ __device__ EdgePos_t num_edges() const {
      return (get_end_edge_idx () != -1) ? (get_end_edge_idx () - get_start_edge_idx () + 1) : 0;
    }
  };

  typedef VertexID Edge;

  CSR::Vertex* vertices;
  CSR::Edge* edges;
  float* weights;
  int n_vertices;
  EdgePos_t n_edges;

public:
  CSR (int _n_vertices, EdgePos_t _n_edges)
  {
    n_vertices = _n_vertices;
    n_edges = _n_edges;
    //TODO: Can we allocate it in pinned memory?
    edges = new CSR::Edge[n_edges];
    weights = new float[n_edges];
    vertices = new CSR::Vertex[n_vertices];
  }

  void print (std::ostream& os) const 
  {
    for (int i = 0; i < n_vertices; i++) {
      os << vertices[i].id << " ";
      for (EdgePos_t edge_iter = vertices[i].start_edge_id;
           edge_iter <= vertices[i].end_edge_id; edge_iter++) {
        os << edges[edge_iter] << " " << weights[edge_iter] << " ";
      }
      os << std::endl;
    }
  }

  __host__ __device__
  EdgePos_t get_start_edge_idx (VertexID vertex_id) const 
  {
    if (!(vertex_id < n_vertices && 0 <= vertex_id)) {
      printf ("vertex_id %d, n_vertices %d\n", vertex_id, n_vertices);
      assert (false);
    }
    return vertices[vertex_id].start_edge_id;
  }

  __host__ __device__
  EdgePos_t get_end_edge_idx (VertexID vertex_id) const 
  {
    if (!(vertex_id < n_vertices && 0 <= vertex_id)) {
      printf ("vertex_id %d\n", vertex_id);
    }
    assert (vertex_id < n_vertices && 0 <= vertex_id);
    return vertices[vertex_id].end_edge_id;
  }

  __host__ __device__
  bool has_edge (VertexID u, VertexID v) const 
  {
    //TODO: Since graph is sorted, do this using binary search
    for (EdgePos_t e = get_start_edge_idx (u); e <= get_end_edge_idx (u); e++) {
      if (edges[e] == v) {
        return true;
      }
    }

    return false;
  }

  EdgePos_t n_edges_for_vertex (VertexID vertex) const 
  {
    return (get_end_edge_idx (vertex) != -1) ? (get_end_edge_idx (vertex) - get_start_edge_idx (vertex) + 1) : 0;
  }
  __host__ __device__
  const CSR::Edge* get_edges () const  {return &edges[0];}
  __host__ __device__
  const float* get_weights () const 
  {
    return &weights[0];
  }

  __host__ __device__
  const CSR::Vertex* get_vertices () const  {return &vertices[0];}

  __host__ __device__
  VertexID get_n_vertices () const  {return n_vertices;}

  __host__ __device__
  void copy_vertices (CSR* src, int start, int end)
  {
    for (int i = start; i < end; i++) {
      vertices[i] = src->get_vertices()[i];
    }
  }

  __host__ __device__
  void copy_edges (CSR* src, EdgePos_t start, EdgePos_t end)
  {
    for (EdgePos_t i = start; i < end; i++) {
      edges[i] = src->get_edges ()[i];
    }
  }

  __host__ __device__
  int get_n_edges () const  {return n_edges;}

  VertexRange iterate_vertices() const 
  {
    return VertexRange(0, get_n_vertices()-1);
  }

  VertexID get_invalid_vertex() const 
  {
    return get_n_vertices()+1;
  }
  
  __host__
  bool has_vertex(VertexID v) const {
    return 0 <= v && v < n_vertices;
  }
};

class CSRPartition
{
public:
  ~CSRPartition() {}
  const VertexID first_vertex_id;
  const VertexID last_vertex_id;
  const EdgePos_t first_edge_idx;
  const EdgePos_t last_edge_idx;
  const CSR::Vertex *vertices;
  const CSR::Edge *edges;
  const float* weights;

   __host__
  CSRPartition (int _start, int _end, EdgePos_t _edge_start_idx, EdgePos_t _edge_end_idx, const CSR::Vertex* _vertices, const CSR::Edge* _edges,
                const float* _weights) : 
                first_vertex_id (_start), last_vertex_id(_end), vertices(_vertices), edges(_edges), 
                first_edge_idx(_edge_start_idx), last_edge_idx (_edge_end_idx), weights(_weights)
  {
    
  }

  __host__ __device__ __forceinline__
  EdgePos_t get_start_edge_idx (int vertex_id) const  {
    if (!(vertex_id <= last_vertex_id && first_vertex_id <= vertex_id)) {
      printf ("vertex_id %d, end_vertex %d, start_vertex %d\n", vertex_id, last_vertex_id, first_vertex_id);
      assert (false);
    }
    return vertices[vertex_id - first_vertex_id].start_edge_id;
  }

  __host__ __device__ __forceinline__
  EdgePos_t get_end_edge_idx (int vertex_id) const 
  {
    assert (vertex_id <= last_vertex_id && first_vertex_id <= vertex_id);
    return vertices[vertex_id - first_vertex_id].end_edge_id;
  }

  __host__ __device__ __forceinline__
  float get_max_weight (int vertex_id) const 
  {
    assert (vertex_id <= last_vertex_id && first_vertex_id <= vertex_id);
    return vertices[vertex_id - first_vertex_id].max_weight;
  }
  
  __host__ __device__ __forceinline__
  CSR::Edge get_edge (EdgePos_t idx)  const 
  {
    assert (idx >= first_edge_idx && idx <= last_edge_idx);
    return edges[idx - first_edge_idx];
  }

  __host__ __device__ __forceinline__
  float get_weight(EdgePos_t idx) const 
  {
    assert (idx >= first_edge_idx && idx <= last_edge_idx);
    return weights[idx - first_edge_idx];
  }

  VertexID get_vertex_for_edge_idx (EdgePos_t idx) const 
  {
    for (int v = first_vertex_id; v < last_vertex_id; v++) {
      if (idx >= get_start_edge_idx (v) && idx <= get_end_edge_idx (v)) {
        return v;
      }
    }

    return -1;
  }

  VertexRange get_vertex_range() const 
  {
    return VertexRange(first_vertex_id, last_vertex_id);
  }

  VertexRange iterate_num_vertices() const 
  {
    return VertexRange(0, get_n_vertices()-1);
  }

  __host__ __device__ __forceinline__ EdgePos_t get_n_edges_for_vertex (VertexID v) const 
  {
    return (get_end_edge_idx (v) != -1) ? (get_end_edge_idx(v) - get_start_edge_idx (v) + 1) : 0;
  }

  __host__ __device__ __forceinline__
  const CSR::Edge* get_edges () const 
  {
    return edges;
  }

  __host__ __device__ __forceinline__
  const float* get_weights () const 
  {
    return weights;
  }

  __host__ __device__ __forceinline__
  const CSR::Edge* get_edges (VertexID v) const 
  {
    return &edges[get_start_edge_idx(v) - first_edge_idx];
  }

  __host__ __device__ __forceinline__
  const float* get_weights (VertexID v) const 
  {
    return &weights[get_start_edge_idx(v) - first_edge_idx];
  }

  __host__ __device__ __forceinline__
  const CSR::Vertex* get_vertices () const 
  {
    return vertices; 
  }

  __host__ __device__ __forceinline__
  VertexID get_n_vertices () const 
  {
    return last_vertex_id - first_vertex_id + 1;
  }

  __host__ __device__ __forceinline__
  EdgePos_t get_n_edges () const 
  {
    return last_edge_idx - first_edge_idx + 1;
  }

  __host__ __device__  __forceinline__
  bool has_vertex (int v) const 
  {
    return v >= first_vertex_id && v <= last_vertex_id;
  }

  __host__ __device__ __forceinline__
  bool has_edge (VertexID src, VertexID dst) const 
  {
    for (EdgePos_t i = get_start_edge_idx(src); i <= get_end_edge_idx(src); i++) {
      if (get_edge(i) == dst) {
        return true;
      }
    }

    return false;
  }

  __host__ __device__ __forceinline__
  bool has_edge_logn (VertexID src, VertexID dst) const 
  {
    EdgePos_t l = get_start_edge_idx(src);
    EdgePos_t r = get_end_edge_idx(src);
    
    while (l <= r) {
      EdgePos_t m = l + (r-l)/2;
      if (get_edge(m) == dst)
        return true;
      
      if (get_edge(m) < dst)
        l = m + 1;
      else
        r = m - 1;
    }
    return false;
  }

  __host__ __device__ __forceinline__
  VertexID get_vertex_idx(VertexID v) const
  {
    assert (has_vertex (v));
    return v - first_vertex_id;
  }

  __host__ __device__ __forceinline__
  VertexID get_invalid_vertex() const 
  {
#ifdef ENABLE_GRAPH_PARTITION_FOR_GLOBAL_MEM
    //Implement this
    assert(false);
#endif
    return get_n_vertices()+1;
  }

  __device__ __host__ 
  const CSR::Vertex* get_vertex(VertexID id) const {return &vertices[id];} 
  // static struct HasVertex {
  //   bool operator () (CSRPartition& partition, const VertexID& v) const {
  //     return (partition.first_vertex_id >= v && v <= partition.last_vertex_id);
  //   }
  // }
};

struct GPUCSRPartition
{
  CSRPartition* device_csr;
  CSR::Vertex* device_vertex_array;
  CSR::Edge* device_edge_array;
  float* device_weights_array;
};

#ifdef USE_CONSTANT_MEM
  __constant__ unsigned char csr_constant_buff[sizeof(CSR)];
#endif

void csr_from_graph (CSR* csr, Graph& graph)
{
  EdgePos_t edge_iterator = 0;
  auto graph_vertices = graph.get_vertices ();

  for (size_t i = 0; i < graph_vertices.size (); i++) {
    ::Vertex& vertex = graph_vertices[i];
    csr->vertices[i].set_from_graph_vertex (graph_vertices[i]);
    csr->vertices[i].set_start_edge_id (edge_iterator);
    csr->vertices[i].set_max_weight(vertex.max_weight());
    for (auto edge : vertex.get_edges ()) {
      csr->edges[edge_iterator] = edge.first;
      csr->weights[edge_iterator] = edge.second;
      edge_iterator++;
    }

    if (vertex.get_edges().size() == 0)
      csr->vertices[i].set_end_edge_id (-1);
    else
      csr->vertices[i].set_end_edge_id (edge_iterator-1);

    // if (i == 45241) {
    //   printf("start %d end %d\n", csr->vertices[i].get_start_edge_idx(), csr->vertices[i].get_end_edge_idx());
    // }
  }
}

#endif