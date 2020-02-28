#include "graph.hpp"

#ifndef __CSR_HPP__
#define __CST_HPP__

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

  __host__ __device__ int get_start_edge_idx () {return start_edge_id;}
  __host__ __device__ int get_end_edge_idx () {return end_edge_id;}
  __host__ __device__ VertexID get_id () {return id;}
  __host__ __device__ void set_start_edge_id (int start) {start_edge_id = start;}
  __host__ __device__ void set_end_edge_id (int end) {end_edge_id = end;}
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
  const int first_vertex_id;
  const int last_vertex_id;
  const int first_edge_idx;
  const int last_edge_idx;
  const CSR::Vertex *vertices;
  const CSR::Edge *edges;

   __host__
  CSRPartition (int _start, int _end, int _edge_start_idx, int _edge_end_idx, const CSR::Vertex* _vertices, const CSR::Edge* _edges) : 
                first_vertex_id (_start), last_vertex_id(_end), vertices(_vertices), edges(_edges), 
                first_edge_idx(_edge_start_idx), last_edge_idx (_edge_end_idx)
  {
    
  }

  __host__ __device__
  int get_start_edge_idx (int vertex_id) {
    if (!(vertex_id <= last_vertex_id && first_vertex_id <= vertex_id)) {
      printf ("vertex_id %d, end_vertex %d, start_vertex %d\n", vertex_id, last_vertex_id, first_vertex_id);
      assert (false);
    }
    return vertices[vertex_id - first_vertex_id].start_edge_id;
  }

  __host__ __device__
  int get_end_edge_idx (int vertex_id)
  {
    assert (vertex_id <= last_vertex_id && first_vertex_id <= vertex_id);
    return vertices[vertex_id - first_vertex_id].end_edge_id;
  }
  
  __host__ __device__
  CSR::Edge get_edge (int idx) 
  {
    assert (idx >= first_edge_idx && idx <= last_edge_idx);
    return edges[idx - first_edge_idx];
  }

  int get_vertex_for_edge_idx (int idx)
  {
    for (int v = first_vertex_id; v < last_vertex_id; v++) {
      if (idx >= get_start_edge_idx (v) && idx <= get_end_edge_idx (v)) {
        return v;
      }
    }

    return -1;
  }

  __host__ __device__
  const CSR::Edge* get_edges () 
  {
    return edges;
  }

  __host__ __device__
  const CSR::Vertex* get_vertices () 
  {
    return vertices;
  }

  __host__ __device__
  size_t get_n_vertices ()
  {
    return last_vertex_id - first_vertex_id + 1;
  }

  __host__ __device__
  size_t get_n_edges ()
  {
    return last_edge_idx - first_edge_idx + 1;
  }

  __host__ __device__ 
  bool is_vertex_in_partition (int v) 
  {
    return v >= first_vertex_id && v <= last_vertex_id;
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

#endif