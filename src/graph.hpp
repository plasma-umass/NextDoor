#include <vector>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <map>

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#define LINE_SIZE 1024*1024

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
  void update_edges (std::unordered_map <int, int>& prev_to_new_ids) 
  {
    for (size_t i = 0; i < edges.size (); i++) {
      edges[i] = prev_to_new_ids[edges[i]];
    }

    sort_edges ();
  }

  void remove_duplicate_edges () 
  {
    std::unordered_set <int> set_edges = std::unordered_set<int> (edges.begin(), edges.end ());
    edges = std::vector<int> (set_edges.begin (), set_edges.end ());
    //sort_edges ();
  }

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
  Graph ()
  {}
  enum {
    EdgeList,
    AdjacenyList,
  } GraphFileType;
  
  void load_from_edge_list (FILE* fp, bool weighted) 
  {
    assert (fp != nullptr);
    n_edges = 0;
    std::map<size_t, std::vector<std::pair<size_t, float>>> vertex_to_edges;
    size_t max_vertex = 0;
    while (true) {
      char line[LINE_SIZE];

      if (fgets (line, LINE_SIZE, fp) == nullptr) {
        break;
      }

      size_t src, dst;
      float weight;

      size_t bytes_read = sscanf (line, "%ld %ld %f\n", &src, &dst, &weight);
      assert (bytes_read > 0);

      if (vertex_to_edges.find(src) == vertex_to_edges.end()) {
        vertex_to_edges[src] = std::vector<std::pair<size_t, float>>();
      }

      vertex_to_edges[src].push_back(std::make_pair(dst, weight));
      if (vertex_to_edges.find(dst) == vertex_to_edges.end()) {
        vertex_to_edges.emplace(dst, std::vector<std::pair<size_t, float>>());
      }

      max_vertex = max(src, max_vertex);
      max_vertex = max(dst, max_vertex);
      n_edges++;
    }    

    for (size_t i = 0; i <= max_vertex; i++) {
      //std::cout << "running for " << i << std::endl;
      if (vertex_to_edges.find(i) == vertex_to_edges.end()) {
        vertex_to_edges.emplace(i, std::vector<std::pair<size_t, float>>());
      }
    }
        
    for (auto it : vertex_to_edges) {
      size_t v = it.first;
      vertices.push_back(Vertex(v, v));
      for (auto e : it.second) {
        size_t dst = std::get<0>(e);
        if (!(dst < vertex_to_edges.size())) {
          printf (" dst %d vertex_to_edges.size() %d\n", dst, vertex_to_edges.size());
        }
        //assert (dst < vertex_to_edges.size());
        vertices[vertices.size()-1].add_edge(dst);
      }
    }
    //Sort vertices by number of edges
    // std::sort (vertices.begin (), vertices.end (), Vertex::compare_vertex);

    // std::unordered_map <int, int> previous_id_to_new_ids;
    // for (int i = 0; i < vertices.size (); i++) {
    //   previous_id_to_new_ids[vertices[i].get_id ()] = i;
    //   vertices[i].set_id (i);
    // }

    // for (int i = 0; i < vertices.size (); i++) {
    //   vertices[i].update_edges (previous_id_to_new_ids);
    // }
  }

  void load_from_adjacency_list (FILE* fp) 
  {
    assert (fp != nullptr);
    n_edges = 0;

    while (true) {
      char line[LINE_SIZE];

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

      //vertex.remove_duplicate_edges ();

      vertex.sort_edges ();
      vertices.push_back (vertex);
    }

    //Sort vertices by number of edges
    // std::sort (vertices.begin (), vertices.end (), Vertex::compare_vertex);

    // std::unordered_map <int, int> previous_id_to_new_ids;
    // for (int i = 0; i < vertices.size (); i++) {
    //   previous_id_to_new_ids[vertices[i].get_id ()] = i;
    //   vertices[i].set_id (i);
    // }

    // for (int i = 0; i < vertices.size (); i++) {
    //   vertices[i].update_edges (previous_id_to_new_ids);
    // }
  }

  const std::vector<Vertex>& get_vertices () {return vertices;}
  int get_n_edges () {return n_edges;}

  void print (std::ostream& os) 
  {
    for (auto v : vertices) {
      v.print (os);
    }
  }
};
#endif