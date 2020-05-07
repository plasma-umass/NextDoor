#include <vector>
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>

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

  Graph (FILE* fp) 
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
    // for (int i = 0; i < vertices.size (); i++) {
    //   if (vertices[i].get_id() != i) {
    //     vertices[i].set_id(i);//vertices.insert (i, Vertex (i, 0));
    //   }
    // }
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