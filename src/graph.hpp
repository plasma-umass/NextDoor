#include <vector>
#include <algorithm>
#include <iostream>

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
    n_edges = 0;
    
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
  }

  const std::vector<Vertex>& get_vertices () {return vertices;}
  int get_n_edges () {return n_edges;}


};
#endif