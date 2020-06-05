#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <assert.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <tuple>
#include <iterator>

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

  Vertex ():label(-1),id(-1){}

  void set_id (int _id) {id = _id;}
  int get_id () {return id;}
  int get_label () {return label;}
  int set_label (int l) {label = l;}
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

  void load_from_edge_list_binary (std::string file_path, bool weighted) 
  {
    FILE* fp = fopen (file_path.c_str(), "rb");
    if (fp == nullptr) {
      std::cout << "File '" << file_path << "' not found" << std::endl;
      return;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    size_t max_vertex = 0;
    char *string = new char[fsize + 1];
    fread(string, 1, fsize, fp);
    std::cout << "graph string loaded " << std::endl;

    n_edges = 0;

    // std::unordered_set<int> vertices_with_degree;
    
    for (size_t s = 0; s < fsize; s += 12) {
      int src = *(int*)(string+s);
      int dst = *(int*)(string+s+4);
      float weight = *(float*)(string+s+8);

      if (src > vertices.size()) {
        vertices.resize(src+1);
        // for (size_t i = sz; i <= src; i++) {
        //   vertices.push_back(Vertex(i, i));
        // }
      }

      if (dst > vertices.size()) {
        vertices.resize(dst+1);
        // for (size_t i = sz; i <= src; i++) {
        //   vertices.push_back(Vertex(i, i));
        // }
      }

      vertices[src].add_edge(dst);

      n_edges++;
    }

    delete string;
    fclose(fp);

    for (auto v : vertices) {
      v.sort_edges();
    }
  }

  
  void load_from_edge_list_txt (FILE* fp, bool weighted) 
  {
    assert (fp != nullptr);
    n_edges = 0;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);  /* same as rewind(f); */
    size_t max_vertex = 0;
    std::vector<std::tuple<size_t, size_t>> all_edges;

    for (int part = 0; part < 1; part++) {
      char *string = new char[fsize/1 + 1];
      fread(string, 1, fsize/1, fp);
      std::cout << "graph string loaded " << std::endl;
      std::string ss = std::string(string)+"\n";
      std::stringstream iss(ss);
      delete string;

      while (true) {
        size_t src = 0, dst = 0;
        float weight = 0;
        if (iss.eof())
          break;
        
        iss >> src;
        if (iss.eof())
          break;
        iss >> dst;
        if (iss.eof())
          break;
        iss >> weight;
        if (weight == 0.0) {
          break;
        }
        //all_edges.push_back (std::make_tuple(src, dst));

        if (src > vertices.size()) {
          vertices.resize(src+1);
          // for (size_t i = sz; i <= src; i++) {
          //   vertices.push_back(Vertex(i, i));
          // }
        }
        if (dst > vertices.size()) {
          vertices.resize(dst+1);
          // for (size_t i = sz; i <= src; i++) {
          //   vertices.push_back(Vertex(i, i));
          // }
        }
        vertices[src].add_edge(dst);
        max_vertex = max(src, max_vertex);
        max_vertex = max(dst, max_vertex);
        n_edges++;
      }
    }

    std::cout << "adj list created " << std::endl;

    // for (size_t i = 0; i <= max_vertex; i++) {
    //   //std::cout << "running for " << i << std::endl;
    //   if (vertex_to_edges.find(i) == vertex_to_edges.end()) {
    //     vertex_to_edges.emplace(i, std::vector<std::pair<size_t, float>>());
    //   }
    // }

    // vertices = std::vector<Vertex>(max_vertex+1);
    // for (size_t v = 0; v <= max_vertex; v++) {
    //   vertices[v].set_id(v);
    //   vertices[v].set_label(v);
    // }

    // for (auto edge : all_edges) {
    //   size_t src = std::get<0>(edge);
    //   size_t dst = std::get<1>(edge);
    //   vertices[src].add_edge(dst);
    // }

    std::cout << "graph created " << std::endl;
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