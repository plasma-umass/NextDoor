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
#include <set>
#include <tuple>
#include <iterator>
#include <utility>

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#define LINE_SIZE 1024*1024

class Vertex
{
private:
  int id;
  std::vector<std::pair<int, float>> edges;

public:
  Vertex (int _id) : id (_id)
  {
  }

  Vertex ():id(-1){}

  void set_id (int _id) {id = _id;}
  int get_id () {return id;}
  void add_edge (int vertexID, float weight) {edges.push_back (std::make_pair(vertexID, weight));}
  void sort_edges () {std::sort (edges.begin(), edges.end ());}
  void update_edges (std::unordered_map <int, int>& prev_to_new_ids) 
  {
    for (size_t i = 0; i < edges.size (); i++) {
      edges[i].first = prev_to_new_ids[edges[i].first];
    }

    sort_edges ();
  }

  void remove_duplicate_edges () 
  {
    std::set<std::pair<int,float>> set_edges = std::set<std::pair<int,float>> (edges.begin(), edges.end ());
    edges = std::vector<std::pair<int,float>> (set_edges.begin (), set_edges.end ());
    //sort_edges ();
  }

  std::vector <std::pair<int, float>>& get_edges () {return edges;}
  void print (std::ostream& os)
  {
    os << id << " " << " ";
    for (auto edge : edges) {
      os << edge.first << " " << edge.second << " ";
    }

    os << std::endl;
  }

  static bool compare_vertex (Vertex& v1, Vertex& v2) 
  {
    return v1.edges.size () < v2.edges.size ();
  }

  float max_weight()
  {
    float w = 0.0f;

    for (auto p : edges) {
      w = max(w, p.second);
    }

    return w;
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
    char *string = new char[fsize + 1];
    if (fread(string, 1, fsize, fp) != (size_t)fsize) {
      std::cout << "" << std::endl;
      abort();
    }
    std::cout << "Graph Binary Loaded" << std::endl;

    n_edges = 0;
    
    for (size_t s = 0; s < (size_t)fsize; s += 12) {
      int src = *(int*)(string+s);
      int dst = *(int*)(string+s+4);
      float weight = *(float*)(string+s+8);

      if ((size_t)src >= vertices.size()) {
        vertices.resize(src+1);
        // for (size_t i = sz; i <= src; i++) {
        //   vertices.push_back(Vertex(i, i));
        // }
      }

      if ((size_t)dst >= vertices.size()) {
        vertices.resize(dst+1);
        // for (size_t i = sz; i <= src; i++) {
        //   vertices.push_back(Vertex(i, i));
        // }
      }

      vertices[src].add_edge(dst, weight);

      n_edges++;
    }

    delete string;
    fclose(fp);
    printf("Vertices and Edges loaded\n");

    #pragma omp parallel for
    for (size_t v = 0; v < vertices.size(); v++) {
      vertices[v].sort_edges();
    }
    printf("Edges sorted\n");
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
      if (fread(string, 1, fsize/1, fp) != (size_t)fsize/1) {
        std::cout<<"Error reading at "<<__FILE__<<":"<<__LINE__<<std::endl;
        abort();
      }
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
        vertices[src].add_edge(dst, 0.0f);
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

  void load_from_adjacency_list(char* graph_file)
  {
    FILE* fp = fopen(graph_file, "r");
    if (fp == nullptr) {
      std::cout << "File '" << graph_file << "' cannot open" << std::endl;
      exit(EXIT_SUCCESS);
    }

    n_edges = 0;

    while (true) {
      char line[LINE_SIZE];

      if (fgets(line, LINE_SIZE, fp) == nullptr) {
        break;
      }

      int id;
      int vars_filled;

      vars_filled = sscanf(line, "%d", &id);
      Vertex vertex(id);
      char* _line = line + chars_in_int(id);
      do {
        int num;
        float weight;
        int chars_read = 0;

        vars_filled = sscanf(_line, "%d %f%n", &num, &weight, &chars_read);
        //printf("_line '%s' vars_filled %d chars_read %d\n", _line, vars_filled, chars_read);
        if (vars_filled == 2) {
          vertex.add_edge(num, weight);
          _line += chars_read;//chars_in_int(num);
          n_edges++;
        }

      } while (vars_filled == 2);

      //vertex.remove_duplicate_edges (); 

      vertex.sort_edges();
      vertices.push_back(vertex);
    }

    //Rename vertices so that each vertex is between [0, N]

    std::unordered_map<int, int> origIDToNewID;

    for (size_t i = 0; i < vertices.size(); i++) {
      origIDToNewID[vertices[i].get_id()] = i;
      vertices[i].set_id (i);
    }

    for (size_t i = 0; i < vertices.size (); i++) {
       vertices[i].update_edges (origIDToNewID);
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

    fclose(fp);
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