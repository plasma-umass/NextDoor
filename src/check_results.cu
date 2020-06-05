bool check_result(CSR* csr, EdgePos_t*** final_map_vertex_to_additions, EdgePos_t** additions_sizes, VertexID** neighbors)
{
  //Check result by traversing all sampled neighbors and making
  //sure that if neighbors at kth-hop is an adjacent vertex of one
  //of the k-1th hop neighbors.

  //First create the adjacency matrix.

  std::unordered_map<VertexID, std::unordered_set<VertexID>> adj_matrix;

  for (VertexID v : csr->iterate_vertices()) {
    adj_matrix[v] = std::unordered_set<VertexID> ();
    for (EdgePos_t i = csr->get_start_edge_idx(v); 
         i <= csr->get_end_edge_idx(v); i++) {
      VertexID e = csr->get_edges()[i];
      adj_matrix[v].insert(e);
    }
  }

  //Now check the correctness
  for (VertexID vertex : csr->iterate_vertices()) {
    std::vector<VertexID> prev_hop_neighbors;

    for (int hop = 1; hop < size(); hop++) {
      EdgePos_t start_idx = final_map_vertex_to_additions[hop][0][2*vertex];
      EdgePos_t n_additions = additions_sizes[hop][2*vertex + 1];
      
      if (n_additions == 0) {
        EdgePos_t prev_start = final_map_vertex_to_additions[hop-1][0][2*vertex];
        for (EdgePos_t prev_neighbr_idx = prev_start;
          prev_neighbr_idx < prev_start + additions_sizes[hop-1][2*vertex + 1]; 
          prev_neighbr_idx++) {
            VertexID prev_neighbor = neighbors[hop-1][prev_neighbr_idx];
            if (!(adj_matrix[prev_neighbor].size() == 0)) {
              printf ("vertex %d prev_neighbor %d hop %d\n", vertex, prev_neighbor, hop);
            }
            assert (adj_matrix[prev_neighbor].size() == 0);
        }
      } else {
        for (EdgePos_t neighbr_idx = start_idx; neighbr_idx < start_idx + n_additions;
            neighbr_idx++) {
          VertexID neighbor = neighbors[hop][neighbr_idx];
          EdgePos_t prev_start = final_map_vertex_to_additions[hop-1][0][2*vertex];
          bool found = false;
          for (EdgePos_t prev_neighbr_idx = prev_start;
            prev_neighbr_idx < prev_start + additions_sizes[hop-1][2*vertex + 1]; 
            prev_neighbr_idx++) {
              VertexID prev_neighbor = neighbors[hop-1][prev_neighbr_idx];
              found = adj_matrix[prev_neighbor].count(neighbor) > 0;
              if (found)
                break;
          }

          if (!found) {
            std::cout << "Neighbor " << neighbor << " not found in " << hop << "-hop neighbors of " << vertex << " with sample set size " << n_additions << std::endl;
          }
          assert (found);
        }
      }
    }
  }

  return true;
}