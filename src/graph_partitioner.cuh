void create_csr_partitions (CSR* csr, std::vector<CSRPartition>& csr_partitions, const size_t effective_partition_size)
{
  std::vector<std::tuple<VertexID, VertexID, EdgePos_t, EdgePos_t>> vertex_partition_positions_vector;

  //Create Partitions.
  VertexID u = 0;
  EdgePos_t partition_edge_start_idx = 0; 

  while (u < csr->get_n_vertices ()) {
    EdgePos_t n_edges = 0;
    VertexID u_start = u;
    EdgePos_t end_edge_idx = 0;
    VertexID u_end = csr->get_n_vertices () - 1;
    EdgePos_t edges = 0;

    for (VertexID v = u; v < csr->get_n_vertices (); v++) {
      EdgePos_t start = csr->get_start_edge_idx (v);
      const EdgePos_t end = csr->get_end_edge_idx (v);
      if (end != -1) {
        if (v == u) {
          //std::cout << "1829: " << " partition_edge_start_idx " << partition_edge_start_idx << " u " << u << " start " << start << " end " << end << std::endl;
        }
        if (v == u && partition_edge_start_idx >= start) {
          start = partition_edge_start_idx;
        }
        edges = end - start + 1;
        assert (edges >= 0);
      } else {
        edges = 0;
      }
      if ((n_edges + edges) * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex) >= effective_partition_size) {
        end_edge_idx = (effective_partition_size - (v-u_start + 1)*sizeof(CSR::Vertex))/sizeof (CSR::Edge) - n_edges;
        //std::cout << " v " << v << " n_edges " << n_edges << " u " << u_start  << "  sizeof (CSR::Edge) " << sizeof (CSR::Edge) <<  " sizeof(CSR::Vertex) " << sizeof(CSR::Vertex) << " end_edge_idx " << end_edge_idx << " effective_partition_size " << effective_partition_size << " start " << start << " end " << end << std::endl;
        if (v == 4847570) {
          printf ("(n_edges + edges) * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex) %d effective_partition_size %d", (n_edges + edges) * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex), effective_partition_size);
        }
        if (end_edge_idx < edges) {
          u = v;
          u_end = v - 1;
          end_edge_idx = start - 1;
        } else if (end_edge_idx == edges) {
          u_end = v;
          u = v + 1;
          end_edge_idx += start - 1; //Including last edge
        } else {
          assert (false);
        }
        if (u_end == 4847570) {
          printf ("aaa (n_edges + edges) * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex) %d effective_partition_size %d", (n_edges + edges) * sizeof (CSR::Edge) + (v-u_start + 1)*sizeof(CSR::Vertex), effective_partition_size);
        }
        std::cout << "u_end : " << u_end << " u_start: "  << u_start  << std::endl;
        if (u_end < u_start) 
        {
          std::cout << "u_end : " << u_end << " u_start: "  << u_start  << std::endl;
          std::cout << "ERROR: Cannot create partition " << std::endl;
          assert (false);
        }

        break;
      }

      n_edges += edges;
    }

    if (u_end == csr->get_n_vertices() - 1) {
      vertex_partition_positions_vector.push_back (std::make_tuple (u_start, u_end, partition_edge_start_idx, csr->get_n_edges() - 1));
    } else {
      vertex_partition_positions_vector.push_back (std::make_tuple (u_start, u_end, partition_edge_start_idx, (end_edge_idx == 0) ? csr->get_end_edge_idx (u_end) : end_edge_idx));
    }
    if (u_end == 4847570) {
      std::cout << "for 4847570 end_edge_idx: " << end_edge_idx << std::endl;
    }
    //Vertex partition: [u_start, u_end]. Edge partition is all edges from u_start to u_end if end_edge_idx = 0. otherwise all edges of vertices from u_start to u_end - 1 and edges of u_end u_end.start_edge_idx to end_edge_idx.
    
    partition_edge_start_idx = end_edge_idx + 1;

    if (u_end == csr->get_n_vertices () - 1) {
      break;
    }
  }

  std::cout << __LINE__ << ": " << partition_edge_start_idx << " " << csr->get_n_edges () - 1 << std::endl;


  //Create remaining partitions if last vertex's edges are remaining
  if (partition_edge_start_idx != 1 && partition_edge_start_idx < csr->get_n_edges ()) {
    assert ((csr->get_n_edges () - partition_edge_start_idx) * sizeof (CSR::Edge) + (1)*sizeof(CSR::Vertex) <= effective_partition_size);
    vertex_partition_positions_vector.push_back (std::make_tuple (csr->get_n_vertices () - 1, csr->get_n_vertices () - 1, partition_edge_start_idx, csr->get_n_edges ()- 1));
  }

  //Create partitions
  for (auto p : vertex_partition_positions_vector) {
    VertexID u = std::get<0> (p);
    VertexID v = std::get<1> (p);
    EdgePos_t start = std::get<2> (p);
    EdgePos_t end = std::get<3> (p);

    CSR::Vertex* vertex_array = new CSR::Vertex[v - u + 1];
    memcpy (vertex_array, &csr->get_vertices ()[u], (v-u + 1)*sizeof(CSR::Vertex));
    vertex_array[0].set_start_edge_id (start);
    vertex_array[v-u].set_end_edge_id (end);

    CSR::Edge* edge_array = new CSR::Edge[end - start + 1];
    memcpy (edge_array, &csr->get_edges ()[start], (end - start + 1)*sizeof (CSR::Edge));
    CSRPartition part = CSRPartition (u, v, start, end, vertex_array, edge_array);
    csr_partitions.push_back (part);
  }

  /** Check if partitions created are correct**/
  //Sum of edges of all partitions is equal to N_EDGES
  EdgePos_t sum_partition_edges = 0;

  for (int id = 0; id < (int)csr_partitions.size (); id++) {
    auto part = csr_partitions[id];
    std::cout << id << " " << part.last_edge_idx << " " << part.first_edge_idx << " " << part.first_vertex_id << " " << part.last_vertex_id << std::endl;
    if (part.last_edge_idx != -1) {
      sum_partition_edges += part.last_edge_idx - part.first_edge_idx + 1;
    }
  }

  if (!(sum_partition_edges == csr->get_n_edges())) {
    std::cout << __LINE__ <<": "<<sum_partition_edges  << " " << csr->get_n_edges() << std::endl;
  }
  assert (sum_partition_edges == csr->get_n_edges());
 
  VertexID sum_vertices = 0;
  for (int p = 0; p < (int)csr_partitions.size (); p++) {
    if (p > 0 && csr_partitions[p].first_vertex_id == csr_partitions[p-1].last_vertex_id) {
      sum_vertices += csr_partitions[p].last_vertex_id - (csr_partitions[p].first_vertex_id);
    } else {
      sum_vertices += csr_partitions[p].last_vertex_id - csr_partitions[p].first_vertex_id + 1;
    }
  }

  assert (sum_vertices == csr->get_n_vertices());

  EdgePos_t equal_edges = 0;

  /*Check if union of all partitions is equal to the graph*/
  for (int p = 0; p < (int)csr_partitions.size (); p++) {
    VertexID u = csr_partitions[p].first_vertex_id;
    VertexID v = csr_partitions[p].last_vertex_id;
    EdgePos_t end = csr_partitions[p].last_edge_idx;
    EdgePos_t start = csr_partitions[p].first_edge_idx;
    for (VertexID vertex = u; vertex <= v; vertex++) {
      EdgePos_t _start = csr->get_start_edge_idx (vertex);
      if (p > 0 && vertex == csr_partitions[p-1].last_vertex_id) {
        _start = start;
      }
      EdgePos_t _end = csr->get_end_edge_idx (vertex);
      VertexID part_start = csr_partitions[p].get_start_edge_idx (vertex);
      VertexID part_end = csr_partitions[p].get_end_edge_idx (vertex);
      
      if (_end != -1 && part_end != -1) {
        while (_start <= _end && _start <= end && part_start <= part_end) {
          if (!(csr->get_edges ()[_start] == csr_partitions[p].get_edge (part_start))) {
            std::cout << "part_start " << part_start << " part_end " << 
            part_end << " _start " << _start << " _end " << _end << " vertex " 
            << vertex << std::endl;  
            abort ();
          }
          
          equal_edges++;
          part_start++;
          _start++;
        }
      }
    }
  }

  assert (equal_edges == csr->get_n_edges());  
  /********Checking DONE*******/
}