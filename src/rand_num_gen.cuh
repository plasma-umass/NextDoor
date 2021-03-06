#ifndef __RAND_NUM_GEN_CUH__
#define __RAND_NUM_GEN_CUH__

class RandNumGen {
private:
  float* rand_nums;
  size_t num_vertices;
  size_t rand_nums_per_vertex;
  float* d_rand;
  RandNumGen *d_ptr;
  size_t rand_size;

  RandNumGen(size_t _rand_nums_per_vertex, 
             size_t _num_vertices, float* _d_rand) :
    rand_nums(nullptr), d_rand(_d_rand),
    rand_nums_per_vertex(_rand_nums_per_vertex),
    num_vertices(_num_vertices)
  {
    rand_size = _num_vertices*_rand_nums_per_vertex;
  }

public:
  __device__
  RandNumGen() {}

  __device__ inline
  void init(const RandNumGen& x)
  {
    rand_nums_per_vertex = x.rand_nums_per_vertex;
    num_vertices = x.num_vertices;
    d_rand = x.d_rand;
    rand_size = x.rand_size;
  }

  __device__ inline
  void init(size_t _rand_nums_per_vertex, size_t _num_vertices, float* _d_rand)
  {
    rand_nums_per_vertex = _rand_nums_per_vertex;
    num_vertices = _num_vertices;
    d_rand = _d_rand;
  }

  RandNumGen(EdgePos_t _rand_nums_per_vertex,
             size_t _num_vertices) :
    rand_nums(nullptr), d_rand(nullptr),
    rand_nums_per_vertex(_rand_nums_per_vertex),
    num_vertices(_num_vertices)
  {
    rand_size = _num_vertices*_rand_nums_per_vertex;
  }
  
  void gen_random_nums()
  {
    rand_nums = new float[rand_size];
    for (size_t n = 0; n < rand_size; n++) {
      float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      if (r == 1.0f) {
        r = 0.99f;
      }
#ifndef NDEBUG
      if (!(0.0f <= r && r < 1.0f))
        printf ("r %f\n", r);
      assert (0.0f <= r && r < 1.0f);
#endif
      rand_nums[n] = r;
    }
  }
  
  void free()
  {
    if (rand_nums) {
      delete rand_nums;
      rand_nums = nullptr;
    }
    
    if (d_rand) {
      CHK_CU(cudaFree(d_rand));
      d_rand = nullptr;
    }

    if (d_ptr) {
      CHK_CU(cudaFree(d_ptr));
      d_ptr = nullptr;
    }
  }

  ~RandNumGen()
  {
    
  }

  __device__ __forceinline__
  float rand_float(const VertexID vertex, const int n) const {
    size_t access = (vertex*rand_nums_per_vertex + n);
#ifndef NDEBUG
    if (!(access < rand_size)) {
      printf ("access %ld v %d n %d rand_size %ld\n", access, vertex, n, rand_size);
    }
    assert (access < rand_size);
#endif
    float f = d_rand[access];

#ifndef NDEBUG
    if (!(0 <= f && f <= 1.0)) {
      printf ("f %f not in range [0, 1]\n", f);
    }
    assert (0 <= f && f <= 1.0);
#endif

    return f;
  }

  __device__ __forceinline__
  double rand_double(const VertexID vertex, const EdgePos_t n) const {
    return (double)rand_float(vertex, n);
  }
  __device__ __forceinline__
  static EdgePos_t rand_int(curandState* state, const EdgePos_t n) {
    float ff = curand_uniform(state)*n;
    EdgePos_t id =  min((EdgePos_t)ff, n-1);//(EdgePos_t)round(ff) - 1.0f;
    return id;
  }

  __device__ __forceinline__
  EdgePos_t rand_neighbor(const VertexID vertex, const int n, 
                                 const EdgePos_t num_edges) const {
    constexpr int sz = sizeof(EdgePos_t);
    if(sz == 4) {
      float ff = 0.5f + rand_float(vertex, n)*num_edges;
      EdgePos_t id =  (EdgePos_t)round(ff) - 1.0f;
#ifndef NDEBUG
      if (!(0 <= id && id < num_edges))
      printf ("vertex %d num_edges %d id %d ff %f\n", vertex, num_edges, id, ff);
        assert(0 <= id && id < num_edges);
#endif
      return id;
    } else {
      assert (sz == 8);
      double ff = 0.5f + rand_double(vertex, n)*num_edges;
      EdgePos_t id =  (EdgePos_t)round(ff) - 1.0f;
#ifndef NDEBUG
      if (!(0 <= id && id < num_edges))
      printf ("vertex %d num_edges %ld id %ld ff %lf\n", vertex, (long)num_edges, (long)id, ff);
        assert(0 <= id && id < num_edges);
#endif
      return id;
    } 
  }

  // __device__ 
  // inline float rand_float(int thread_id) const {
  //   assert (thread_id < rand_size);
  //   return d_rand[thread_id];
  // }

  __host__ 
  inline float rand_float_cpu(EdgePos_t num) const {return rand_nums[num];}
  
  RandNumGen* to_device()
  {
    CHK_CU(cudaMalloc(&d_rand, rand_size*sizeof(float)));
    CHK_CU(cudaMemcpy(d_rand, rand_nums, rand_size*sizeof(float), cudaMemcpyHostToDevice));

    RandNumGen temp = RandNumGen(rand_nums_per_vertex, num_vertices, d_rand);
    std::cout << "Temp size " << temp.rand_size << std::endl;
    CHK_CU(cudaMalloc(&d_ptr, sizeof(RandNumGen)));
    CHK_CU(cudaMemcpy(d_ptr, &temp, sizeof(RandNumGen), cudaMemcpyHostToDevice));

    return d_ptr;
  }
};

#endif