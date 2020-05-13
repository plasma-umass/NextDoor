#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iterator>
#include "csr.hpp"

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

namespace utils {

  __device__
  EdgePos_t atomicAdd (EdgePos_t* ptr, const EdgePos_t val)
  {
    if (sizeof(EdgePos_t) == 8) {
      return (EdgePos_t)::atomicAdd((unsigned long long*)ptr, (unsigned long long)val);
    } else {
      return (EdgePos_t)::atomicAdd((int*)ptr, (int)val);
    }
  }

  double convertTimeValToDouble (struct timeval _time)
  {
    return ((double)_time.tv_sec) + ((double)_time.tv_usec)/1000000.0f;
  }

  template<class T>
  void print_container(T const &s)
  {
      std::copy(s.begin(),
              s.end(),
              std::ostream_iterator<int>(std::cout, " "));
  }

  struct timeval getTimeOfDay ()
  {
    struct timeval _time;

    if (gettimeofday (&_time, NULL) == -1) {
      fprintf (stderr, "gettimeofday returned -1\n");
      perror ("");
      abort ();
    }

    return _time;
  }

  __device__ __host__ bool intervals_intersect (int x1, int x2, int y1, int y2)
  {
    return x1 <= y2 && y1 <= x2;
  }


  bool is_cuda_error (cudaError_t error) 
  {
    //cudaError_t error = cudaGetLastError ();
    if (error != cudaSuccess) {
      const char* error_string = cudaGetErrorString (error);
      std::cout << "Cuda Error: " << error_string << std::endl;
      return true;
    }

    return false;
  }

  void get_num_roots_distribution(int* roots_distribution, int* ranges, int num_ranges,
                                  EdgePos_t* src_num_roots, VertexRange src_range)
  {
    for (VertexID src : src_range) {
      EdgePos_t num_roots = src_num_roots[2*src + 1];
      if (num_roots <= ranges[0]) {
      } else {
        for (int i = 1; i < num_ranges - 1; i++) {
          if (num_roots >= ranges[i-1] and num_roots < ranges[i]) {
            roots_distribution[i] += num_roots;
            break;
          }
        }
        
        if (num_roots >= ranges[num_ranges]) {
          roots_distribution[num_ranges] += num_roots;
        }
      }
    }
  }

  template<class T1, class T2>
  inline T1 next_multiple(const T1 val, const T2 divisor)
  {
    if (val%divisor == 0) return val;
    return (val/divisor + 1)*divisor;
  }

  template<class T>
  inline T thread_block_size(const T total, const T tb_size)
  {
    if (total%tb_size == 0)
      return total/tb_size;
    return total/tb_size +1;
  }

  template<class T1, class T2, class T3>
  inline void set_till_next_multiple(T1& val, const T2 divisor, T3* mem, 
                                     const T3 init_val)
  {
    while (val%divisor != 0)
      mem[val++] = -1;
  }

  template<class T>
  class RangeIterator
  {
    public:
      class iterator 
      {
      private:
         T it;

      public: 
        iterator(T _it) : it(_it) {}
        iterator operator++() {it++; return *this;}
        iterator operator--() {it--; return *this;}
        T operator*() {return it;}

        bool operator==(const iterator& rhs) {return it == rhs.it;}
        bool operator!=(const iterator& rhs) {return it != rhs.it;}
      };

    private:
      T first;
      T last;

    public:
      RangeIterator(T _first, T _last) : first(_first), last(_last) {}

      iterator begin() const
      {
        return iterator(first);
      }

      iterator end() const
      {
        return iterator(last+1);
      }
  };

  template<class T>
  size_t sizeof_vector(const T& vec)
  {
    return vec.size()*sizeof(vec[0]);
  }

  #define CHK_CU(x) assert (utils::is_cuda_error (x) == false);
}

namespace LoadBalancing {
  enum LoadBalancingThreshold{
    GridLevel = 32,
    BlockLevel = 0,
    SubWarpLevel = 0,
  };

  enum LoadBalancingTBSizes {
    GridLevelTBSize = 32,
    BlockLevelTBSize = 32,
    SubWarpLevelTBSize = 32,
  };

  bool is_grid_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots >= LoadBalancingThreshold::GridLevel;
  }

  bool is_block_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots < LoadBalancingThreshold::GridLevel and num_roots >= LoadBalancingThreshold::BlockLevel;
  }

  bool is_subwarp_level_assignment(const EdgePos_t num_roots) 
  {
    return num_roots < LoadBalancingThreshold::SubWarpLevel;
  }

  void num_gpu_threads(const VertexRange src_range, const EdgePos_t* src_num_roots, 
                       EdgePos_t& num_grid_threads, EdgePos_t& num_block_threads, 
                       EdgePos_t& num_subwarp_threads)
  {
    num_grid_threads = 0;
    num_block_threads = 0;
    num_subwarp_threads = 0;
    for (VertexID src : src_range) {
      EdgePos_t num_roots = src_num_roots[2*src + 1];
      if (is_grid_level_assignment(num_roots)) {
        num_grid_threads += num_roots;
        num_grid_threads = utils::next_multiple(num_grid_threads, 
                                                GridLevelTBSize);
      } else if (is_block_level_assignment(num_roots)) {
        num_block_threads += num_roots;
        // num_block_threads = utils::next_multiple(num_block_threads, 
        //                                          BlockLevelTBSize);
      } else if (is_subwarp_level_assignment(num_roots)) {
        num_subwarp_threads += num_roots;
        // num_subwarp_threads = utils::next_multiple(num_subwarp_threads, 
        //                                            SubWarpLevelTBSize);
      }
    }
  }
};

#endif