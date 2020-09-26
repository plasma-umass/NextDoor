#include "csr.hpp"

#ifndef __SAMPLE_HPP__
#define __SAMPLE_HPP__

class Sample
{
  private:
    int numVertices_;
    VertexID_t startingVertex_;
    VertexID_t* vertices_;
  
  public:
    Sample(int numVertices, VertexID_t startingVertex) : numVertices_(numVertices), startingVertex_(startingVertex) {}
    Sample(VertexID_t startingVertex) : numVertices_(1), startingVertex_(startingVertex) {}

    
};

#endif