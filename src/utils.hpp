#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

double convertTimeValToDouble (struct timeval _time)
{
  return ((double)_time.tv_sec) + ((double)_time.tv_usec)/1000000.0f;
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

#endif