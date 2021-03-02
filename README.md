# cuSampler
cuSampler accelerating graph sampling using GPUs.
cuSampler uses <it>transit</it> parallelism that improves memory access locality and decreases control flow divergence.
More details can be found in our EuroSys'21 paper. 


# Code Structure
* `apps` directory contains following applications. These applications are provided with their Python bindings to integrate these applications in existing GNNs:
  * `graphSAGE-2-Hops.cu` GraphSAGE [1] 2-Hop application
  * `fastgcnSampling.cu` FastGCN [2] Sampling application
  * `ladiesSampling.cu`  LADIES [3] Sampling 
  * `clusterGCNSampling.cu` ClusterGCN [4] Sampling

* `src` directory contains the source of cuSampler. It contains following files and directories
  * `apps` contains following applications:
    * `clusterGCNSampling.cu`  ClusterGCN [4] Sampling 
    * `importanceSampling.cu` FastGCN [2] and LADIES [3] Sampling 
    * `khop.cu` GraphSAGE [1] 2-Hop application
    * `multiRW.cu` Multi Dimension Random Walk [5] 
    * `mvsSampling.cu` Minimum Variance Sampling [6]
    * `randomWalks.cu` DeepWalk [7], PPR [8], and node2vec [9] random walks.
  * `tests` uses above applications as unit tests and provides a function to check results of these applications. 
  * `csr.hpp` and `graph.hpp` contains data structures for graphs.
  * `utils.hpp` contains utility functions.
  * `nextdoor.cu` contains cuSampler kernels and execution logic.
  * `libNextDoor.hpp` header file that contains functions to integrate cuSampler's execution logic in an external program.
  * `nextDoorModule.cu` contains implementation of functions defined in `libNextDoor.hpp` that can be included in both Python 2 and Python 3 modules.

* `example` directory builds Uniform Random Walk as an example application.

# Graph Inputs

Graph Inputs used in the paper can be downloaded from the url: <b>TODO</b>. These graphs are obtained from snap.stanford.edu and each edge is assigned a random weight within [1, 5).

# Building Example Sampling Application
`example/uniformRandomWalk.cu` shows how to develop a Uniform Random Walk application using NextDoor.
To build the application, run 
```
make
```
To run the application on a graph in `input` directory.
```
./uniformRandomWalk ../ 
```
<b>TODO</b>

# Existing Sampling Applications
`src/apps/` directory contains implementation of these sampling applications:
* Single Dimension Random Walks: DeepWalk, PPR, and node2vec are implemented in `randomWalks.cu`
* K-Hop neighborhood sampling application is implemented in `khop.cu`.
* Multi Dimension Random Walk application is implemented in `multiRW.cu`
* Minimal Variance Sampling is implemented in `mvsSampling.cu`
* FastGCN and LADIES are implemented in `importanceSampling.cu`
* ClusterGCN is implemented in `clusterGCNSampling.cu`

To build and use these applications refer to Benchmark section.

# Benchmarking

Existing sampling applications are used as benchmarks in the paper. Following instructions can be used to do Single GPU or Multi GPU benchmarking:
### Single GPU Benchmarking
* To do Single GPU benchmarking, first clone the repo:
```
git clone https://github.com/plasma-umass/nextdoor/
cd nextdoor
```
* Download the graph datasets from <b>TODO</b> and copy them to input directory

```
wget <URL>
unzip inputs.tar.gz ./input/
```

* Make single GPU google test

```
make all-singleGPU-tests
```

* Now binaries for all applications are present in `./build/tests/singleGPU/`. To execute one of them say 'khop' on all input graphs do:
```
./build/tests/singleGPU/khop
```

### MultiGPU Benchmarking

First follow all steps described in Single GPU Benchmarking section. Then follow these steps:
* Make multi-gpu tests
```
make all-multiGPU-tests
``` 
* Binaries for all applications are present in `./build/tests/multiGPU`.
* To execute an application on more than one GPU, set the environment variable `CUDA_DEVICES` with all GPU device ids. For example, to execute KHop on GPUs [0,1,3], run:
```
CUDA_DEVICES=0,1,3 ./build/tests/multiGPU/khop
```
