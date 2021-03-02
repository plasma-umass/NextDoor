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

# Graph Inputs

Graph Inputs used in the paper can be downloaded from the url: <b>TODO</b>

# Building Example Sampling Application

# Existing Sampling Applications

# Benchmarking


