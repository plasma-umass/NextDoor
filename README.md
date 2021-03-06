# NextDoor
NextDoor accelerating graph sampling using GPUs.
NextDoor uses <it>transit</it> parallelism that improves memory access locality and decreases control flow divergence.
More details can be found in our EuroSys'21 paper. 


# Code Structure
* `src` directory contains the source of NextDoor. It contains following files and directories
  * `apps` contains following applications. These applications are provided with their Python bindings to integrate these applications in existing GNNs:
    * `clusterGCNSampling`  ClusterGCN Sampling 
    * `fastgcn` FastGCN Sampling
    * `ladies` LADIES Sampling 
    * `khop` GraphSAGE 2-Hop application
    * `multiRW` Multi Dimension Random Walk  
    * `mvsSampling` Minimum Variance Sampling
    * `randomWalks` DeepWalk, PPR, and node2vec random walks.
  * `tests` uses above applications as unit tests and provides a function to check results of these applications. 
  * `csr.hpp` and `graph.hpp` contains data structures for graphs.
  * `utils.hpp` contains utility functions.
  * `nextdoor.cu` contains NextDoor kernels and execution logic.
  * `libNextDoor.hpp` header file that contains functions to integrate NextDoor's execution logic in an external program.
  * `nextDoorModule.cu` contains implementation of functions defined in `libNextDoor.hpp` that can be included in both Python 2 and Python 3 modules.

* `example` directory builds Uniform Random Walk as an example application.

# Graph Inputs

Graph Inputs used in the paper can be downloaded from the url: https://drive.google.com/file/d/19duY39ygWS3RAiqEetHUKdv-4mi_F7vj/view?usp=sharing. These graphs are obtained from snap.stanford.edu and each edge is assigned a random weight within [1, 5).
Below commands download the dataset and then extract it to input folder.

```
sh ./downloadDatasets.sh
unzip input.zip
```

# Building Example Sampling Application
`example/uniformRandomWalk.cu` shows how to develop a Uniform Random Walk application using NextDoor.
First declare a sample class that can be used to store extra information.
Then create an application struct that implements several functions required to execute a random walk. These functions are:
* `steps`: returns the number of steps to travel from the starting vertices of a sample.
* `stepSize`: returns the number of vertices to sample at every step.
* `next`: returns the sampled vertex
* `samplingType`: returns the kind of sampling, i.e., <i>IndividualNeighborhood</i> or <i>CollectiveNeighborhood</i>.
* `stepTransits`: Transit vertex for given step.

Other functions that are needed to be defined includes:
* `initialSample`: Initial transit of sample
* `numSamples`: Number of samples
* `initialSampleSize`: Number of initial transits
* `initializeSample`: Initialize the sample struct

To build the application, run 
```
make
```

To run the application on PPI graph in `input` directory (see Graph Inputs section to download graphs) using 
TransitParallel with Load Balancing execute following command.
```
./uniformRandWalk -g ../input/ppi.data -t edge-list -f binary -n 1 -k TransitParallel -l
```

Command Line Arguments are:
* -g is path to graph, 
* -t is the type of storage of graph, which can be either a list of edges (edge-list) or adjacency list (adj-list), 
* -f is the format of storage, which can be either binary or text, 
* -n describes number of times to run it
* -k describes the kind of kernel to use Transit Parallel or Sample Parallel 
* -l describes if load balancing, caching, and other optimizations must be applied on Transit Parallel. This argument is valid for Transit Parallel only.


Python 2 and 3 modules are also generated using make and can be used as follows. Both modules defines following functions.
* initSampling(graph): It takes a path to graph that is stored in a binary edge-list format.
* sample(): This will perform the sampling
* finalSamples(): This will return the final samples in the form of a Python List.
* finalSampleLength(): This will return the size of each sample.
* finalSamplesArray(): This avoids building a list and gives a pointer to the final samples array of NextDoor. This pointer can then be converted to a numpy array.

```
$ python2
Python 2.7.17 (default, Sep 30 2020, 13:38:04)
[GCC 7.5.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import UniformRandomWalkPy2
>>> UniformRandomWalkPy2.initSampling("../input/ppi.data")
Graph Binary Loaded
Using GPUs: [0,]
Final Size of each sample: 100
Maximum Neighbors Sampled at each step: 1
Number of Samples: 56944
Maximum Threads Per Kernel: 57088
>>> UniformRandomWalkPy2.sample()
SampleParallel: End to end time 0.00748205 secs
```

### Executing other applications
Similarly, other applications in `src/apps` folder can be used.

# Benchmarking

Existing sampling applications are used as benchmarks in the paper. Following instructions can be used to do Single GPU or Multi GPU benchmarking:
### Single GPU Benchmarking
* To do Single GPU benchmarking, first clone the repo:
```
git clone --recurse-submodules https://github.com/plasma-umass/NextDoor.git
```
* Download the graph datasets from https://drive.google.com/file/d/19duY39ygWS3RAiqEetHUKdv-4mi_F7vj/view?usp=sharing and copy them to input directory

```
./downloadDataset.sh
unzip input.zip
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
