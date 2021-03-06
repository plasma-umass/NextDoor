# cuSampler
cuSampler accelerating graph sampling using GPUs.
cuSampler uses <it>transit</it> parallelism that improves memory access locality and decreases control flow divergence.
More details can be found in our EuroSys'21 paper. 


# Code Structure
* `src` directory contains the source of cuSampler. It contains following files and directories
  * `apps` contains following applications. These applications are provided with their Python bindings to integrate these applications in existing GNNs:
    * `clusterGCNSampling`  ClusterGCN [4] Sampling 
    * `fastgcn` FastGCN [2] Sampling
    * `ladies` LADIES[3] Sampling 
    * `khop` GraphSAGE [1] 2-Hop application
    * `multiRW` Multi Dimension Random Walk [5] 
    * `mvsSampling` Minimum Variance Sampling [6]
    * `randomWalks` DeepWalk [7], PPR [8], and node2vec [9] random walks.
  * `tests` uses above applications as unit tests and provides a function to check results of these applications. 
  * `csr.hpp` and `graph.hpp` contains data structures for graphs.
  * `utils.hpp` contains utility functions.
  * `nextdoor.cu` contains cuSampler kernels and execution logic.
  * `libNextDoor.hpp` header file that contains functions to integrate cuSampler's execution logic in an external program.
  * `nextDoorModule.cu` contains implementation of functions defined in `libNextDoor.hpp` that can be included in both Python 2 and Python 3 modules.

* `example` directory builds Uniform Random Walk as an example application.

# Graph Inputs

Graph Inputs used in the paper can be downloaded from the url: https://drive.google.com/file/d/19duY39ygWS3RAiqEetHUKdv-4mi_F7vj/view?usp=sharing. These graphs are obtained from snap.stanford.edu and each edge is assigned a random weight within [1, 5).

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

# Benchmarking

Existing sampling applications are used as benchmarks in the paper. Following instructions can be used to do Single GPU or Multi GPU benchmarking:
### Single GPU Benchmarking
* To do Single GPU benchmarking, first clone the repo:
```
git clone --recurse-submodules https://github.com/plasma-umass/NextDoor.git
```
* Download the graph datasets from https://drive.google.com/file/d/19duY39ygWS3RAiqEetHUKdv-4mi_F7vj/view?usp=sharing and copy them to input directory

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19duY39ygWS3RAiqEetHUKdv-4mi_F7vj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19duY39ygWS3RAiqEetHUKdv-4mi_F7vj" -O input.zip && rm -rf /tmp/cookies.txt 'https://docs.google.com/uc?export=download&id=19duY39ygWS3RAiqEetHUKdv-4mi_F7vj -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19duY39ygWS3RAiqEetHUKdv-4mi_F7vj" -O input.zip && rm -rf /tmp/cookies.txt
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

# References
<b>todo</b>
[1] GraphSAINT: Graph Sampling Based Inductive Learning Method (ICLR 2020)

[2] Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks (KDD 2020)

[3] Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (KDD 2019)

[4] Layer-Dependent Importance Sampling for Training Deep and Large Graph Convolutional Networks (Neurips 2019)

[5] FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (ICLR 2018)
