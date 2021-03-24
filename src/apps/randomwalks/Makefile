include ../../../common.mk

all: bin py2module py3module

bin: DeepWalkBin PPRBin Node2VecBin

py2module: DeepWalkPy2module PPRPy2module Node2VecPy2module

py3module: DeepWalkPy3module PPRPy3module Node2VecPy3module

DeepWalkPy2module: deepwalk.cu
	nvcc deepwalk.cu -DPYTHON_2 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_2_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o DeepWalkSamplingPy2.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

DeepWalkPy3module: deepwalk.cu
	nvcc deepwalk.cu -DPYTHON_3 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_3_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o DeepWalkSamplingPy3.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

DeepWalkBin: deepwalk.cu
	nvcc deepwalk.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o DeepWalkSampling -lcurand -Xptxas -O3


PPRPy2module: ppr.cu
	nvcc ppr.cu -DPYTHON_2 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_2_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o PPRSamplingPy2.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

PPRPy3module: ppr.cu
	nvcc ppr.cu -DPYTHON_3 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_3_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o PPRSamplingPy3.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

PPRBin: ppr.cu
	nvcc ppr.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o PPRSampling -lcurand -Xptxas -O3


Node2VecPy2module: node2vec.cu
	nvcc node2vec.cu -DPYTHON_2 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_2_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o Node2VecSamplingPy2.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

Node2VecPy3module: node2vec.cu
	nvcc node2vec.cu -DPYTHON_3 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_3_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o Node2VecSamplingPy3.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

Node2VecBin: node2vec.cu
	nvcc node2vec.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o Node2VecSampling -lcurand -Xptxas -O3

clean:
	rm DeepWalkSampling *.so PPRSampling Node2VecSampling