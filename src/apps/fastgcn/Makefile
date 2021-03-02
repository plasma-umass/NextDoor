include ../../../common.mk

all: bin py2module py3module

py2module: pythonModule.cu
	nvcc pythonModule.cu -DPYTHON_2 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I/usr/include/python2.7/ -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o FastGCNSamplingPy2.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

py3module: pythonModule.cu
	nvcc pythonModule.cu -DPYTHON_2 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I/usr/include/python2.7/ -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o FastGCNSamplingPy3.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

bin: pythonModule.cu
	nvcc pythonModule.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o FastGCNSampling -lcurand -Xptxas -O3

clean:
	rm FastGCNSampling *.so