include ../../../common.mk

all: bin py2module py3module

py2module: main.cu
	nvcc main.cu -DPYTHON_2 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_2_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o FastGCNSamplingPy2.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

py3module: main.cu
	nvcc main.cu -DPYTHON_3 -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I$(PYTHON_3_INCLUDE_PATH) -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o FastGCNSamplingPy3.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

bin: main.cu
	nvcc main.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o FastGCNSampling -lcurand -Xptxas -O3

clean:
	rm FastGCNSampling *.so