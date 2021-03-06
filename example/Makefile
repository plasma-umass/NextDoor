include ../common.mk

all: uniformRandWalk py2module py3module

py2module: uniformRandWalk.cu
	nvcc uniformRandWalk.cu -DPYTHON_2 -I../AnyOption/ ../AnyOption/anyoption.cpp -I$(PYTHON_2_INCLUDE_PATH) -I../src  -I../ -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o UniformRandomWalkPy2.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

py3module: uniformRandWalk.cu
	nvcc uniformRandWalk.cu -DPYTHON_3 -I../AnyOption/ ../AnyOption/anyoption.cpp -I$(PYTHON_3_INCLUDE_PATH) -I../src -I../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o UniformRandomWalkPy3.so -shared -lcurand -Xptxas -O3 -Xcompiler -fPIC

uniformRandWalk: 
	nvcc ./uniformRandWalk.cu -I../src/ -I../AnyOption ../AnyOption/anyoption.cpp -lcurand -lpthread $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -o $@

clean:
	rm uniformRandWalk
