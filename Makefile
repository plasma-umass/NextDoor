all: gpu

cpuO0:
	g++ src/main.cpp -std=c++11 -O0 -g -o cpu
cpu: 
	g++ src/main.cpp -std=c++11 -O2 -o cpu

gpuO0: src/nextdoor.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g -G -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -I. -I/mnt/homes/abhinav/cub-1.8.0 -IAnyOption/ AnyOption/anyoption.cpp -o gpu -Xptxas -O0 -lcurand

gpu-gdb: src/nextdoor.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g  -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -I/mnt/homes/abhinav/cub-1.8.0 -IAnyOption/ AnyOption/anyoption.cpp -o gpu -lcurand

gpu: src/main.cu src/csr.hpp src/utils.hpp src/graph.hpp src/pinned_memory_alloc.hpp src/sampler.cuh src/check_results.cu src/nextdoor.cu
	nvcc --default-stream per-thread $<  -std=c++11 -arch=compute_61 -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -IAnyOption/ AnyOption/anyoption.cpp -O3 -g -o gpu -Xptxas -O3 -Xcompiler -Wall -lcurand

gpuRelease: src/nextdoor.cu src/csr.hpp src/utils.hpp src/graph.hpp src/pinned_memory_alloc.hpp
	nvcc $< -std=c++11 -arch=compute_61 -code=sm_61 -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -O3 -o gpu -Xptxas -O3 -Xcompiler -Wall -DNDEBUG -lcurand

graphSageIntegration: src/nextDoorModule.cu src/csr.hpp src/utils.hpp src/graph.hpp src/pinned_memory_alloc.hpp src/sampler.cuh src/check_results.cu src/nextdoor.cu
	nvcc --default-stream per-thread $< -I/usr/include/python2.7/ -std=c++11 -arch=compute_61 -I/mnt/homes/abhinav/cub-1.8.0 -IAnyOption/ AnyOption/anyoption.cpp -I./Heap-Layers -O3 -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC

test: khopTest uniformRandWalkTest deepWalkTest
	
khopTest: src/tests/khopTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu
	nvcc $< -Igoogletest/googletest/include/ -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lgtest -lpthread googletest/googletest/src/gtest_main.cc -arch=compute_61 -code=sm_61 -Xcompiler -fopenmp -o $@

deepWalkTest: src/tests/deepWalk.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu
	nvcc $< -Igoogletest/googletest/include/ -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lgtest -lpthread googletest/googletest/src/gtest_main.cc -arch=compute_61 -code=sm_61 -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

uniformRandWalkTest: src/tests/uniformRandWalk.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu
	nvcc $< -Igoogletest/googletest/include/ -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lgtest -lpthread googletest/googletest/src/gtest_main.cc -arch=compute_61 -code=sm_61 -Xcompiler -fopenmp -o $@

layerTest: src/tests/layerTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu
	nvcc $< -Igoogletest/googletest/include/ -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lgtest -lpthread googletest/googletest/src/gtest_main.cc -arch=compute_61 -code=sm_61 -Xcompiler -fopenmp -o $@

fastgcn_sampling: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu 
	nvcc $< -IAnyOption/ AnyOption/anyoption.cpp -Isrc -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lpthread  -arch=compute_61 -code=sm_61 -Xcompiler -fopenmp -o $@

fastgcn_samplingIntegrationPython2: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu src/libNextDoor.hpp src/nextDoorModule.cu
	nvcc $< -DPYTHON_2 -IAnyOption/ AnyOption/anyoption.cpp -I/usr/include/python2.7/ -Isrc -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lpthread  -arch=compute_61 -code=sm_61 -Xcompiler -fopenmp -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC

fastgcn_samplingIntegrationPython3: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu src/libNextDoor.hpp src/nextDoorModule.cu
	nvcc $< -DPYTHON_3 -IAnyOption/ AnyOption/anyoption.cpp -I/usr/include/python3.7m/ -Isrc -I/mnt/homes/abhinav/cub-1.8.0 -Lgoogletest/build/lib/ -lcurand -lpthread  -arch=compute_61 -code=sm_61 -Xcompiler -fopenmp -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o
