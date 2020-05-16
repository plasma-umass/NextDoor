all: gpu

cpuO0:
	g++ src/main.cpp -std=c++11 -O0 -g -o cpu
cpu: 
	g++ src/main.cpp -std=c++11 -O2 -o cpu

gpuO0: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g -G -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -I. -o gpu -Xptxas -O0 -lcurand
gpu-gdb: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g  -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -o gpu -lcurand

gpu: src/main.cu src/csr.hpp src/utils.hpp src/graph.hpp src/pinned_memory_alloc.hpp src/sampler.cuh
	nvcc --default-stream per-thread $< -std=c++11 -arch=compute_61 -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -O3 -g -o gpu -Xptxas -O3 -Xcompiler -Wall -lcurand
gpuRelease: src/main.cu src/csr.hpp src/utils.hpp src/graph.hpp src/pinned_memory_alloc.hpp
	nvcc $< -std=c++11 -arch=compute_61 -code=sm_61 -I./Heap-Layers -I/mnt/homes/abhinav/cub-1.8.0 -O3 -o gpu -Xptxas -O3 -Xcompiler -Wall -DNDEBUG -lcurand

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o
