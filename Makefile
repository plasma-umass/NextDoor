all: gpu

cpuO0:
	g++ src/main.cpp -std=c++11 -O0 -g -o cpu
cpu: 
	g++ src/main.cpp -std=c++11 -O2 -o cpu

gpuO0: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g -G -o gpu -Xptxas -O0
gpu-gdb: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g -o gpu

gpu: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O3 -o gpu -Xptxas -O3,-dlcm=ca

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o
