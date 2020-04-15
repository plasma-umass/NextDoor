all: gpu

cpuO0:
	g++ src/main.cpp -std=c++11 -O0 -g -o cpu
cpu: 
	g++ src/main.cpp -std=c++11 -O2 -o cpu

gpuO0: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g -G -I/mnt/homes/abhinav/cub-1.8.0 -o gpu -Xptxas -O0
gpu-gdb: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -O0 -g  -I/mnt/homes/abhinav/cub-1.8.0 -o gpu

gpu: src/main.cu
	nvcc $< -std=c++11 -arch=compute_61 -I/mnt/homes/abhinav/cub-1.8.0 -O2 -o gpu -Xptxas -O2 -Xcompiler -Wall

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o
