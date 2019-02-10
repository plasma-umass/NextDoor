all: cpu gpu

cpuO0:
	g++ src/main.cpp -std=c++11 -O0 -g -o cpu
cpu: 
	g++ src/main.cpp -std=c++11 -O2 -o cpu

gpuO0:
	nvcc src/main.cu -std=c++11 -arch=compute_60 -O0 -g -G -o gpu

gpu:
	nvcc src/main.cu -std=c++11 -arch=compute_60 -O3 -o gpu

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o
