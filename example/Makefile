all: uniformRandWalk

uniformRandWalk: 
	nvcc ./uniformRandWalk.cu -I../src/ -I../AnyOption ../AnyOption/anyoption.cpp -lcurand -lpthread -arch=compute_61 -code=sm_61 -O3 -Xptxas -O3 -o $@