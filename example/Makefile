include ../common.mk

all: uniformRandWalk

uniformRandWalk: 
	nvcc ./uniformRandWalk.cu -I../src/ -I../AnyOption ../AnyOption/anyoption.cpp -lcurand -lpthread $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -o $@

clean:
	rm uniformRandWalk
