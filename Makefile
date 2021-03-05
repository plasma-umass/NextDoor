include common.mk

MKDIR_P = mkdir -p
SINGLEGPU_TEST_BUILD_DIR = ./build/tests/singleGPU
MULTIGPU_TEST_BUILD_DIR = ./build/tests/multiGPU
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
TEST_LFLAGS = -lcurand -lgtest -lpthread

all: tests
directories: 
	${MKDIR_P} $(SINGLEGPU_TEST_BUILD_DIR) $(MULTIGPU_TEST_BUILD_DIR)

#**************TESTS********************#
tests: all-singleGPU-tests all-multiGPU-tests

all-singleGPU-tests: directories $(SINGLEGPU_TEST_BUILD_DIR)/deepwalk $(SINGLEGPU_TEST_BUILD_DIR)/ppr \
$(SINGLEGPU_TEST_BUILD_DIR)/node2vec $(SINGLEGPU_TEST_BUILD_DIR)/khop $(SINGLEGPU_TEST_BUILD_DIR)/fastgcn \
$(SINGLEGPU_TEST_BUILD_DIR)/ladies \
$(SINGLEGPU_TEST_BUILD_DIR)/multirw $(SINGLEGPU_TEST_BUILD_DIR)/clustergcn $(SINGLEGPU_TEST_BUILD_DIR)/mvs \
$(SINGLEGPU_TEST_BUILD_DIR)/layer

all-multiGPU-tests: directories $(MULTIGPU_TEST_BUILD_DIR)/deepwalk $(MULTIGPU_TEST_BUILD_DIR)/khop $(MULTIGPU_TEST_BUILD_DIR)/fastgcn \
	$(MULTIGPU_TEST_BUILD_DIR)/ladies \
	$(MULTIGPU_TEST_BUILD_DIR)/multirw $(MULTIGPU_TEST_BUILD_DIR)/clustergcn $(MULTIGPU_TEST_BUILD_DIR)/mvs \
	$(MULTIGPU_TEST_BUILD_DIR)/ppr $(MULTIGPU_TEST_BUILD_DIR)/node2vec

$(SINGLEGPU_TEST_BUILD_DIR)/khop: src/tests/singleGPU/khop.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/khop/khop.cu 
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp  -maxrregcount=40 -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/deepwalk: src/tests/singleGPU/deepwalk.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomwalks/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/ppr: src/tests/singleGPU/ppr.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomwalks/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/node2vec: src/tests/singleGPU/node2vec.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomwalks/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/fastgcn: src/tests/singleGPU/fastgcn.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/fastgcn/fastgcnSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/ladies: src/tests/singleGPU/ladies.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/ladies/ladiessampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/multirw: src/tests/singleGPU/multirw.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/multiRW/multiRW.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/clustergcn: src/tests/singleGPU/clustergcn.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/clustergcn/clusterGCNSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/mvs: src/tests/singleGPU/mvs.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/mvs/mvsSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/layer: src/tests/singleGPU/layer.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/layer/layerSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

#******Multi GPU Tests********#
$(MULTIGPU_TEST_BUILD_DIR)/khop: src/tests/multiGPU/khop.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/khop/khop.cu 
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp  -maxrregcount=40 -o $@

$(MULTIGPU_TEST_BUILD_DIR)/deepwalk: src/tests/multiGPU/deepwalk.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomwalks/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/node2vec: src/tests/multiGPU/node2vec.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomwalks/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/ppr: src/tests/multiGPU/ppr.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomwalks/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/fastgcn: src/tests/multiGPU/fastgcn.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/fastgcn/fastgcnSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/ladies: src/tests/multiGPU/ladies.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/ladies/ladiessampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/multirw: src/tests/multiGPU/multirw.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/multiRW/multiRW.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/clustergcn: src/tests/multiGPU/clustergcn.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/clustergcn/clusterGCNSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/mvs: src/tests/multiGPU/mvs.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/mvs/mvsSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@
########################################

#*************APPS*********************#
clusterGCNSampling: apps/clustergcn.cu src/nextdoor.cu src/nextDoorModule.cu src/main.cu src/check_results.cu
	nvcc $< -IAnyOption/ AnyOption/anyoption.cpp -DPYTHON_3 -I/usr/include/python3.7m/ -Isrc -lcurand -lpthread $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

fastgcn_sampling: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu 
	nvcc $< -IAnyOption/ AnyOption/anyoption.cpp -Isrc  -lcurand -lpthread  $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@
#**************************************#

#*************Python Modules*******#
fastgcn_samplingIntegrationPython2: src/apps/fastgcn/pythonModule.cu src/nextdoor.cu src/main.cu src/libNextDoor.hpp src/nextDoorPythonModule.cu
	nvcc $< -DPYTHON_2 -IAnyOption/ AnyOption/anyoption.cpp -I/usr/include/python2.7/ -Isrc  -lcurand -lpthread  $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC

fastgcn_samplingIntegrationPython3: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu src/libNextDoor.hpp src/nextDoorModule.cu
	nvcc $< -DPYTHON_3 -IAnyOption/ AnyOption/anyoption.cpp -I/usr/include/python3.7m/ -Isrc -lcurand -lpthread  $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC
####################################

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o build/*
