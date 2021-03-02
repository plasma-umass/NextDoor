MKDIR_P = mkdir -p
SINGLEGPU_TEST_BUILD_DIR = ./build/tests/singleGPU
MULTIGPU_TEST_BUILD_DIR = ./build/tests/multiGPU
TEST_INCLUDE_DIRS = -Igoogletest/googletest/include/  -Lgoogletest/build/lib/ -Isrc/ -Isrc/apps/ -Isrc/tests
GOOGLE_TEST_MAIN = googletest/googletest/src/gtest_main.cc
ARCH_CODE_FLAGS = -arch=compute_61 -code=sm_61
TEST_LFLAGS = -lcurand -lgtest -lpthread

all: tests
directories: 
	${MKDIR_P} $(SINGLEGPU_TEST_BUILD_DIR) $(MULTIGPU_TEST_BUILD_DIR)

#**************TESTS********************#
tests: all-singleGPU-tests all-multiGPU-tests

all-singleGPU-tests: directories $(SINGLEGPU_TEST_BUILD_DIR)/deepWalkTest $(SINGLEGPU_TEST_BUILD_DIR)/khopTest $(SINGLEGPU_TEST_BUILD_DIR)/layerTest $(SINGLEGPU_TEST_BUILD_DIR)/multiRWTest $(SINGLEGPU_TEST_BUILD_DIR)/subGraphSamplingTests $(SINGLEGPU_TEST_BUILD_DIR)/mvsSamplingTests 

all-multiGPU-tests: directories $(MULTIGPU_TEST_BUILD_DIR)/deepWalkTest $(MULTIGPU_TEST_BUILD_DIR)/khopTest $(MULTIGPU_TEST_BUILD_DIR)/layerTest $(MULTIGPU_TEST_BUILD_DIR)/multiRWTest $(MULTIGPU_TEST_BUILD_DIR)/subGraphSamplingTests $(MULTIGPU_TEST_BUILD_DIR)/mvsSamplingTests

$(SINGLEGPU_TEST_BUILD_DIR)/khopTest: src/tests/singleGPU/khopTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/khop.cu 
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp  -maxrregcount=40 -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/deepWalkTest: src/tests/singleGPU/deepWalk.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/layerTest: src/tests/singleGPU/layerTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/importanceSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/multiRWTest: src/tests/singleGPU/multiRW.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/multiRW.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/subGraphSamplingTests: src/tests/singleGPU/subGraphSamplingTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/clusterGCNSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(SINGLEGPU_TEST_BUILD_DIR)/mvsSamplingTests: src/tests/singleGPU/mvs.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/mvsSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

#******Multi GPU Tests********#
$(MULTIGPU_TEST_BUILD_DIR)/khopTest: src/tests/multiGPU/khopTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/khop.cu 
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp  -maxrregcount=40 -o $@

$(MULTIGPU_TEST_BUILD_DIR)/deepWalkTest: src/tests/multiGPU/deepWalk.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/randomWalks.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -O3 -Xptxas -O3 -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/layerTest: src/tests/multiGPU/layerTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/importanceSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/multiRWTest: src/tests/multiGPU/multiRW.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/multiRW.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/subGraphSamplingTests: src/tests/multiGPU/subGraphSamplingTests.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/clusterGCNSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

$(MULTIGPU_TEST_BUILD_DIR)/mvsSamplingTests: src/tests/multiGPU/mvs.cu src/nextdoor.cu src/tests/testBase.h src/check_results.cu src/apps/mvsSampling.cu
	nvcc $< $(TEST_INCLUDE_DIRS) $(TEST_LFLAGS) $(GOOGLE_TEST_MAIN) $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@
########################################

#*************APPS*********************#
clusterGCNSampling: apps/clustergcn.cu src/nextdoor.cu src/nextDoorModule.cu src/main.cu src/check_results.cu
	nvcc $< -IAnyOption/ AnyOption/anyoption.cpp -DPYTHON_3 -I/usr/include/python3.7m/ -Isrc -lcurand -lpthread $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@

fastgcn_sampling: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu 
	nvcc $< -IAnyOption/ AnyOption/anyoption.cpp -Isrc  -lcurand -lpthread  $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o $@
#**************************************#

#*************Python Modules*******#
fastgcn_samplingIntegrationPython2: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu src/libNextDoor.hpp src/nextDoorModule.cu
	nvcc $< -DPYTHON_2 -IAnyOption/ AnyOption/anyoption.cpp -I/usr/include/python2.7/ -Isrc  -lcurand -lpthread  $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC

fastgcn_samplingIntegrationPython3: apps/fastgcn_sampling.cu src/nextdoor.cu src/main.cu src/libNextDoor.hpp src/nextDoorModule.cu
	nvcc $< -DPYTHON_3 -IAnyOption/ AnyOption/anyoption.cpp -I/usr/include/python3.7m/ -Isrc -lcurand -lpthread  $(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -o NextDoor.so -shared -lcurand -Xptxas -O3 -Xcompiler -Wall,-fPIC
####################################

clean:
	rm -rf cpu gpu *.h.gch *.o src/*.h.gch src/*.o src/*.o build/*
