EXECUTABLE := cuda_matmul

CU_FILES   := cuda_matmul.cu
CC_FILES   := main.cpp ../sparse_matrix.cpp
CU_DEPS    :=

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -I. -I.. -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -Xcompiler -fPIC

OBJS=$(OBJDIR)/main.o $(OBJDIR)/cuda_matmul.o $(OBJDIR)/sparse_matrix.o

.PHONY: dirs clean

all: dirs $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/sparse_matrix.o: ../sparse_matrix.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: main.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/cuda_matmul.o: cuda_matmul.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
