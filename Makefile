# Makefile for tvm_sgemm

# ----- User settings -----
TVM_HOME     := /home/jin/tvm
#TVM_HOME     := /media/jin/nvme1n1p1/tvm

# ----- Toolchain -----
CXX          := g++
CXXFLAGS     := -std=c++20

# ----- Include paths -----
INCLUDES     := \
	-I$(TVM_HOME)/include \
	-I$(TVM_HOME)/3rdparty/dmlc-core/include \
	-I$(TVM_HOME)/3rdparty/dlpack/include \
	-I$(TVM_HOME)/ffi/include

# ----- Linker settings -----
LDFLAGS      := -L$(TVM_HOME)/build
LDLIBS       := -ltvm_runtime

# ----- Targets -----
TARGET       := tvm_sgemm
SRCS         := main.cpp
OBJS         := $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Remove artifacts
clean:
	rm -f $(OBJS) $(TARGET)