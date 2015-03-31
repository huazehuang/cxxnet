ifndef config
ifdef CXXNET_CONFIG
	config = $(CXXNET_CONFIG)
else ifneq ("$(wildcard ../config.mk)","")
	config = ../config.mk
else
	config = ../make/config.mk
endif
endif

# use customized config file
include $(config)
include ../mshadow/make/mshadow.mk

# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC =nvcc

# caffe path
export CAFFEINCLUDE = /home/zehua/deep/caffe/include
export CAFFELIB = /home/zehua/deep/caffe/build/lib

export CFLAGS_CONV = -I$(CAFFEINCLUDE) -I../src/ -DMSHADOW_DIST_PS=0  -g -O3 -I../mshadow/  -fPIC $(MSHADOW_CFLAGS) 
export LDFLAGS_CONV = -L$(CAFFELIB) -lcaffe -pthread $(MSHADOW_LDFLAGS) -lglog

export CFLAGS = -Wall -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../mshadow/ -I.. -DMSHADOW_USE_MKL=0

export LDFLAGS= ../nnet_cpu.o ../nnet_gpu.o ../layer_cpu.o ../layer_gpu.o ../updater_cpu.o ../updater_gpu.o ../nnet_ps_server.o
export NVCCFLAGS = -g -O3 -ccbin $(CXX)


# specify tensor path
BIN = im2bin
CONVBIN = convert_mean convert
OBJ =
CUOBJ =
CUBIN =
.PHONY: clean all

all: $(CONVBIN) $(BIN) $(OBJ) $(CUBIN) $(CUOBJ) 

im2bin: im2bin.cpp
convertMean: convertMean.cpp
convert: convert.cpp

$(BIN) :
	$(CXX) $(CFLAGS) $(CFLAGS_CONV) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS) $(LDFLAGS_CONV)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)

$(CONVBIN) :
	$(CXX) $(CFLAGS) $(CFLAGS_CONV) -o $@ $(filter %.cpp %.o %.c, $^)  $(LDFLAGS) $(LDFLAGS_CONV)

clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) $(CONVBIN) *~

