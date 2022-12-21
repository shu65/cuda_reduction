CXXFLAGS=-g
NVCCFLAGS=-g -G

LDFLAGS =

SRCDIR=src
CU_SRCS=$(shell find $(SRCDIR) -name '*.cu')
OBJS=$(CU_SRCS)
CPP_SRCS=$(shell find $(SRCDIR) -name '*.cpp')
OBJS+=$(CPP_SRCS)
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
TARGET=main

.SUFFIXES: .o

.PHONY: all
all:$(TARGET)

$(TARGET): $(OBJS)
	nvcc $(LDFLAGS) $(OBJS) -o $@

%.o: %.cpp
	nvcc -c $(CXXFLAGS) $< -o $@

%.o: %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@ 


.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)