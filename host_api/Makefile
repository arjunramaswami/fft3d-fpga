# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Host Code used to run FFT3d kernels 

# Check if Intel OpenCL SDK is installed and path set
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# Compilers
CC := gcc

# Compiler Flags
CFLAGS := -march=native -fno-omit-frame-pointer -g 
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -fsanitize=leak -DDEBUG
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops
endif
SRCS := $(wildcard main.c common/*.c api/*.c)
CFLAGS := $(CFLAGS) $(SRCS) 

# Include Directories 
AOCL_HEADER_DIRS := $(shell aocl compile-config)
INC := -Icommon -Iapi 
INC_DIRS :=  $(FFTW_INC) $(AOCL_HEADER_DIRS) $(INC)

# Linker Directories
AOCL_LINKER_DIRS = $(shell aocl ldflags) 
LIB_DIRS :=  $(FFTW_LIB) $(AOCL_LINKER_FLAGS)

# Libraries
AOCL_LIBS = $(shell aocl ldlibs)
ifeq ($(DEBUG),1)
LIBS := -lrt -lpthread -lfftw3 -lfftw3f
else
LIBS := -lrt -lpthread
endif
LIBS := $(LIBS) $(AOCL_LIBS)

# Target
TARGETDIR := .
TARGET := host

all : 
	mkdir -p $(TARGETDIR)
	$(CC) $(CFLAGS) $(INC_DIRS) $(LIB_DIRS)	$(LIBS) -o $(TARGETDIR)/$(TARGET)

clean :
	rm -f $(TARGETDIR)/$(TARGET)

.PHONY : all clean # ---------------