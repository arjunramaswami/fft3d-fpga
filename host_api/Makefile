# You must configure ALTERAOCLSDKROOT to point the root directory of the Intel(R) FPGA SDK for OpenCL(TM)
# software installation.

ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )

# Compilation flags
ifeq ($(DEBUG),1)
CFLAGS += -std=c11 -lm -g -DDEBUG
else
CFLAGS += -std=c11 -lm -march=native -fno-omit-frame-pointer -g  -O3 -funroll-loops
endif

# Compiler
CC := gcc

# Target
TARGET := host
TARGET_DIR := .

# FFTW
FFTW_INC_DIR := $(FFTW_INC)
FFTW_LIB_DIR := $(FFTW_DIR)

# Directories
INC_DIRS :=  $(FFTW_INC_DIR)
LIB_DIRS :=  $(FFTW_LIB_DIR)

# Files
INCS := $(wildcard common/*.h fpga/*.h)
SRCS := $(wildcard common/*.c fpga/*.c main.c)

ifeq ($(DEBUG),1)
LIBS := rt pthread fftw3 fftw3f
else
LIBS := rt pthread
endif

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)
	
# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)

.PHONY : all clean # ---------------