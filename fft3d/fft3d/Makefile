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
CXXFLAGS += -g -DDEBUG
else
CXXFLAGS += -O2
endif

# Compiler
CXX := g++

# Target
TARGET := host
TARGET_DIR := bin

# Directories
INC_DIRS := host/inc ../common/inc host/fftw/include host/fftwF/include
LIB_DIRS := host/fftw/lib host/fftwF/lib

# Files
INCS := $(wildcard host/inc/*.h)
SRCS := $(wildcard host/src/*.cpp host/src/*.c ../common/src/AOCLUtils/*.cpp main.cpp)
LIBS := rt pthread fftw3 fftw3f

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
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

# KERNEL
# ---------------

# Target
TARGET_EMU_DIR := emulation
TARGET_SYN_DIR := synthesis
TARGET_INTER_DIR := report
TARGET_PROF_DIR := profile

TARGET_KERNEL := fft3d.aocx
TARGET_INTERMEDIATE := fft3d.aocr

# Files
KERNEL_SRCS := $(wildcard device/fft3d.cl)

# Board
#BOARD := p385a_sch_ax115
BOARD := p520_hpc_sg280l


emulate: $(TARGET_DIR)/$(TARGET_EMU_DIR)/$(TARGET_KERNEL)

# Kernel Emulation target
$(TARGET_DIR)/$(TARGET_EMU_DIR)/$(TARGET_KERNEL): Makefile $(KERNEL_SRCS) \
										$(TARGET_DIR)/$(TARGET_EMU_DIR) 
	$(ECHO)aoc -march=emulator -board=$(BOARD) -v -g $(KERNEL_SRCS) -o $@ 
# aoc -v -march=emulator device/dft.cl -o bin/dft.aocx --board p385a_mac_ax115

# kernel synthesis
syn: $(TARGET_DIR)/$(TARGET_SYN_DIR)/$(TARGET_KERNEL)

$(TARGET_DIR)/$(TARGET_SYN_DIR)/$(TARGET_KERNEL): Makefile $(KERNEL_SRCS) \
										$(TARGET_DIR)/$(TARGET_SYN_DIR)
	aoc -fp-relaxed --board $(BOARD) -g -v $(KERNEL_SRCS) -o $@ > syn_$$(date +%Y%m%d%H%M%S).log 2>&1 
	#nohup aoc -fp-relaxed -g -v $(KERNEL_SRCS) -o $@ > test.log &

profile: $(TARGET_DIR)/$(TARGET_PROF_DIR)/$(TARGET_KERNEL)

$(TARGET_DIR)/$(TARGET_PROF_DIR)/$(TARGET_KERNEL): Makefile $(KERNEL_SRCS) \
										$(TARGET_DIR)/$(TARGET_PROF_DIR)
	aoc -fp-relaxed -profile=all --board $(BOARD) -g -v $(KERNEL_SRCS) -o $@ > profile_$$(date +%Y%m%d%H%M%S).log 2>&1
	#nohup aoc -fp-relaxed -profile=all --board $(BOARD) -g -v $(KERNEL_SRCS) -o $@ > test_report.log
# kernel intermediate synthesis: report
report: $(TARGET_DIR)/$(TARGET_INTER_DIR)/$(TARGET_INTERMEDIATE)

$(TARGET_DIR)/$(TARGET_INTER_DIR)/$(TARGET_INTERMEDIATE): Makefile $(KERNEL_SRCS) \
										$(TARGET_DIR)/$(TARGET_INTER_DIR)
	aoc -fp-relaxed -g -report --board $(BOARD) -v -rtl $(KERNEL_SRCS) -o $@ > report.log
	#aoc -fp-relaxed -g -report --board $(BOARD) -v -c $(KERNEL_SRCS) -o $@ &

$(TARGET_DIR)/$(TARGET_EMU_DIR) : 
	$(ECHO)mkdir -p $@

$(TARGET_DIR)/$(TARGET_SYN_DIR) : 
	$(ECHO)mkdir -p $@

$(TARGET_DIR)/$(TARGET_INTER_DIR) : 
	$(ECHO)mkdir -p $@

$(TARGET_DIR)/$(TARGET_PROF_DIR) : 
	$(ECHO)mkdir -p $@
# nohup aoc --fp-relaxed --profile ../../device/matrix_mul.cl > test.log &
