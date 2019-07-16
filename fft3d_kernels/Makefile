# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Multi Step Compilation for Intel OpenCL Kernel Code

# Check if Intel OpenCL SDK is installed and path set
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# Compilers
AOC := aoc

# Compiler Flags
AOCFLAGS   := -g -v
FPFLAGS    :=
EMUFLAGS   := -march=emulator
SIMFLAGS   := -march=simulator 
PROFFLAGS  := -fp-relaxed -profile=all
INTERFLAGS := -fp-relaxed -report -rtl
SYNFLAGS   := -fp-relaxed
	
# Libraries 

# Board Name
BOARD := p520_hpc_sg280l     # Stratix 10
#BOARD := p385a_sch_ax115    # Arria 10

# Paths to targets
TARGET_PATH    := fpgabitstream/fft3d
TARGET_DP_DIR  := $(TARGET_PATH)/synthesis_dp
TARGET_SP_DIR  := $(TARGET_PATH)/synthesis_sp

# Pass FPGA_DP=1 as cmd line arg to make to activate double precision compilation 
ifeq ($(FPGA_DP),1)
	TARGET_DIR := $(TARGET_DP_DIR)
 	FPFLAGS :=  $(FPFLAGS)
else
	TARGET_DIR := $(TARGET_SP_DIR)
	FPFLAGS := $(FPFLAGS) -D __FPGA_SP
endif

TARGET_EMU_DIR   ?= $(TARGET_DIR)/emulation
TARGET_SIM_DIR   ?= $(TARGET_DIR)/simulation
TARGET_INTER_DIR ?= $(TARGET_DIR)/report
TARGET_PROF_DIR  ?= $(TARGET_DIR)/profile
TARGET_SYN_DIR   ?= $(TARGET_DIR)/synthesis
	
# Files
KERNEL_SRCS := ./fft3d.cl

# Targets
TARGET_KERNEL := fft3d.aocx
TARGET_INTERMEDIATE := fft3d.aocr

default: help 
	$(error No target specified)

help: 
	$(info ***************************************************)
	$(info Choose one of the steps to compile the FFT3d kernel)
	$(info ***************************************************)
	$(info emulate  =  Compile kernels for emulation)
	$(info syn      =  Synthesize kernels  )
	$(info inter    =  Intermediate Report )
	$(info profile  =  Profile kernels  )
	$(info sim      =  Simulate kernels )
	$(info clean    =  deletes syn folders and files)
	$(info emuclean =  deletes emulation folders and files)
	$(info fullclean = deletes all files )
	$(info help     =  Prints help )
	$(info ***************************************************)
	$(info Command Line Flags with Make:					  )
	$(info ***************************************************)
	$(info FPGA_DP=1   = enables emulation with double precision floating operations)
	$(info ***************************************************)

# Kernel Emulation
emulate: 
	mkdir -p $(TARGET_EMU_DIR)
	aoc $(FPFLAGS) $(AOCFLAGS) $(EMUFLAGS) $(KERNEL_SRCS) -o $(TARGET_EMU_DIR)/$(TARGET_KERNEL)

# kernel intermediate compilation for report
inter: 
	mkdir -p $(TARGET_INTER_DIR)
	aoc $(FPFLAGS) $(AOCFLAGS) $(INTERFLAGS) $(KERNEL_SRCS) -o $(TARGET_INTER_DIR)/$(TARGET_INTERMEDIATE)

# Kernel Simulation - Only single precision floating point ops
sim: 
	mkdir -p $(TARGET_SIM_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(SIMFLAGS) $(KERNEL_SRCS) -o $(TARGET_SIM_DIR)/$(TARGET_KERNEL)

# Kernel Profiling - Only single precision floating point ops
profile: 
	mkdir -p $(TARGET_PROF_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(PROFFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_PROF_DIR)/$(TARGET_KERNEL) 
	#@echo aoc -D __FPGA_SP $(AOCFLAGS) $(PROFFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_PROF_DIR)/$(TARGET_KERNEL) > profile_$$(date +%Y%m%d%H%M%S).log 2>&1

# Kernel Synthesis - Only single precision floating point ops
syn: 
	mkdir -p $(TARGET_SYN_DIR)
	aoc -D __FPGA_SP $(AOCFLAGS) $(SYNFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_SYN_DIR)/$(TARGET_KERNEL) 
	#@echo aoc -D __FPGA_SP $(AOCFLAGS) $(SYNFLAGS) --board $(BOARD) $(KERNEL_SRCS) -o $(TARGET_SYN_DIR)/$(TARGET_KERNEL) > syn_$$(date +%Y%m%d%H%M%S).log 2>&1 
	#nohup aoc -fp-relaxed -g -v $(KERNEL_SRCS) -o $@ > test.log &

clean:
	rm -rf d$(TARGET_SYN_DIR)/
emuclean: 
	rm -rf $(TARGET_EMU_DIR)/
fullclean: 
	rm -rf $(TARGET_PATH)/

.PHONY : clean emuclean fullclean syn profile sim inter emulate help default # ---------------