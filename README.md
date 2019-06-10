# FFT3d for FPGAs

This repository contains the OpenCL implementation of FFT3d for Intel FPGAs 
as well as an API to make use of the implementation for further development and
experimentation. 

Currently tested only for Intel Arria 10 and Stratix 10 FPGAs. 

## FPGA Kernel Code
The folder `fft3d_kernels` contains the OpenCL kernel code with instructions on
synthesizing them.

## API (WORK IN PROGRESS!)
A sample set of files in `host_api` contains APIs that facilitates development
of kernel code for FPGAs. 

 - Making use of the OpenCL APIs to setup and execute the bitstreams generated 
   by the FFT3d kernels code for the FPGAs
 - Evaluate the correctness of the code by comparing with FFTW

