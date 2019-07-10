/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef FFT3D_CONFIG_H
#define FFT3D_CONFIG_H

// toggling between precision
#ifdef __FPGA_SP
#pragma message " Single Precision Activated"
    typedef float2 cmplx;
#else
#pragma message " Double Precision Activated"
    typedef double2 cmplx;
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#endif //  FFT3D_CONFIG_H
