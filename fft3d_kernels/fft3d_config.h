/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef fft3d_config_h__
#define fft3d_config_h__

// determines FFT size 
#ifndef LOGN
#  define LOGN 6
#endif

// toggling between precision
#ifdef __FPGA_SP
    typedef float2 cmplx;
#else
    typedef double2 cmplx;
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#endif // fft3d_config_h__
