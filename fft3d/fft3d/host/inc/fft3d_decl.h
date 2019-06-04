/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef fft3d_decl_h__
#  define fft3d_decl_h__

// FFT operates with complex numbers - store them in a struct
typedef struct {
  double x;
  double y;
} double2;

typedef struct {
  float x;
  float y;
} float2;

#ifdef __FPGA_SP
    typedef float2 cmplx;
#else
    typedef double2 cmplx;
#endif

#endif // fft3d_decl_h__