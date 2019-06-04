/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

//#include "fft_decl.h"

#ifndef fft3d_h__
#define fft3d_h__

extern int fpga_initialize_();

extern bool fpga_check_bitstream_(int N[3]);

extern void fpga_fft3d_sp_(int data_path_len, char *data_path, int direction, int N[3], cmplx *din);
//extern int fft3d(int, cmplex*, cmplex*, bool);
extern void fpga_final_();

#endif // fft3d_h__
