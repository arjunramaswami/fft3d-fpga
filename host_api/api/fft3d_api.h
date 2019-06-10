/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#ifndef fft3d_api_h__
#   define fft3d_api_h__

// Initialize FPGA
extern int fpga_initialize_();

// Check fpga bitstream present in directory
extern bool fpga_check_bitstream_(int N[3]);

// Single precision FFT3d procedure
extern void fpga_fft3d_sp_(int data_path_len, char *data_path, int direction, cmplx *din);

// Double precision FFT3d procedure
extern void fpga_fft3d_dp_(int data_path_len, char *data_path, int direction, cmplx *din);

// Initialize FPGA
extern void fpga_final_();

#endif // fft3d_api_h__
