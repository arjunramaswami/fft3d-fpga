/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <fftw3.h>

// common dependencies
#include "CL/opencl.h"
#include "api/fft_api.h"  // Common declarations and API
#include "api/fft_fpga.h"  // Common declarations and API

// local dependencies
#include "common/argparse.h"  // Cmd-line Args to set some global vars
#include "common/helper.h"  // Cmd-line Args to set some global vars

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

void main(int argc, const char **argv) {
  unsigned i = 0;
  double fpga_runtime = 0.0, fftw_runtime = 0.0;
  const int inp_filename_len = 50;
  char inp_fname[inp_filename_len];
  char *bitstream_path = "../fft3d_kernels/fpgabitstream";

  cmplx *fft_data;
  // Need distinct data for sp and dp FFTW for separate function calls
  fftwf_complex *fftw_sp_data;
  fftw_complex *fftw_dp_data;

  // Cmd line argument declarations
  int N[3] = {64, 64, 64};
  unsigned iter = 1, inverse = 0;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N[1], "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N[2], "FFT 3rd Dim Size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_BOOLEAN('b',"back", &inverse, "Backward/inverse FFT"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FPGA", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  printf("------------------------------\n");
  printf("%s FFT3d Size : %d %d %d\n", inverse ? "Backward":"Forward", N[0], N[1], N[2]);
  printf("------------------------------\n");

  // Allocate mem for input buffer and fftw buffer
  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
#ifdef __FPGA_SP
  printf("Allocating Single Precision Floating Points\n");
  int num_bytes = snprintf(inp_fname, inp_filename_len, "../inputfiles/input_f%d_%d_%d.inp", N[0], N[1], N[2]);
  if(num_bytes > inp_filename_len){
    printf("Insufficient buffer size to store path to inputfile\n");
    exit(1);
  }
  fftw_sp_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
  get_sp_input_data(fft_data, fftw_sp_data, N, inp_fname);
#else
  printf("Allocating Double Precision Floating Points\n");
  int num_bytes = snprintf(inp_fname, inp_filename_len, "../inputfiles/input_d%d_%d_%d.inp", N[0], N[1], N[2]);
  if(num_bytes > inp_filename_len){
    printf("Insufficient buffer size to store path to inputfile\n");
    exit(1);
  }
  fftw_dp_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);
  get_dp_input_data(fft_data, fftw_dp_data, N, inp_fname);
#endif

  // execute FFT3d iter number of times
  for( i = 0; i < iter; i++){
    printf("\nCalculating %sFFT3d - %d\n", inverse ? "inverse ":"", i);

    // initialize FPGA
    if (fpga_initialize_()){
      printf("Error initializing FPGA. Exiting\n");
      if(fft_data)
        free(fft_data);
      #ifdef __FPGA_SP
        fftwf_free(fftw_sp_data);
      #else
        fftw_free(fftw_dp_data);
      #endif 
      exit(1);
    }

    // check if required bitstream exists
    if(!fpga_check_bitstream_(bitstream_path, N)){
      printf("Bitstream not found. Exiting\n");
      if(fft_data)
        free(fft_data);
      #ifdef __FPGA_SP
        fftwf_free(fftw_sp_data);
      #else
        fftw_free(fftw_dp_data);
      #endif 
      exit(1);
    }

    // execute fpga fft3d
    double start = getTimeinMilliSec();
#ifdef __FPGA_SP
    fpga_fft3d_sp_(!inverse, N, fft_data);
#else
    fpga_fft3d_dp_(!inverse, N, fft_data);
#endif
    double stop = getTimeinMilliSec();
    fpga_runtime += stop - start;

    printf("Computing FFTW\n");
#ifdef __FPGA_SP
    fftw_runtime = compute_sp_fftw(fftw_sp_data, N, inverse);
    printf("Checking SP Correctness\n");
    verify_sp_fft(fft_data, fftw_sp_data, N);
#else
    fftw_runtime = compute_dp_fftw(fftw_dp_data, N, inverse);
    printf("Checking DP Correctness\n");
    verify_dp_fft(fft_data, fftw_dp_data, N);
#endif
  }

  // Print performance metrics
  compute_metrics( fpga_runtime, fftw_runtime, iter, N);

  // Free the resources allocated
  printf("Cleaning up\n");
  if(fft_data)
    free(fft_data);
  #ifdef __FPGA_SP
    fftwf_free(fftw_sp_data);
  #else
    fftw_free(fftw_dp_data);
  #endif 
}