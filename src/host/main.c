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

void print_config(int N, int dim, int iter, int inv, int sp);

void main(int argc, const char **argv) {
  int N = 64, dim = 1, iter = 1, inv = 0, sp = 0;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "FFT Points"),
    OPT_INTEGER('d',"dim", &dim, "Dimensions"),
    OPT_BOOLEAN('s',"sp", &sp, "Single Precision"),
    OPT_INTEGER('i',"iter", &iter, "Iterations"),
    OPT_BOOLEAN('b',"back", &inv, "Backward FFT"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT using FPGA", "FFT size and dimensions are mandatory, default dimension and number of iterations are 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Print to console the configuration chosen to execute during runtime
  print_config(N, dim, iter, inv, sp);

  /*
  // Allocate mem for input buffer and fftw buffer
  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
#ifdef __FPGA_SP
  printf("Obtaining Single Precision Data\n");
  int num_bytes = snprintf(inp_fname, inp_filename_len, "../inputfiles/input_f%d_%d_%d.inp", N[0], N[1], N[2]);
  if(num_bytes > inp_filename_len){
    printf("Insufficient buffer size to store path to inputfile\n");
    exit(1);
  }
  fftw_sp_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
  get_sp_input_data(fft_data, fftw_sp_data, N, inp_fname);
#else
  printf("Obtaining Double Precision Data\n");
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
    printf("Computing %sFFT3d - %d time\n", inverse ? "inverse ":"", i+1);

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
    fpga_computetime += fpga_fft3d_sp_(!inverse, N, fft_data);
#else
    fpga_computetime += fpga_fft3d_dp_(!inverse, N, fft_data);
#endif
    double stop = getTimeinMilliSec();
    fpga_runtime += stop - start;

    printf("\nComputing FFTW\n");
#ifdef __FPGA_SP
    fftw_runtime = compute_sp_fftw(fftw_sp_data, N, inverse);
    printf("\nChecking Correctness\n");
    verify_sp_fft(fft_data, fftw_sp_data, N);
#else
    fftw_runtime = compute_dp_fftw(fftw_dp_data, N, inverse);
    printf("\nChecking Correctness\n");
    verify_dp_fft(fft_data, fftw_dp_data, N);
#endif
  }

  // Print performance metrics
  compute_metrics( fpga_runtime, fpga_computetime, fftw_runtime, iter, N);

  // Free the resources allocated
  printf("\nCleaning up\n\n");
  if(fft_data)
    free(fft_data);
  #ifdef __FPGA_SP
    fftwf_free(fftw_sp_data);
  #else
    fftw_free(fftw_dp_data);
  #endif 
  */
}

void print_config(int N, int dim, int iter, int inv, int sp){
  printf("\n------------------------------------------\n");
  printf("FFT Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = %d%s \n", N, dim == 1 ? "" : dim == 2 ? "^2" : "^3");
  printf("Precision          = %s \n",  sp==1 ? "Single": "Double");
  printf("Direction          = %s \n", inv ? "Backward":"Forward");
  printf("Placement          = In Place    \n");
  printf("Iterations         = %d \n", iter);
  printf("--------------------------------------------\n\n");
}