/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <sys/time.h>

// common dependencies
#include "CL/opencl.h"
#include "api/fft_fpga.h"  // Common declarations and API

// local dependencies
#include "common/argparse.h"  // Cmd-line Args to set some global vars

static void get_input_data(int N[3]);
static void print_output(int N[3], double fftw_time, double fpga_runtime, int iter, char *fname);
static void check_correctness_fftw(int N[3], cmplx *h_outData);
static double compute_fftw(int N[3], int inverse);
static void cleanup_fft();

static cmplx *fft_data, *fft_data_out, *fft_verify;

#ifdef __FPGA_SP
  fftwf_plan plan;
  fftwf_complex *fftw_data;
#else
  fftw_plan plan;
  fftw_complex *fftw_data;
#endif

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};
/******************************************************************************
 * \brief  compute the offset in the matrix based on the indices of dim given
 * \param  i, j, k - indices of different dimensions used to find the 
 *         coordinate in the matrix 
 * \param  N - fft size
 * \retval linear offset in the flattened 3d matrix
 *****************************************************************************/
unsigned coord(unsigned N[3], unsigned i, unsigned j, unsigned k) {
  // TODO : works only for uniform dims
  return i * N[0] * N[1] + j * N[2] + k;
}
/******************************************************************************
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 *****************************************************************************/
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0e3;
}
int main(int argc, const char **argv) {
  unsigned i = 0;
  double fpga_runtime = 0.0, fftw_time = 0.0;
  char *data_path = "../fft3d_kernels/fpgabitstream";

  // Cmd line argument declarations
  int N[3] = {64, 64, 64};
  unsigned iter = 1, inverse = 0;
  char *fname = NULL;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N[0], "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N[1], "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N[2], "FFT 3rd Dim Size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_BOOLEAN('b',"back", &inverse, "Backward/inverse FFT"),
    OPT_STRING('o',"output",&fname,"Output filename"), 
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
  printf("Allocating Memory ...\n");
  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  fft_data_out = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  fft_verify = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  #if __FPGA_SP
    fftw_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
  #else
    fftw_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);
  #endif

  // Initialize input and produce verification data
  get_input_data(N);

  // execute FFT3d iter number of times
  for( i = 0; i < iter; i++){
    printf("\nCalculating %sFFT3d - %d\n", inverse ? "inverse ":"", i);

    // initialize FPGA
    if (fpga_initialize_()){
      printf("Error initializing FPGA\n");
      // cleanup
    }

    // check if required bitstream exists
    if(!fpga_check_bitstream_(data_path, N)){
      printf("Bitstream not found\n");
    }

    // execute fpga fft3d
    double start = getTimeinMilliSec();
#if __FPGA_SP
    fpga_fft3d_sp_(inverse, N, fft_data);
#else
    fpga_fft3d_dp_(inverse, N, fft_data);
#endif
    double stop = getTimeinMilliSec();
    fpga_runtime += stop - start;

    // cleanup fpga
    printf("Cleanup up ... \n");

    printf("Computing FFTW\n");
    compute_fftw(N, inverse);
    check_correctness_fftw(N, fft_data);
  }

  // Free the resources allocated
  cleanup_fft();

  // Print performance metrics
  print_output(N, fftw_time, fpga_runtime, iter, fname);

  return 1;
}

static void get_input_data(int N[3]){
  unsigned i = 0, j = 0, k = 0, where = 0;
  const unsigned fname_len = 100;
  char fname[fname_len];

  #if __FPGA_SP
    sprintf(fname,"../inputfiles/input_f%d_%d_%d.inp", N[0], N[1], N[2]);
  #else
    sprintf(fname,"../inputfiles/input_d%d_%d_%d.inp", N[0], N[1], N[2]);
  #endif

  FILE *fp = fopen(fname,"r");
  if(fp != NULL){
      printf("Scanning data set from file - %s\n\n",fname);

      for (i = 0; i < N[0]; i++) {
        for (j = 0; j < N[1]; j++) {
          for ( k = 0; k < N[2]; k++) {
            where = coord(N, i, j, k);

            #ifdef __FPGA_SP
            fscanf(fp, "%f %f ", &fft_data[where].x, &fft_data[where].y);
            //printf("%f %f ", fft_data[where].x, fft_data[where].y);
            #else
            fscanf(fp, "%lf %lf ", &fft_data[where].x, &fft_data[where].y);
            #endif

            fftw_data[where][0] = fft_verify[where].x = fft_data[where].x;
            fftw_data[where][1] = fft_verify[where].y = fft_data[where].y;
            //printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
          }
        }
      }
  }
  else{
      printf("Data not available. Printing random data set to file %s\n",fname);

      fp = fopen(fname,"w");
      for (i = 0; i < N[0]; i++) {
        for (j = 0; j < N[1]; j++) {
          for ( k = 0; k < N[2]; k++) {
            where = coord(N, i, j, k);

          #ifdef __FPGA_SP
            fft_data[where].x = (float)((float)rand() / (float)RAND_MAX);
            fft_data[where].y = (float)((float)rand() / (float)RAND_MAX);
            fprintf(fp, "%f %f ", fft_data[where].x, fft_data[where].y);
          #else
            fft_data[where].x = (double)((double)rand() / (double)RAND_MAX);
            fft_data[where].y = (double)((double)rand() / (double)RAND_MAX);
            fprintf(fp, "%lf %lf ", fft_data[where].x, fft_data[where].y);
          #endif

            fftw_data[where][0] = fft_verify[where].x = fft_data[where].x;
            fftw_data[where][1] = fft_verify[where].y = fft_data[where].y;
            printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
          }
        }
      }
      fclose(fp);
  }
}

static void print_output(int N[3], double fftw_time, double fpga_runtime, int iter, char *fname){
  char filename[] = "../../outputfiles/";
  if(fname!= NULL){
      strcat(filename, fname);
  }
  else{
      printf("No output file entered\n");
      strcat(filename, "noname.csv");
  }

/*
  printf("Printing to %s\n", filename);

  FILE *fp = fopen(filename,"r");

  if(fp == NULL){
    fp = fopen(filename,"w");
    fprintf(fp,"device,N,runtime,throughput\n");
  }
  else{
    fp = fopen(filename,"a");
  }

  printf("\nNumber of runs: %d\n\n", iter);
  printf("\tFFT Size\tRuntime(ms)\tThroughput(GFLOPS/sec)\t\n");
  printf("fpga:");
  fprintf(fp, "fpga,");

  if(fpga_runtime != 0.0){
    fpga_runtime = fpga_runtime / iter;
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fpga_runtime * 1E-6)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2] * (log((double)N[0])/log((double)2))/(fpga_runtime * 1E-6 * 1E9);
    printf("\t  %d³ \t\t %.4f \t  %.4f \t\n", N[0], (fpga_runtime * 1E-3), gflops);
    fprintf(fp, "%d,%.4f,%.4f\n", N[0], (fpga_runtime * 1E-3), gflops);
  }
  else{
    printf("Error in FFT3d \n");
  }

    printf("fftw:"); 
    fprintf(fp, "fftw,"); 
    if(fftw_time != 0.0){
      fftw_time = fftw_time / iter;
      double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_time * 1E-6)) * 1E-9;
      double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(fftw_time * 1E-6 * 1E9);
      printf("\t  %d³ \t\t %.4f \t  %.4f \t\n", N[0], (fftw_time * 1E-3), gflops);
      fprintf(fp, "%d,%.4f,%.4f\n", N[0], (fftw_time * 1E-3), gflops);
    }
    else{
      printf("Error in FFT3d \n");
    }

  fclose(fp);
  */

}

static double compute_fftw(int N[3], int inverse){

#if __FPGA_SP
  printf("\nPlanning %sSingle precision FFTW ... \n", inverse ? "inverse ":"");
  if(inverse){
      plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }
  printf("Computing Single Precision FFTW\n");
  double start = getTimeinMilliSec();
  fftwf_execute(plan);
  double stop = getTimeinMilliSec();

#else
  printf("\nPlanning %sDouble precision FFTW ... \n", inverse ? "inverse ":"");
  if(inverse){
      plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  printf("Computing Double Precision FFTW\n");
  double start = getTimeinMilliSec();
  fftw_execute(plan);
  double stop = getTimeinMilliSec();

#endif
  return (stop - start);
}

static void check_correctness_fftw(int N[3], cmplx *h_outData){
  unsigned where, i = 0, j = 0, k = 0;
  #if __FPGA_SP
    float mag_sum = 0, noise_sum = 0, magnitude, noise;
  #else
    double mag_sum = 0, noise_sum = 0, magnitude, noise;
  #endif

  for (i = 0; i < N[0]; i++) {
    for (j = 0; j < N[1]; j++) {
        for (k = 0; k < N[2]; k++) {
            where = coord(N, i, j, k);
            // printf(" %d : %lf %lf - %lf %lf\n", where, h_outData[where].x, h_outData[where].y, fftw_data_out[where][0], fftw_data_out[where][1]);
            magnitude = fftw_data[where][0] - fft_data[where].x;
            noise = fftw_data[where][1] - fft_data[where].y;
            printf(" %d : fpga - (%e %e)  cpu - (%e %e) diff - (%e %e) \n", where, fft_data[where].x, fft_data[where].y, fftw_data[where][0], fftw_data[where][1], magnitude, noise);

/*
            decimal magnitude = fftw_data[where][0] * fftw_data[where][0] + \
                              fftw_data[where][1] * fftw_data[where][1];
            decimal noise = (fftw_data[where][0] - fft_data[where].x) \
                * (fftw_data[where][0] - fft_data[where].x) + 
                (fftw_data[where][1] - fft_data[where].y) * (fftw_data[where][1] - fft_data[where].y);
            mag_sum += magnitude;
            noise_sum += noise;
*/
        }
    }
  }

  //float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  //printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");
  //mag_sum = mag_sum / (N * N * N);
  //noise_sum = noise_sum / (N * N * N);
  //printf("\t Average difference : %e, %e \n\n", mag_sum, noise_sum);
}

static void cleanup_fft(){
  if(fft_data)
    free(fft_data);

  if(fft_data_out)
    free(fft_data_out);

  if(fft_verify)
    free(fft_verify);

  #ifdef __FPGA_SP
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_data);
  #else
    fftw_destroy_plan(plan);
    fftw_free(fftw_data);
  #endif
}
