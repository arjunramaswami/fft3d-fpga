/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <fftw3.h>
#include <sys/time.h>

// common dependencies
#include "fft3d_decl.h"  // Common declarations

// local dependencies
#include "fft3d_api.h"   // APIs 
#include "argparse.h"    // Cmd-line Args to set some global vars

// global variables
static cmplx *fft_data, *fft_verify;

unsigned int iter = 1;             // number of iterations of FFT3d using same input
int N[3] = {64, 64, 64};  // Size of FFT3d
bool inverse = false;              // Default as forward FFT

double runtime = 0.0, fftw_time = 0.0;
char *fname = NULL;

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

#if __FPGA_SP
  fftwf_plan plan;
  fftwf_complex *fftw_data;
#else
  fftw_plan plan;
  fftw_complex *fftw_data;
#endif


// function prototypes
static void allocate_mem();
static void get_input_data();

//static void print_output(int N, double fftw_time, double runtime, int iter, char *fname);
static void check_correctness_fftw(cmplx *h_outData);
static double compute_fftw(bool inverse);
static void cleanup_fft();

// provides a linear offset in the input array
int coord(int i, int j, int k) {
  return i * N[0] * N[1] + j * N[2] + k;
}

int main(int argc, const char **argv) {
  int N1 = 64, N2 = 64, N3 = 64;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('m',"n1", &N1, "FFT 1st Dim Size"),
    OPT_INTEGER('n',"n2", &N2, "FFT 2nd Dim Size"),
    OPT_INTEGER('p',"n3", &N3, "FFT 3rd Dim Size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_BOOLEAN('b',"back", &inverse, "Backward/inverse FFT"),
    OPT_STRING('o',"output",&fname,"Output filename"), 
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FPGA", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  N[0] = N1;
  N[1] = N2;
  N[2] = N3;

  // Allocate mem for input buffer and fftw buffer
  printf("\nAllocating (%d, %d, %d) memory for FFT3d on the host ... \n\n", N[0],  N[1],  N[2]);
  allocate_mem();

  // Initialize input and produce verification data
  printf("Getting input data ... \n\n");
  get_input_data();

  // execute FFT3d iter number of times
  for( int i = 0; i < iter; i++){
    printf("\n Iteration %d : Calculating %sFFT3d of size (%d, %d, %d) \n\n", i, inverse ? "inverse ":"", N[0], N[1], N[2]);

    if (!fpga_initialize_()){
      printf("Error initializing FPGA\n");
      // cleanup
    }

    if(!fpga_check_bitstream_(N)){
      printf("Bitstream not found\n");
    }
    //runtime+= fft3d(N, fft_data, fft_data_out, inverse);
    fpga_fft3d_sp_(0, "", inverse, N, fft_data);

    printf("Cleanup up ... \n");
    fpga_final_();

    // Verify data
    /*
    fourier_transform_3d(lognr, fft_verify, inverse);
    check_correctness(N, fft_verify, fft_data);
    */

#ifdef DEBUG
    fftw_time+= compute_fftw(inverse);
    check_correctness_fftw(fft_data);
#endif
  }

  // Free the resources allocated
  cleanup_fft();

  // Print performance metrics
  // print_output(fftw_time, runtime, iter, fname);

  return 1;
}

static void allocate_mem(){

  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  fft_verify = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);

  #if __FPGA_SP
    fftw_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
  #else
    fftw_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);
  #endif
}

static void get_input_data(){

  char fname[50];

  #if __FPGA_SP
    sprintf(fname,"../inputfiles/input_f%d_%d_%d.inp", N[0], N[1], N[2]);
  #else
    sprintf(fname,"../inputfiles/input_d%d_%d_%d.inp", N[0], N[1], N[2]);
  #endif

  FILE *fp = fopen(fname,"r");
  if(fp != NULL){
      printf("\tScanning data set from file %s ..\n\n",fname);

      unsigned int where;
      for (int i = 0; i < N[0]; i++) {
        for (int j = 0; j < N[1]; j++) {
          for (int k = 0; k < N[2]; k++) {
            where = coord(i, j, k);

            #if __FPGA_SP
              fscanf(fp, "%f %f ", &fft_data[where].x, &fft_data[where].y);
            #else
              fscanf(fp, "%lf %lf ", &fft_data[where].x, &fft_data[where].y);
            #endif

            fftw_data[coord(i, j, k)][0] = fft_verify[coord(i, j, k)].x = fft_data[where].x;
            fftw_data[coord(i, j, k)][1] = fft_verify[coord(i, j, k)].y = fft_data[where].y;
            //printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
            //printf(" %d %d %d - input[%d] = (%lf, %lf) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where);
          }
        }
      }

  }else{
      printf("\tData not available. Printing random data set to file %s ..\n\n",fname);
      fp = fopen(fname,"w");

      unsigned int where;
      for (int i = 0; i < N[0]; i++) {
        for (int j = 0; j < N[1]; j++) {
          for (int k = 0; k < N[2]; k++) {
            where = coord(i, j, k);

#if __FPGA_SP 
            fft_data[where].x = (float)((float)rand() / (float)RAND_MAX);
            fft_data[where].y = (float)((float)rand() / (float)RAND_MAX);
            fprintf(fp, "%f %f ", fft_data[where].x, fft_data[where].y);
#else
            fft_data[where].x = (double)((double)rand() / (double)RAND_MAX);
            fft_data[where].y = (double)((double)rand() / (double)RAND_MAX);
            fprintf(fp, "%lf %lf ", fft_data[where].x, fft_data[where].y);
#endif
            fftw_data[coord(i, j, k)][0] = fft_verify[coord(i, j, k)].x = fft_data[where].x;
            fftw_data[coord(i, j, k)][1] = fft_verify[coord(i, j, k)].y = fft_data[where].y;
            // printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, h_inData[where].x, h_inData[where].y, where, fftw_data[where][0], fftw_data[where][1]);
            //printf(" %d %d %d - input[%d] = (%lf, %lf) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where);
          }
        }
      }
      fclose(fp);
  }
}

static void cleanup_fft(){

  if(fft_data)
    free(fft_data);

  if(fft_verify)
    free(fft_verify);

  #if __FPGA_SP
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_data);
  #else
    fftw_destroy_plan(plan);
    fftw_free(fftw_data);
   #endif
}
static void print_output(int N, double fftw_time, double fpga_time, int iter, char *fname){
  char filename[] = "../../outputfiles/";
  if(fname!= NULL){
      strcat(filename, fname);
  }
  else{
      printf("No output file entered.\n");
      strcat(filename, "noname.csv");
  }

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

  if(fpga_time != 0.0){
    fpga_time = fpga_time / iter;
    double gpoints_per_sec = ( N * N * N / (fpga_time * 1E-6)) * 1E-9;
    double gflops = 3 * 5 * N * N * N * (log((double)N)/log((double)2))/(fpga_time * 1E-6 * 1E9);
    printf("\t  %d³ \t\t %.4f \t  %.4f \t\n", N, (fpga_time * 1E-3), gflops);
    fprintf(fp, "%d,%.4f,%.4f\n", N, (fpga_time * 1E-3), gflops);
  }
  else{
    printf("Error in FFT3d \n");
  }

#ifdef DEBUG
    printf("fftw:"); 
    fprintf(fp, "fftw,"); 
    if(fftw_time != 0.0){
      fftw_time = fftw_time / iter;
      double gpoints_per_sec = ( N * N * N / (fftw_time * 1E-6)) * 1E-9;
      double gflops = 3 * 5 * N * N * N * (log((double)N)/log((double)2))/(fftw_time * 1E-6 * 1E9);
      printf("\t  %d³ \t\t %.4f \t  %.4f \t\n", N, (fftw_time * 1E-3), gflops);
      fprintf(fp, "%d,%.4f,%.4f\n", N, (fftw_time * 1E-3), gflops);
    }
    else{
      printf("Error in FFT3d \n");
    }
#endif

  fclose(fp);
}

static double compute_fftw(bool inverse){
  time_t current_time;
  struct timeval start, stop;
  double time = 0.0;

#if __FPGA_SP
  printf("\nComputing single precision FFTW ... \n\n");
  if(inverse){
      plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  //gettimeofday(&start, NULL);
  fftwf_execute(plan);
  //gettimeofday(&stop, NULL);

  //time = ((stop.tv_sec - start.tv_sec) * 1E6) + (stop.tv_usec - start.tv_usec);

#else
  printf("\nPlanning double precision FFTW ... \n\n");
  if(inverse){
      plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  //gettimeofday(&start, NULL);

  printf("\nComputing FFTW ... \n\n");
  fftw_execute(plan);
  //gettimeofday(&stop, NULL);

  //time = ((stop.tv_sec - start.tv_sec) * 1E6) + (stop.tv_usec - start.tv_usec);
#endif
  //return time;
  return 1;
}

static void check_correctness_fftw(cmplx *h_outData){

  #if __FPGA_SP
    float mag_sum = 0, noise_sum = 0, magnitude, noise;
  #else
    double mag_sum = 0, noise_sum = 0, magnitude, noise;
  #endif

  unsigned int where;
  for (int i = 0; i < N[0]; i++) {
    for (int j = 0; j < N[1]; j++) {
        for (int k = 0; k < N[2]; k++) {
            where = coord(i, j, k);
            // printf(" %d : %lf %lf - %lf %lf\n", where, h_outData[where].x, h_outData[where].y, fftw_data_out[where][0], fftw_data_out[where][1]);
            magnitude = fftw_data[where][0] - h_outData[where].x;
            noise = fftw_data[where][1] - h_outData[where].y;
            printf(" %d : fpga - (%e %e)  cpu - (%e %e) diff - (%e %e) \n", where, h_outData[where].x, h_outData[where].y, fftw_data[where][0], fftw_data[where][1], magnitude, noise);

/*
            decimal magnitude = fftw_data[where][0] * fftw_data[where][0] + \
                              fftw_data[where][1] * fftw_data[where][1];
            decimal noise = (fftw_data[where][0] - h_outData[where].x) \
                * (fftw_data[where][0] - h_outData[where].x) + 
                (fftw_data[where][1] - h_outData[where].y) * (fftw_data[where][1] - h_outData[where].y);

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
