#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <sys/time.h>
#include <fftw3.h>
#include <string.h>

#include "fft3d.h"
#include "fft_decl.h"
#include "test_fft.h"
#include "argparse.h"

static void allocate_mem(int N);
static void get_input_data(int N);
static void print_output(int N, double fftw_time, double runtime, int iter, char *fname);
static void check_correctness_fftw(int N, cmplex *h_outData);
static double compute_fftw(int N, bool inverse);
static void cleanup_fft();

static cmplex *fft_data, *fft_data_out, *fft_verify;

#if TYPE_FLOAT == 1
  fftwf_plan plan;
  fftwf_complex *fftw_data, *fftw_data_out;
#else
  fftw_plan plan;
  fftw_complex *fftw_data, *fftw_data_out;
#endif

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

int main(int argc, const char **argv) {
  int iter = 1;
  int lognr = 6;
  int N = ( 1 << LOGN );
  double runtime = 0.0, fftw_time = 0.0;
  bool inverse = false;
  char *fname = NULL;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"nsize", &N, "FFT size"),
    OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
    OPT_BOOLEAN('b',"back", &inverse, "Backward/inverse FFT"),
    OPT_STRING('o',"output",&fname,"Output filename"), 
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Computing FFT3d using FPGA", "FFT size is mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Allocate mem for input buffer and fftw buffer
  allocate_mem(N);

  // Initialize input and produce verification data
  get_input_data(N);

  // execute FFT3d iter number of times
  for( int i = 0; i < iter; i++){
    printf("\nCalculating %sFFT3d - %d\n", inverse ? "inverse ":"", i);
    runtime+= fft3d(N, fft_data, fft_data_out, inverse);

    // Verify data
    /*
    fourier_transform_3d(lognr, fft_verify, inverse);
    check_correctness(N, fft_verify, fft_data);
    */

#ifdef DEBUG
    fftw_time+= compute_fftw(N, inverse);
    check_correctness_fftw(N, fft_data_out);
#endif
  }

  // Free the resources allocated
  cleanup_fft();

  // Print performance metrics
  print_output(N, fftw_time, runtime, iter, fname);

  return 1;
}

static void allocate_mem(int N){
  fft_data = (cmplex *)malloc(sizeof(cmplex) * N * N * N);
  fft_data_out = (cmplex *)malloc(sizeof(cmplex) * N * N * N);
  fft_verify = (cmplex *)malloc(sizeof(cmplex) * N * N * N);

  #if TYPE_FLOAT == 1
    fftw_data = (fftwf_complex* )fftwf_malloc(sizeof(fftwf_complex) * N * N * N);
    fftw_data_out = (fftwf_complex* )fftwf_malloc(sizeof(fftwf_complex) * N * N * N);
  #else
    fftw_data = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * N * N * N);
    fftw_data_out = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * N * N * N);
  #endif

  printf("\n\tMemory allocated..\n\n");

}

static void get_input_data(int N){

  char fname[50];

  #if TYPE_FLOAT == 1
  sprintf(fname,"../inputfiles/input_f%d.inp", N);
  #else
  sprintf(fname,"../inputfiles/input_d%d.inp", N);
  #endif

  FILE *fp = fopen(fname,"r");

  if(fp != NULL){
      printf("\tScanning data set from file %s ..\n\n",fname);

      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          for ( int k = 0; k < N; k++) {
            int where = coord(i, j, k, N);

            #if TYPE_FLOAT == 1
            fscanf(fp, "%f %f ", &fft_data[where].x, &fft_data[where].y);
            #else
            fscanf(fp, "%lf %lf ", &fft_data[where].x, &fft_data[where].y);
            #endif

            fftw_data[coord(i, j, k, N)][0] = fft_verify[coord(i, j, k, N)].x = fft_data[where].x;
            fftw_data[coord(i, j, k, N)][1] = fft_verify[coord(i, j, k, N)].y = fft_data[where].y;
            //printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
          }
        }
      }

  }else{

      printf("\tData not available. Printing random data set to file %s ..\n\n",fname);

      fp = fopen(fname,"w");

      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          for ( int k = 0; k < N; k++) {
            int where = coord(i, j, k, N);

            fft_data[where].x = (decimal)((decimal)rand() / (decimal)RAND_MAX);
            fft_data[where].y = (decimal)((decimal)rand() / (decimal)RAND_MAX);

            #if TYPE_FLOAT == 1
            fprintf(fp, "%f %f ", fft_data[where].x, fft_data[where].y);
            #else
            fprintf(fp, "%lf %lf ", fft_data[where].x, fft_data[where].y);
            #endif

            fftw_data[coord(i, j, k, N)][0] = fft_verify[coord(i, j, k, N)].x = fft_data[where].x;
            fftw_data[coord(i, j, k, N)][1] = fft_verify[coord(i, j, k, N)].y = fft_data[where].y;
            // printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, h_inData[where].x, h_inData[where].y, where, fftw_data[where][0], fftw_data[where][1]);
          }
        }
      }
      fclose(fp);
  }
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

static double compute_fftw(int N, bool inverse){
  time_t current_time;
  struct timeval start, stop;
  double time = 0.0;

#if TYPE_FLOAT == 1
  printf("\nComputing single precision FFTW ... \n\n");
  if(inverse){
      plan = fftwf_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data_out[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftwf_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data_out[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  gettimeofday(&start, NULL);
  fftwf_execute(plan);
  gettimeofday(&stop, NULL);

  time = ((stop.tv_sec - start.tv_sec) * 1E6) + (stop.tv_usec - start.tv_usec);

#else
  printf("\nComputing double precision FFTW ... \n\n");
  if(inverse){
      plan = fftw_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data_out[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftw_plan_dft_3d( N, N, N, &fftw_data[0], &fftw_data_out[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  gettimeofday(&start, NULL);
  fftw_execute(plan);
  gettimeofday(&stop, NULL);

  time = ((stop.tv_sec - start.tv_sec) * 1E6) + (stop.tv_usec - start.tv_usec);
#endif
  return time;
}

static void check_correctness_fftw(int N, cmplex *h_outData){

  decimal mag_sum = 0;
  decimal noise_sum = 0;
  int where;
  decimal magnitude, noise;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            where = coord(i, j, k, N);
            // printf(" %d : %lf %lf - %lf %lf\n", where, h_outData[where].x, h_outData[where].y, fftw_data_out[where][0], fftw_data_out[where][1]);
            magnitude = fftw_data_out[where][0] - h_outData[where].x;
            noise = fftw_data_out[where][1] - h_outData[where].y;
            printf(" %d : fpga - (%e %e)  cpu - (%e %e) diff - (%e %e) \n", where, h_outData[where].x, h_outData[where].y, fftw_data_out[where][0], fftw_data_out[where][1], magnitude, noise);

            /*
            decimal magnitude = fftw_data_out[where][0] * fftw_data_out[where][0] + \
                              fftw_data_out[where][1] * fftw_data_out[where][1];
            decimal noise = (fftw_data_out[where][0] - h_outData[where].x) \
                * (fftw_data_out[where][0] - h_outData[where].x) + 
                (fftw_data_out[where][1] - h_outData[where].y) * (fftw_data_out[where][1] - h_outData[where].y);

            mag_sum += magnitude;
            noise_sum += noise;
            */
        }
    }
  }

  //float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  //printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");
  /*
  mag_sum = mag_sum / (N * N * N);
  noise_sum = noise_sum / (N * N * N);
  printf("\t Average difference : %e, %e \n\n", mag_sum, noise_sum);
  */

}

static void cleanup_fft(){

  if(fft_data)
    free(fft_data);

  if(fft_data_out)
    free(fft_data_out);

  if(fft_verify)
    free(fft_verify);

  #if TYPE_FLOAT == 1
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_data);
    fftwf_free(fftw_data_out);
  #else
    fftw_destroy_plan(plan);
    fftw_free(fftw_data);
    fftw_free(fftw_data_out);
  #endif
}

