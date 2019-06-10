/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

/******************************************************************************
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 *****************************************************************************/
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0e3;
}

/******************************************************************************
 * \brief  compute the offset in the matrix based on the indices of dim given
 * \param  i, j, k - indices of different dimensions used to find the 
 *         coordinate in the matrix 
 * \param  N - fft size
 * \retval linear offset in the flattened 3d matrix
 *****************************************************************************/
unsigned int coord( unsigned int i, unsigned int j, unsigned int k, int N[3]) {
  // TODO : works only for uniform dims
  return i * N[0] * N[1] + j * N[2] + k;
}

/******************************************************************************
 * \brief  print time taken for fpga and fftw runs to a file
 * \param  fftw_time, fpga_time: double
 * \param  iter - number of iterations of each
 * \param  N - fft size
 * \param  fname - filename given through cmd line arg
 *****************************************************************************/
void print_output(double fftw_time, double fpga_time, int iter, int N[3], char* fname){
  char filename[] = "../outputfiles/";
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
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fpga_time * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2] * (log((double)N[0])/log((double)2))/(fpga_time * 1E-3 * 1E9);
    printf("\t  %d³ \t\t %.4f \t  %.4f \t\n", N[0], fpga_time, gflops);
    fprintf(fp, "%d,%.4f,%.4f\n", N[0], fpga_time, gflops);
  }
  else{
    printf("Error in FFT3d \n");
  }

#ifdef DEBUG
    printf("fftw:"); 
    fprintf(fp, "fftw,"); 
    if(fftw_time != 0.0){
      fftw_time = fftw_time / iter;
      double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_time * 1E-3)) * 1E-9;
      double gflops = 3 * 5 * N[0] * N[1] * N[2] * (log((double)N[0])/log((double)2))/(fftw_time * 1E-3 * 1E9);
      printf("\t  %d³ \t\t %.4f \t  %.4f \t\n", N[0], fftw_time, gflops);
      fprintf(fp, "%d,%.4f,%.4f\n", N[0], fftw_time, gflops);
    }
    else{
      printf("Error in FFT3d \n");
    }
#endif

  fclose(fp);
}