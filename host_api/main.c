/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef DEBUG
#include <fftw3.h>
#endif

// common dependencies
#include "fpga/fft3d_decl.h"  // Common declarations

// local dependencies
#include "fpga/fft3d_api.h"   // APIs 
#include "common/argparse.h"  // Cmd-line Args to set some global vars
#include "common/helper.h"    // helper functions

// global variables
static cmplx *fft_data, *fft_verify;
unsigned int iter = 1;    // number of iterations of FFT3d using same input
int N[3] = {64, 64, 64};  // Size of FFT3d
bool inverse = false;     // Default as forward FFT
char *fname = NULL;

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

#ifdef DEBUG
  #if __FPGA_SP
    fftwf_plan plan;
    fftwf_complex *fftw_data;
  #else
    fftw_plan plan;
    fftw_complex *fftw_data;
  #endif

  // function declarations for FFTW computation enabled using DEBUG flag on compilation
  static double compute_fftw();
  static void verify_fftw();
#endif  //DEBUG

// function declarations
static void allocate_mem();
static void get_input_data();
static void cleanup();



// --- CODE -------------------------------------------------------------------

int main(int argc, const char **argv) {
  int N1 = 64, N2 = 64, N3 = 64;
  double fpga_time = 0.0, fftw_time = 0.0;

  // get cmd line params and save to global vars
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

    // initialize FPGA
    if (fpga_initialize_()){
      printf("Error initializing FPGA\n");
      // cleanup
    }

    // check if required bitstream exists
    if(!fpga_check_bitstream_(N)){
      printf("Bitstream not found\n");
    }

    // execute fpga fft3d
    double start = getTimeinMilliSec();
#if __FPGA_SP
    fpga_fft3d_sp_(0, "", inverse, fft_data);
#else
    fpga_fft3d_dp_(0, "", inverse, fft_data);
#endif
    double stop = getTimeinMilliSec();
    fpga_time += stop - start;

    // cleanup fpga
    printf("Cleanup up ... \n");
    fpga_final_();

#ifdef DEBUG
    // Verify fpga fft3d with fftw fft3d
    fftw_time+= compute_fftw();
    verify_fftw();
#endif
  }

  // Free the resources allocated
  cleanup();

  // Print performance metrics
  print_output(fftw_time, fpga_time, iter, N, fname);

  return 1;
}

/******************************************************************************
 * \brief  allocate memory for execution and verification
 *****************************************************************************/
static void allocate_mem(){
  fft_data = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  fft_verify = (cmplx *)malloc(sizeof(cmplx) * N[0] * N[1] * N[2]);

  
#ifdef DEBUG
  #if __FPGA_SP
    fftw_data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N[0] * N[1] * N[2]);
  #else
    fftw_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);
  #endif
#endif

}

 /******************************************************************************
 * \brief  create random values for FFT computation or read existing ones if 
 *         already saved in a file
 *****************************************************************************/
static void get_input_data(){

  char fname[50];
  unsigned int where;

  // TODO : Location of input and output files
  #if __FPGA_SP
    sprintf(fname,"../inputfiles/input_f%d_%d_%d.inp", N[0], N[1], N[2]);
  #else
    sprintf(fname,"../inputfiles/input_d%d_%d_%d.inp", N[0], N[1], N[2]);
  #endif

  // if file already present, read from it
  FILE *fp = fopen(fname,"r");
  if(fp != NULL){
      printf("Scanning data set from file %s ..\n\n",fname);

      for (unsigned int i = 0; i < N[0]; i++) {
        for (unsigned int j = 0; j < N[1]; j++) {
          for (unsigned int k = 0; k < N[2]; k++) {
            where = coord(i, j, k, N);

            #if __FPGA_SP
              fscanf(fp, "%f %f ", &fft_data[where].x, &fft_data[where].y);
            #else
              fscanf(fp, "%lf %lf ", &fft_data[where].x, &fft_data[where].y);
            #endif

#ifdef DEBUG
            fftw_data[coord(i, j, k, N)][0] = fft_verify[coord(i, j, k, N)].x = fft_data[where].x;
            fftw_data[coord(i, j, k, N)][1] = fft_verify[coord(i, j, k, N)].y = fft_data[where].y;
#endif            
            // TODO : debug to print to a file
            //printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where, fftw_data[where][0], fftw_data[where][1]);
            //printf(" %d %d %d - input[%d] = (%lf, %lf) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where);
          }
        }
      }
  } else{
      printf("\tData not available. Printing random data set to file %s ..\n\n",fname);

      // Create random data and print to a file to be reused 
      fp = fopen(fname,"w");
      for (unsigned int i = 0; i < N[0]; i++) {
        for (unsigned int j = 0; j < N[1]; j++) {
          for (unsigned int k = 0; k < N[2]; k++) {
            where = coord(i, j, k, N);

#if __FPGA_SP 
            fft_data[where].x = (float)((float)rand() / (float)RAND_MAX);
            fft_data[where].y = (float)((float)rand() / (float)RAND_MAX);
            fprintf(fp, "%f %f ", fft_data[where].x, fft_data[where].y);
#else
            fft_data[where].x = (double)((double)rand() / (double)RAND_MAX);
            fft_data[where].y = (double)((double)rand() / (double)RAND_MAX);
            fprintf(fp, "%lf %lf ", fft_data[where].x, fft_data[where].y);
#endif
#ifdef DEBUG
            fftw_data[coord(i, j, k, N)][0] = fft_verify[coord(i, j, k, N)].x = fft_data[where].x;
            fftw_data[coord(i, j, k, N)][1] = fft_verify[coord(i, j, k, N)].y = fft_data[where].y;
#endif
            // TODO: Debug prints
            // printf(" %d %d %d - input[%d] = (%f, %f) fftw[%d] = (%f, %f) \n", i, j, k, where, h_inData[where].x, h_inData[where].y, where, fftw_data[where][0], fftw_data[where][1]);
            //printf(" %d %d %d - input[%d] = (%lf, %lf) \n", i, j, k, where, fft_data[where].x, fft_data[where].y, where);
          }
        }
      }

      fclose(fp);
  }
}

/******************************************************************************
 * \brief free memory allocated for computation and verification using FFTW
 *****************************************************************************/
static void cleanup(){
  if(fft_data)
    free(fft_data);

  if(fft_verify)
    free(fft_verify);

#ifdef DEBUG
  #if __FPGA_SP
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_data);
  #else
    fftw_destroy_plan(plan);
    fftw_free(fftw_data);
   #endif
#endif
}
#ifdef DEBUG
/******************************************************************************
 * \brief  compute FFTW 
 * \retval double : time taken to compute FFTW
 *****************************************************************************/
static double compute_fftw(){


#if __FPGA_SP
  printf("\nPlanning %ssingle precision FFTW ... \n", inverse ? "inverse ":"");
  if(inverse){
      plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftwf_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }
  double start = getTimeinMilliSec();
  fftwf_execute(plan);
  double stop = getTimeinMilliSec();

#else
  printf("\nPlanning %sdouble precision FFTW ... \n", inverse ? "inverse ":"");
  if(inverse){
      plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
      plan = fftw_plan_dft_3d( N[0], N[1], N[2], &fftw_data[0], &fftw_data[0], FFTW_FORWARD, FFTW_ESTIMATE);
  }

  printf("\nComputing FFTW ... \n\n");

  double start = getTimeinMilliSec();
  fftw_execute(plan);
  double stop = getTimeinMilliSec();

#endif
  return (stop - start);
}

/******************************************************************************
 * \brief  verify computed fft3d with FFTW fft3d
 *****************************************************************************/
static void verify_fftw(){
  unsigned int where;
  #if __FPGA_SP
    float mag_sum = 0, noise_sum = 0, magnitude, noise;
  #else
    double mag_sum = 0, noise_sum = 0, magnitude, noise;
  #endif

  for (unsigned int i = 0; i < N[0]; i++) {
    for (unsigned int j = 0; j < N[1]; j++) {
        for (unsigned int k = 0; k < N[2]; k++) {
            where = coord(i, j, k, N);
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
#endif