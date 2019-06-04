#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "fft3d_decl.h"

/*
static void fourier_transform_gold(int lognr_points, cmplex *data, bool inverse);
static void fourier_stage(int lognr_points, cmplex * data);
// int coord(int iteration, int i, int j, int N);
void check_correctness(int N, cmplex *h_verify, cmplex *h_outData);

// Reference Fourier Transform 3d
void fourier_transform_3d(int lognr_points, cmplex *h_verify, bool inverse){

  int N = (1 << lognr_points);
  cmplex *h_verify_tmp = (cmplex *)malloc(sizeof(cmplex) * N * N * N);

  struct timeval start, stop;
  double time = 0.0;
  gettimeofday(&start, NULL);

  // Run reference code
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
     fourier_transform_gold(lognr_points, h_verify + coord(i, j, 0, N), inverse);
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        //h_verify_tmp[coord(j, i)] = h_verify[coord(i, j)];
        h_verify_tmp[coord(i, k, j, N)] = h_verify[coord(i, j, k, N)];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
     fourier_transform_gold(lognr_points, h_verify_tmp + coord(i, j, 0, N), inverse);
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        //h_verify_tmp[coord(j, i)] = h_verify[coord(i, j)];
        h_verify[coord(i, k, j, N)] = h_verify_tmp[coord(i, j, k, N)];
      }
    }
  }

  FILE *fp;
  fp = fopen("test.txt", "w+");

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        //h_verify_tmp[coord(j, i)] = h_verify[coord(i, j)];
        fprintf(fp, "(%d) - %lf, %lf\n", coord(i,j,k), h_verify_tmp[coord(i,j,k)].x, h_verify_tmp[coord(i,j,k)].y);
      }
    }
  }
  fclose(fp);

  // 3D
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        //h_verify_tmp[coord(j, i)] = h_verify[coord(i, j)];
        h_verify_tmp[coord(k, j, i, N)] = h_verify[coord(i, j, k, N)];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
     fourier_transform_gold(lognr_points, h_verify + coord(i, j, 0, N), inverse);
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        //h_verify_tmp[coord(j, i)] = h_verify[coord(i, j)];
        h_verify[coord(k, j, i, N)] = h_verify_tmp[coord(i, j, k, N)];
      }
    }
  }
  
  gettimeofday(&stop, NULL);
  time = ((stop.tv_sec - start.tv_sec)* 1E6) + ((stop.tv_usec - start.tv_usec));

  free(h_verify_tmp);

  printf("\tmyFFT Processing time = %.4fms\n", time * 1E-3);
  double gpoints_per_sec = ( N * N * N / time) * 1E-9;
  double gflops = 3 * 5 * N * N * N * (log((double)N)/log((double)2))/(time * 1E6);
  printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);
}

// Reference Fourier transform
static void fourier_transform_gold(int lognr_points, cmplex *data, bool inverse){
   const int nr_points = (1 << lognr_points);

   // The inverse requires swapping the real and imaginary component
   
   if(inverse) {
      for (int i = 0; i < nr_points; i++) {
         decimal tmp = data[i].x;
         data[i].x = data[i].y;
         data[i].y = tmp;;
      }
   }
   // Do a FT recursively
   fourier_stage(lognr_points, data);

   // The inverse requires swapping the real and imaginary component
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         decimal tmp = data[i].x;
         data[i].x = data[i].y;
         data[i].y = tmp;;
      }
   }
}

static void fourier_stage(int lognr_points, cmplex *data) {
   int nr_points = 1 << lognr_points;
   if (nr_points == 1) return;
   cmplex *half1 = (cmplex *)alloca(sizeof(cmplex) * nr_points / 2);
   cmplex *half2 = (cmplex *)alloca(sizeof(cmplex) * nr_points / 2);
   for (int i = 0; i < nr_points / 2; i++) {
      half1[i] = data[2 * i];
      half2[i] = data[2 * i + 1];
   }
   fourier_stage(lognr_points - 1, half1);
   fourier_stage(lognr_points - 1, half2);
   for (int i = 0; i < nr_points / 2; i++) {
      data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
   }
}



void check_correctness(int N, cmplex *h_verify, cmplex *h_outData){

  int mangle;
  decimal mag_sum = 0;
  decimal noise_sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            int where = coord(i, j, k, N);
            //printf(" %lf %lf - %lf %lf\n", h_outData[where].x, h_outData[where].y, h_verify[where].x, h_verify[where].y);
            decimal magnitude = h_verify[where].x * h_verify[where].x + \
                              h_verify[where].y * h_verify[where].y;
            decimal noise = (h_verify[where].x - h_outData[where].x) \
                * (h_verify[where].x - h_outData[where].x) + 
                (h_verify[where].y - h_outData[where].y) * (h_verify[where].y - h_outData[where].y);

            mag_sum += magnitude;
            noise_sum += noise;
        }
    }
  }

  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, db > 120 ? "PASSED" : "FAILED");

}
*/