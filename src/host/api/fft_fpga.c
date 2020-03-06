/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fftw3.h>

// common dependencies
#include "CL/opencl.h"
#include "../common/opencl_utils.h"
#include "fft_api.h"
#include "helper.h"
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA

// host variables
#ifndef KERNEL_VARS
#define KERNEL_VARS
static cl_platform_id platform = NULL;
static cl_device_id *devices;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;
static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
static cl_command_queue queue4 = NULL, queue5 = NULL, queue6 = NULL;
#endif

// Function prototypes
int init();
void cleanup();
static void cleanup_program();
static void init_program(int N[3], char *data_path);
static void queue_setup();
void queue_cleanup();
static double fftfpga_run_3d(int inverse, int N[3], cmplx *c_in);

// --- CODE -------------------------------------------------------------------

int fpga_initialize_(){
   return init();
}

void fpga_final_(){
   cleanup();
}

/******************************************************************************
 * \brief  check whether FFT3d can be computed on the FPGA or not. This depends 
 *         on the availability of bitstreams whose sizes are for now listed here 
 *         If the fft sizes are found and the FPGA is not setup before, it is done
 * \param  data_path - path to the data directory
 * \param  N - integer pointer to the size of the FFT3d
 * \retval true if fft3d size supported
 *****************************************************************************/
int fpga_check_bitstream_(char *data_path, int N[3]){
    static int fft_size[3] = { 0, 0, 0};

    // check the supported sizes
    if( (N[0] == 16 && N[1] == 16 && N[2] == 16) ||
        (N[0] == 32 && N[1] == 32 && N[2] == 32) ||
        (N[0] == 64 && N[1] == 64 && N[2] == 64)  ){

        // if first time
        if( fft_size[0] == 0 && fft_size[1] == 0 && fft_size[2] == 0 ){
          fft_size[0] = N[0];
          fft_size[1] = N[1];
          fft_size[2] = N[2];

          init_program(fft_size, data_path);
        }
        else if( fft_size[0] == N[0] && fft_size[1] == N[1] && fft_size[2] == N[2] ){
          // if same fft size as previous
          // dont do anything
        }
        else{
            // else if different fft size as previous
            // cleanup and initialize
          fft_size[0] = N[0];
          fft_size[1] = N[1];
          fft_size[2] = N[2];

          cleanup_program();
          init_program(fft_size, data_path);
        }

        return 1;
    }
    else{
        return 0;
    } 
}

/******************************************************************************
 * \brief   compute an in-place single precision complex 3D-FFT on the FPGA
 * \param   direction : direction - 1/forward, otherwise/backward FFT3d
 * \param   N   : integer pointer to size of FFT3d  
 * \param   din : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
double fpga_fft3d_sp_(int direction, int N[3], cmplx *din) {
  // setup device specific constructs 
  if(direction == 1){
    return fftfpga_run_3d(0, N, din);
  }
  else{
    return fftfpga_run_3d(1, N, din);
  }
}

/******************************************************************************
 * \brief   compute an in-place double precision complex 3D-FFT on the FPGA
 * \param   direction : direction - 1/forward, otherwise/backward FFT3d
 * \param   N   : integer pointer to size of FFT3d  
 * \param   din : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
double fpga_fft3d_dp_(int direction, int N[3], cmplx *din) {
  // setup device specific constructs 
  if(direction == 1){
    return fftfpga_run_3d(0, N, din);
  }
  else{
    return fftfpga_run_3d(1, N, din);
  }
}

/******************************************************************************
 * \brief   Execute a single precision complex FFT3d
 * \param   inverse : int
 * \param   N       : integer pointer to size of FFT3d  
 * \param   din     : complex input/output single precision data pointer 
 * \retval double : time taken for FFT3d compute 
 *****************************************************************************/
static double fftfpga_run_3d(int inverse, int N[3], cmplx *c_in) {
  cl_int status = 0;
  int inverse_int = inverse;
  cl_kernel fft_kernel = NULL, fft_kernel_2 = NULL;
  cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, transpose_kernel_2 = NULL;
  
// Device memory buffers
  cl_mem d_inData, d_outData;

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  fft_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create fft3da kernel");
  fft_kernel_2 = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create fft3db kernel");
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create transpose kernel");
  transpose_kernel_2 = clCreateKernel(program, "transpose3d", &status);
  checkError(status, "Failed to create transpose3d kernel");

  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplx) * N[0] * N[1] * N[2], NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplx) * N[0] * N[1] * N[2], NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  cmplx *h_inData = (cmplx *)alignedMalloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  if (h_inData == NULL){
    printf("Unable to allocate host memory\n");
    exit(1);
  }
  cmplx *h_outData = (cmplx *)alignedMalloc(sizeof(cmplx) * N[0] * N[1] * N[2]);
  if (h_outData == NULL){
    printf("Unable to allocate host memory\n");
    exit(1);
  }

  memcpy(h_inData, c_in, sizeof(cmplx) * N[0] * N[1] * N[2]);

  queue_setup();

  // Copy data from host to device
  double pcie_wr_start = getTimeinMilliSec();
  status = clEnqueueWriteBuffer(queue6, d_inData, CL_TRUE, 0, sizeof(cmplx) * N[0] * N[1] * N[2], h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");
  double pcie_wr_time = getTimeinMilliSec() - pcie_wr_start;
  status = clFinish(queue6);
  checkError(status, "failed to finish");
  printf("PCIE_Write Time of %d points totalling % bytes %lf\n", pcie_wr_time, N[0] * N[1] * N[2], sizeof(cmplx) * N[0] * N[1] * N[2]);

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 2");
  status = clSetKernelArg(fft_kernel_2, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 3");

  double start = getTimeinMilliSec();
  status = clEnqueueTask(queue1, fetch_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  // Launch the fft kernel - we launch a single work item hence enqueue a task
  status = clEnqueueTask(queue2, fft_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fft kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");

  status = clEnqueueTask(queue4, fft_kernel_2, 0, NULL, NULL);
  checkError(status, "Failed to launch second fft kernel");

  status = clEnqueueTask(queue5, transpose_kernel_2, 0, NULL, NULL);
  checkError(status, "Failed to launch second transpose kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue1);
  checkError(status, "failed to finish");
  status = clFinish(queue2);
  checkError(status, "failed to finish");
  status = clFinish(queue3);
  checkError(status, "failed to finish");
  status = clFinish(queue4);
  checkError(status, "failed to finish");
  status = clFinish(queue5);
  checkError(status, "failed to finish");

  double stop = getTimeinMilliSec();
  double fpga_runtime = stop - start;
   
  // Copy results from device to host
  double pcie_rd_start = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue3, d_outData, CL_TRUE, 0, sizeof(cmplx) * N[0] * N[1] * N[2], h_outData, 0, NULL, NULL);
  checkError(status, "Failed to read data from device");
  double pcie_rd_time = getTimeinMilliSec() - pcie_rd_start;

  printf("PCIE_Read Time of %d points totalling % bytes %lf\n", pcie_rd_time, N[0] * N[1] * N[2], sizeof(cmplx) * N[0] * N[1] * N[2]);

  memcpy(c_in, h_outData, sizeof(cmplx) * N[0] * N[1] * N[2] );

  queue_cleanup();

  if (h_outData)
	  free(h_outData);
  if (h_inData)
	  free(h_inData);

  if (d_inData)
  	clReleaseMemObject(d_inData);
  if (d_outData) 
	  clReleaseMemObject(d_outData);

  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(fft_kernel) 
    clReleaseKernel(fft_kernel);  
  if(fft_kernel_2) 
    clReleaseKernel(fft_kernel_2);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(transpose_kernel_2) 
    clReleaseKernel(transpose_kernel_2); 

  return fpga_runtime;
}


/******************************************************************************
 * \brief   Initialize the program - select device, create context and program
 *****************************************************************************/
void init_program(int N[3], char *data_path){
  cl_int status = 0;

  // use the first device.
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &openCLContextCallBackFxn, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program.
  program = getProgramWithBinary(context, &device, 1, N, data_path);
  if(program == NULL) {
    printf("Failed to create program");
    exit(1);
  }
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

}

/******************************************************************************
 * \brief   Free resources allocated during program initialization
 *****************************************************************************/
void cleanup_program(){
  if(program) 
    clReleaseProgram(program);
  if(context)
    clReleaseContext(context);
}

/******************************************************************************
 * \brief   Initialize the OpenCL FPGA environment - platform and devices
 * \retval  true if error in initialization
 *****************************************************************************/
int init() {
  cl_int status = 0;

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return 1;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);

  return 0;
}

/******************************************************************************
 * \brief   Free resources allocated during initialization - devices
 *****************************************************************************/
void cleanup(){
  cleanup_program();
  free(devices);
}

/******************************************************************************
 * \brief   Create a command queue for each kernel
 *****************************************************************************/
void queue_setup(){
  cl_int status = 0;
  // Create one command queue for each kernel.
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue1");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue2");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue3");
  queue4 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue4");
  queue5 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue5");
  queue6 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue6");
}

/******************************************************************************
 * \brief   Release all command queues
 *****************************************************************************/
void queue_cleanup() {
  if(queue1) 
    clReleaseCommandQueue(queue1);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
  if(queue4) 
    clReleaseCommandQueue(queue4);
  if(queue5) 
    clReleaseCommandQueue(queue5);
  if(queue6) 
    clReleaseCommandQueue(queue6);
}
