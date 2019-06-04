/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <cstring>
#include <sys/time.h>
#include <stdarg.h>
#include <unistd.h> // access in fileExists()

// common dependencies
#include "CL/opencl.h"

//local dependencies
#include "openclUtils.h"
// #include "AOCLUtils/aocl_utils.h"
#include "fft3d_decl.h"
//#include "test_fft.h"

// host variables
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_program program = NULL;

static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
static cl_command_queue queue4 = NULL, queue5 = NULL, queue6 = NULL;

static cl_kernel fft_kernel = NULL, fft_kernel_2 = NULL; 
static cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, transpose_kernel_2 = NULL;

// Device memory buffers
cl_mem d_inData, d_outData;

// Global Variables
static cl_int status = 0;
static int flag = 0;
int fft_size[3] = {0,0,0};
bool fft_size_changed = true;

cmplx *h_outData, *h_inData;
cmplx *h_verify_tmp, *h_verify;

// Function prototypes
bool init();
void cleanup();

// --- CODE -------------------------------------------------------------------

int fpga_initialize_(){
   status = init();
   return status;
}

void fpga_final_(){
   cleanup();
}

/******************************************************************************
 * \brief   Initialize the OpenCL FPGA environment
 * \retval  true if error in initialization
 *****************************************************************************/
bool init() {
  cl_int status;

  // Get the OpenCL platform.
  printf("Testing Platform\n");
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return false;
  }

  // Query the available OpenCL devices.
  printf("Testing Devices\n");
  cl_uint num_devices;
  cl_device_id *devices = getTestDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);

  // use only the first device.
  device = devices[0];

  // Create the context.
  printf("Testing Context\n");
  context = clCreateContext(NULL, 1, &device, &openCLContextCallBackFxn, NULL, &status);
  checkError(status, "Failed to create context");

  return true;
}

static void queue_setup(){
  cl_int status;
  // Create one command queue for each kernel.
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  queue4 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  queue5 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  queue6 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
}

static void init_program(){

  program = getProgramWithBinary(context, &device, 1, fft_size);
  if(program == NULL) {
    printf("Failed to create program");
    exit(0);
  }

  //std::string binary_file = select_binary(N);
  //printf("Using AOCX: %s\n\n", binary_file.c_str());
  //program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  printf("Building Binary\n");
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  printf("Creating Kernels\n");
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

  printf("Creating Buffers\n");
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2], NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2], NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

}

/******************************************************************************
 * \brief   Free resources allocated during initialization
 *****************************************************************************/
void cleanup(){

  if(context)
    clReleaseContext(context);

  if(program) 
    clReleaseProgram(program);

  if(fft_kernel) 
    clReleaseKernel(fft_kernel);  
  if(fft_kernel_2) 
    clReleaseKernel(fft_kernel_2);  
  if(fetch_kernel) 
    clReleaseKernel(fetch_kernel);  
  if(transpose_kernel) 
    clReleaseKernel(transpose_kernel);  
  if(transpose_kernel_2) 
    clReleaseKernel(transpose_kernel_2);  

  if (d_inData)
	clReleaseMemObject(d_inData);
  if (d_outData) 
	clReleaseMemObject(d_outData);
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

/******************************************************************************
 * \brief  check whether FFT3d can be computed on the FPGA or not. This depends 
 *         on the availability of bitstreams whose sizes are for now listed here 
 * \param  N - integer pointer to the size of the FFT3d
 * \retval true if no board binary found in the location
 *****************************************************************************/
bool fpga_check_bitstream_(int N[3]){
    
    if( (N[0] == 16 && N[1] == 16 && N[2] == 16) ||
        (N[0] == 32 && N[1] == 32 && N[2] == 32) ||
        (N[0] == 64 && N[1] == 64 && N[2] == 64)  ){

        if( fft_size[0] == N[0] && fft_size[1] == N[1] && fft_size[2] == N[2] ){
            fft_size_changed = false;
        }
        else{
            fft_size[0] = N[0];
            fft_size[1] = N[1];
            fft_size[2] = N[2];
            fft_size_changed = true;
        }
        return 1;
    }
    else{
        return 0;
    }
}
/******************************************************************************
 * \brief   Execute a single precision complex FFT3d
 * \param   inverse : boolean
 * \param   N       : integer pointer to size of FFT3d  
 * \param   din     : complex input/output single precision data pointer 
 *****************************************************************************/
static void fftfpga_run_3d(bool inverse, cmplx *c_in) {

  int inverse_int = inverse;

  printf("Allocating aligned memory of size (%d, %d, %d) \n", fft_size[0], fft_size[1], fft_size[2]);
  h_inData = (cmplx *)alignedMalloc(sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2]);
  h_outData = (cmplx *)alignedMalloc(sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2]);

  memcpy(h_inData, c_in, sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2]);

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queue6, d_inData, CL_TRUE, 0, sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2], h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  status = clFinish(queue6);
  checkError(status, "failed to finish");
  // Can't pass bool to device, so convert it to int

  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 2");
  status = clSetKernelArg(fft_kernel_2, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 3");

  // Get the iterationstamp to evaluate performance
  /*
  struct timeval start, stop;
  double time = 0.0;
  gettimeofday(&start, NULL);
  */

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

  // Record execution time
  // gettimeofday(&stop, NULL);

  // Copy results from device to host
  status = clEnqueueReadBuffer(queue3, d_outData, CL_TRUE, 0, sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2], h_outData, 0, NULL, NULL);
  checkError(status, "Failed to read data from device");

  // time = ((stop.tv_sec - start.tv_sec) * 1E6) + (stop.tv_usec - start.tv_usec);

  memcpy(c_in, h_outData, sizeof(cmplx) * fft_size[0] * fft_size[1] * fft_size[2] );

  if (h_outData)
	  free(h_outData);

  if (h_inData)
	  free(h_inData);

}
/******************************************************************************
 * \brief   compute an in-place single precision complex 3D-FFT on the FPGA
 * \param   data_path_len - length of the path to the data directory
 * \param   data_path - path to the data directory 
 * \param   direction : direction - 1/forward, otherwise/backward FFT3d
 * \param   N   : integer pointer to size of FFT3d  
 * \param   din : complex input/output single precision data pointer 
 *****************************************************************************/
void fpga_fft3d_sp_(int data_path_len, char *data_path, int direction, int N[3], cmplx *din) {

  // data_path[data_path_len] = '\0';

  printf("Setting up Queues ... \n");
  queue_setup();

  // If fft size changes, need to rebuild program using another binary
  if(fft_size_changed == true){
    /*
    status = select_binary(data_path, N);
    checkError(status, "Failed to select binary as no relevant FFT3d binaries found in the directory!");
    */

    printf("Initializing Program and Binaries ... \n");
    init_program();
  }

  // setup device specific constructs 
  if(direction == 1){
    fftfpga_run_3d(0, din);
  }
  else{
    fftfpga_run_3d(1, din);
  }

  queue_cleanup();
}

