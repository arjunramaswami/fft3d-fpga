#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <cstring>
#include <sys/time.h>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "fft_decl.h"
#include "test_fft.h"

using namespace aocl_utils;

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL, queue2 = NULL, queue3 = NULL, queue4 = NULL, queue5 = NULL;
static cl_kernel fft_kernel = NULL, fft_kernel_2 = NULL; 
static cl_kernel fetch_kernel = NULL, transpose_kernel = NULL, transpose_kernel_2 = NULL;
static cl_program program = NULL;
static cl_int status = 0;
static int flag = 0;

// Function prototypes
static bool init(int N);
void cleanup();
static std::string select_binary(int N);

// Device memory buffers
cl_mem d_inData, d_outData;

// Entry point.
int fft3d(int N, cmplex *h_inData, cmplex *h_outData, bool inverse){
  if(flag == 0){
    init(N);
  }
  flag++;
  /*
  if(!init(N)) {
    return false;
  }
  */

  printf("Launching %s FFT transform \n", inverse ? "inverse ":"");

  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplex) * N * N * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cmplex) * N * N * N, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");

  // Copy data from host to device
  status = clEnqueueWriteBuffer(queue, d_inData, CL_TRUE, 0, sizeof(cmplex) * N * N * N, h_inData, 0, NULL, NULL);
  checkError(status, "Failed to copy data to device");

  // Can't pass bool to device, so convert it to int
  int inverse_int = inverse;

  printf("Kernel initialization is complete.\n");
  status = clSetKernelArg(fetch_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set kernel arg 0");
  status = clSetKernelArg(fft_kernel, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 1");
  status = clSetKernelArg(transpose_kernel, 0, sizeof(cl_mem), (void *)&d_outData);
  checkError(status, "Failed to set kernel arg 2");
  status = clSetKernelArg(fft_kernel_2, 0, sizeof(cl_int), (void*)&inverse_int);
  checkError(status, "Failed to set kernel arg 3");

  // Get the iterationstamp to evaluate performance
  struct timeval start, stop;
  double time = 0.0;
  gettimeofday(&start, NULL);

  status = clEnqueueTask(queue, fetch_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  // Launch the fft kernel - we launch a single work item hence enqueue a task
  status = clEnqueueTask(queue2, fft_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  status = clEnqueueTask(queue3, transpose_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  status = clEnqueueTask(queue4, fft_kernel_2, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  status = clEnqueueTask(queue5, transpose_kernel_2, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  // Wait for all command queues to complete pending events
  status = clFinish(queue);
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
  gettimeofday(&stop, NULL);

  // Copy results from device to host
  status = clEnqueueReadBuffer(queue, d_outData, CL_TRUE, 0, sizeof(cmplex) * N * N * N, h_outData, 0, NULL, NULL);
  checkError(status, "Failed to copy data from device");

  time = ((stop.tv_sec - start.tv_sec) * 1E6) + (stop.tv_usec - start.tv_usec);

  if(flag == 50){
    cleanup();
    flag = 0;
  }

  return time;
}

static bool init(int N) {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return false;
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create one command queue for each kernel.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue4 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue5 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.
  std::string binary_file = select_binary(N);
  printf("Using AOCX: %s\n\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  fft_kernel = clCreateKernel(program, "fft3da", &status);
  checkError(status, "Failed to create kernel");
  fft_kernel_2 = clCreateKernel(program, "fft3db", &status);
  checkError(status, "Failed to create kernel");
  fetch_kernel = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create kernel");
  transpose_kernel = clCreateKernel(program, "transpose", &status);
  checkError(status, "Failed to create kernel");
  transpose_kernel_2 = clCreateKernel(program, "transpose3d", &status);
  checkError(status, "Failed to create kernel");

  return true;
}

static std::string select_binary(int N){
  std::string binary_file;

  switch(N){
    case 16 : 
      //binary_file = getBoardBinaryFile("stratix10/syn16/fft3d", device);
      binary_file = getBoardBinaryFile("emulation/fft3d", device);
      // binary_file = getBoardBinaryFile("profile/fft3d", device);
      //binary_file = getBoardBinaryFile("emu/emu16/fft3d", device);
      break;

    case 32 : 
      //binary_file = getBoardBinaryFile("stratix10/syn32/fft3d", device);
      binary_file = getBoardBinaryFile("emulation/fft3d", device);
      //binary_file = getBoardBinaryFile("stratix10/syn32/fft3d", device);
      //binary_file = getBoardBinaryFile("emu/emu32/fft3d", device);
      break;

    case 64 : 
      //binary_file = getBoardBinaryFile("stratix10/syn64_18.0.1/fft3d", device);
      //binary_file = getBoardBinaryFile("profile/fft3d", device);
      binary_file = getBoardBinaryFile("emulation/fft3d", device);
      //binary_file = getBoardBinaryFile("emu/emu64/fft3d", device);
      break;

    default:
      printf("No binary found\n");
  }

  return binary_file;
}

// Free the resources allocated during initialization
void cleanup() {
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
  
  if(program) 
    clReleaseProgram(program);
  if(context)
    clReleaseContext(context);

  if(queue) 
    clReleaseCommandQueue(queue);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
  if(queue4) 
    clReleaseCommandQueue(queue4);
  if(queue5) 
    clReleaseCommandQueue(queue);

  if (d_inData)
	clReleaseMemObject(d_inData);
  if (d_outData) 
	clReleaseMemObject(d_outData);
}
