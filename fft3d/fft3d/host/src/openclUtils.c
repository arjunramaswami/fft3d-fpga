/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <string.h>
#include <sys/time.h>
#include <stdarg.h>

#include "CL/opencl.h"

#include <unistd.h> // access in fileExists()
#include "ctype.h"

static void tolowercase(char *p, char *q){
  int i;
  char a;
  for(i=0; i<strlen(p);i++){
    a = tolower(p[i]);
    q[i] = a;
  }
}

cl_platform_id findPlatform(char *platform_name){
  unsigned int i;
  cl_uint status;

  // Check if there are any platforms available
  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS){
    printf("Query for number of platforms failed\n");
    // cleanup():
    exit(0);
  }

  // Get ids of platforms available
  cl_platform_id *pids = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);
  status = clGetPlatformIDs(num_platforms, pids, NULL);
  if (status != CL_SUCCESS){
    printf("Query for platform ids failed\n");
    // cleanup():
    exit(0);
  }

  // Convert argument string to lowercase compare platform names
  size_t pl_len = strlen(platform_name);
  char name_search[pl_len];
  tolowercase(platform_name, name_search);

  // Search the platforms for the platform name passed as argument
  size_t sz;
  for(i=0; i<num_platforms; i++){
    // Get the size of the platform name referred to by the id
		status = clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, 0, NULL, &sz);
    if (status != CL_SUCCESS){
      printf("Query for platform info failed\n");
      // cleanup():
      exit(0);
    }

    char pl_name[sz];
    char plat_name[sz];

    // Store the name of string size
	status = clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, sz, pl_name, NULL);
    if (status != CL_SUCCESS){
      printf("Query for platform info failed\n");
      // cleanup():
      exit(0);
    }

    tolowercase(pl_name, plat_name);
    
    if( strstr(plat_name, name_search)){
      return pids[i];
    }

  }
  return NULL;
}

// Returns the list of all devices.
cl_device_id* getTestDevices(cl_platform_id pid, cl_device_type device_type, cl_uint *num_devices) {
  cl_int status;

  // Query for number of devices
  status = clGetDeviceIDs(pid, device_type, 0, NULL, num_devices);
  if(status != CL_SUCCESS){
    printf("Query for number of devices failed\n");
    exit(0);
  }

  //  Based on the number of devices get their device ids
  cl_device_id *dev_ids = new cl_device_id[*num_devices];
  status = clGetDeviceIDs(pid, device_type, *num_devices, dev_ids, NULL);
  if(status != CL_SUCCESS){
    printf("Query for device ids failed\n");
    exit(0);
  }

  return dev_ids;
}

static bool fileTestExists(char* filename){
  printf("filename %s\n", filename);
  if( access( filename, R_OK ) != -1 ) {
    return true;
  } else {
    return false;
  }
}

static bool getBinaryPath(char *path, int *N){

  path[0] = '\0';
  
  switch(N[0]){
    case 16 : 
      printf("Choosing 16 with path %s\n", path);
      // strcat(path, "syn16/");
      strcat(path, "emulation/");
      break;

    case 32 : 
      printf("Choosing 32 with path %s\n", path);
      // strcat(path, "syn32/");
      strcat(path, "emulation/");
      break;

    case 64 : 
      printf("Choosing 64 with path %s\n", path);
      // strcat(_path, "syn64/");
      strcat(path, "bin/emulation/");
      break;

    default:
      printf("Choosing with path %s\n", path);
      //strcat(path, "syn/");
      strcat(path, "emulation/");
      break;
  }
  // strcat(full_path, specific_path); // transfer specific path to filename 
  strcat(path, "fft3d");      // Append filename
  strcat(path, ".aocx");       // Append extension
  strcat(path, "\0");       // Append extension
  printf("Finding Path %s %d \n", path, strlen(path));

  /*
  char cwd[500];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
      printf("Current working dir: %s\n", cwd);
  } else {
      perror("getcwd() error");
  }
  */

  if (!fileTestExists(path)){
    printf("File not found \n");
    return false;
  }

  return true;
}

static char* loadBinary(char *binary_path, size_t *bin_size){

  FILE *fp;

  // Open file and check if it exists
  printf("Opening binary file\n");
  fp = fopen(binary_path, "rb");
  if(fp == 0){
    return NULL;
  }

  // Find the size of the file
  fseek(fp, 0L, SEEK_END);
  *bin_size = ftell(fp);

  //binary = (char *)malloc(*bin_size);
  char *binary = new char[*bin_size];
  rewind(fp);

  printf("Reading binary file of size %d \n", *bin_size);
  if(fread( (void *)binary, *bin_size, 1, fp) == 0) {
    delete[] binary;
    fclose(fp);
    return NULL;
  }

  printf("Binary Read %s \n");
  fclose(fp);
  return binary;
}

cl_program getProgramWithBinary(cl_context context, const cl_device_id *devices, unsigned num_device, int N[3]){
  char bin_path[500];
  char *binary;
  size_t bin_size;
  cl_int bin_status;
  cl_int status;

  char *binaries[num_device];

  printf("Getting Binary path\n");

  // Get the full path of the binary
  if( !getBinaryPath(bin_path, N) ){
    printf("No paths to the binary found\n");
    return NULL;
  }

  // Load binary to character array
  binary = loadBinary(bin_path, &bin_size);
  if(binary == NULL){
    printf("Could not load binary\n");
    return NULL;
  }

  binaries[0] = binary;

  // Create the program.
  printf("Creating Program\n");
  cl_program program = clCreateProgramWithBinary(context, 1, devices, &bin_size, (const unsigned char **) binaries, &bin_status, &status);
  if (status != CL_SUCCESS){
    printf("Query to create program with binary failed\n");
    // cleanup():
    return NULL;
  }
  return program;
}

// Minimum alignment requirement to use DMA
const unsigned OPENCL_ALIGNMENT = 64;
void* alignedMalloc(size_t size){

  void *memptr = NULL;
  int ret = posix_memalign(&memptr, OPENCL_ALIGNMENT, size);
  /*
  if (ret == 0){
    printf("Error on aligned memory allocation \n");
  }
  */
  /*
  if( ! posix_memalign(&memptr, OPENCL_ALIGNMENT, size)){
    printf("Error on aligned memory allocation \n");
    return NULL;
  }
  */
  return memptr;
}

void openCLContextCallBackFxn(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  printf("Context Callback - %s\n", errinfo);
}

void printError(cl_int error) {

  switch(error)
  {
    case CL_INVALID_PLATFORM:
      printf("CL_PLATFORM NOT FOUND OR INVALID ");
      break;
    case CL_INVALID_DEVICE:
      printf("CL_DEVICE NOT FOUND OR INVALID OR DOESN'T MATCH THE PLATFORM ");
      break;
    case CL_INVALID_CONTEXT:
      printf("CL_CONTEXT INVALID ");
      break;
    case CL_OUT_OF_HOST_MEMORY:
      printf("FAILURE TO ALLOCATE RESOURCES BY OPENCL");
      break;
    case CL_DEVICE_NOT_AVAILABLE:
      printf("CL_DEVICE NOT AVAILABLE ALTHOUGH FOUND");
      break;
    case CL_INVALID_QUEUE_PROPERTIES:
      printf("CL_QUEUE PROPERTIES INVALID");
      break;
    case CL_INVALID_PROGRAM:
      printf("CL_PROGRAM INVALID");
      break;
    case CL_INVALID_BINARY:
      printf("CL_BINARY INVALID");
      break;
    case CL_INVALID_KERNEL_NAME:
      printf("CL_KERNEL_NAME INVALID");
      break;
    case CL_INVALID_KERNEL_DEFINITION:
      printf("CL_KERNEL_DEFN INVALID");
      break;
    case CL_INVALID_VALUE:
      printf("CL_VALUE INVALID");
      break;
    case CL_INVALID_BUFFER_SIZE:
      printf("CL_BUFFER_SIZE INVALID");
      break;
    case CL_INVALID_HOST_PTR:
      printf("CL_HOST_PTR INVALID");
      break;
    case CL_INVALID_COMMAND_QUEUE:
      printf("CL_COMMAND_QUEUE INVALID");
      break;
    case CL_INVALID_MEM_OBJECT:
      printf("CL_MEM_OBJECT INVALID");
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      printf("CL_MEM_OBJECT_ALLOCATION INVALID");
      break;
    case CL_INVALID_ARG_INDEX:
      printf("CL_ARG_INDEX INVALID");
      break;
    case CL_INVALID_ARG_VALUE:
      printf("CL_ARG_VALUE INVALID");
      break;
    case CL_INVALID_ARG_SIZE:
      printf("CL_ARG_SIZE INVALID");
      break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
      printf("CL_PROGRAM_EXEC INVALID");
      break;
    case CL_INVALID_KERNEL:
      printf("CL_KERNEL INVALID");
      break;
    case CL_INVALID_KERNEL_ARGS:
      printf("CL_KERNEL_ARG INVALID");
      break;
    case CL_INVALID_WORK_GROUP_SIZE:
      printf("CL_WORK_GROUP_SIZE INVALID");
      break;

    default:
      printf("UNKNOWN ERROR %d\n", error);
  }

}

void _checkError(const char *file, int line, const char *func, cl_int err, const char *msg, ...){

  if(err != CL_SUCCESS){
    printf("ERROR: ");
    printError(err);
    printf("\nError Location: %s:%d:%s\n", file, line, func);


    // custom message 
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    exit(err);
  }
}