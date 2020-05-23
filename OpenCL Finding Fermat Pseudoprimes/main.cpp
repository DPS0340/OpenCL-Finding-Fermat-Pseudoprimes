#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int* input = (int*)calloc(LIST_SIZE, sizeof(int));
    for (i = 0; i < LIST_SIZE; i++) {
        input[i] = i + 2;
    }

    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("FermatAlgorithm.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
        &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, input_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), input, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "fermat", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&output_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Process in groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    int* output = (int*)calloc(LIST_SIZE, sizeof(int));
    ret = clEnqueueReadBuffer(command_queue, output_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), output, 0, NULL, NULL);

    int sum = 0;
    // Display the result to the screen
    for (i = 0; i < LIST_SIZE; i++) {
        sum += output[i];
    }

    printf("range in [2, %d] Fermat pseudoprime Nunbers: %d\n", LIST_SIZE + 2, sum);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(input_mem_obj);
    ret = clReleaseMemObject(output_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(input);
    free(output);

    printf("Press Any Key to continue...\n");
    getchar();
    return 0;
}
