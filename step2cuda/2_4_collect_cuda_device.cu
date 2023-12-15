#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>


int main(int argc, char** argv) {
    //collect device info and print
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf ("cudaGetDeviceCount returned %d\n -> %s\n",
        (int)error, cudaGetErrorString(error));
        printf ("Failed to get CUDA device\n");
        return (EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available CUDA devices\n");
    } else {
        printf("There are %d available CUDA devices\n", deviceCount);
    }

    /*
    cudaGetDevice	(int * device） - Returns the device on which the active host thread executes the device code. 返回当前被选择的设备
    cudaSetDevice 用于选择当前 操作的 device 设备
    size_t size = 1024 * sizeof(float);
    cudaSetDevice(0);            // Set device 0 as current
    float* p0;
    cudaMalloc(&p0, size);       // Allocate memory on device 0
    MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
    cudaSetDevice(1);            // Set device 1 as current
    float* p1;
    cudaMalloc(&p1, size);       // Allocate memory on device 1
    MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
    */
    // support to capture multi-cuda-device
    int driverVersion=0, runtimeVersion=0, curdev=0;
    cudaDeviceProp deviceProp;

    for ( int dev=0; dev<deviceCount; dev++) {
        cudaSetDevice(dev); //select dev
        cudaGetDevice(&curdev);
        cudaGetDeviceProperties(&deviceProp,dev);
        printf("Current active Device %d:\"%s\"\n",curdev,deviceProp.name);

        // get driver version
        cudaDriverGetVersion(&driverVersion); 
        // Returns in *driverVersion the latest version of CUDA supported by the driver. The version is returned as (1000 major + 10 minor). For example, 
        // CUDA 9.2 would be represented by 9020. If no driver is installed, then 0 is returned as the driver version.
        cudaRuntimeGetVersion(&runtimeVersion);
        // The version is returned as (1000 major + 10 minor). For example, CUDA 9.2 would be represented by 9020.
        printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n",
            driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:   %d.%d\n", deviceProp.major,deviceProp.minor);
        printf("  Total amount of global memory:                %.2f GBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/pow(1024.0,3), deviceProp.totalGlobalMem);
        printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
        printf("  Memory Bus width:                             %d-bits\n",deviceProp.memoryBusWidth);
        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                            	%d bytes\n",deviceProp.l2CacheSize);
        }
        printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
            deviceProp.maxTexture1D,deviceProp.maxTexture2D[0],deviceProp.maxTexture2D[1]
            ,deviceProp.maxTexture3D[0],deviceProp.maxTexture3D[1],deviceProp.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
            deviceProp.maxTexture1DLayered[0],deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0],deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
        printf("  Total amount of constant memory               %lu bytes\n",
            deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:      %lu bytes\n",
            deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block:%d\n",
            deviceProp.regsPerBlock);
        printf("  Threads Wrap size:                            %d\n",deviceProp.warpSize);
        printf("  Maximun number of thread per multiprocesser:  %d\n",deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximun number of thread per block:           %d\n",deviceProp.maxThreadsPerBlock);
        printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
            deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
        printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
            deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
        printf("  Maximu memory pitch                           %lu bytes\n",deviceProp.memPitch);

    }

    exit(EXIT_SUCCESS);


}