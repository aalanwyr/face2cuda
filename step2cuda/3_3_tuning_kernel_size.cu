#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include "common.h"


void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
  float * a=MatA;
  float * b=MatB;
  float * c=MatC;
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      c[i]=a[i]+b[i];
    }
    c+=nx;
    b+=nx;
    a+=nx;
  }
}

__global__ void sumMatrix(float *A, float *B, float *C, size_t nx, size_t ny ) {

    // 二维矩阵 一维操作
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = ix + iy * nx;

    if ( ix<nx && iy<ny) {
        C[index] = A[index] + B[index];
    }

}

int main(int argc, char** argv) {

    size_t nx = 1<<13;
    size_t ny = 1<<13;
    size_t nxy = nx * ny;  // Each matrix 4 GB
    size_t nBytes =  nxy * sizeof(float);
    int nMbtyes = nBytes / 1024 / 1024 ;
    printf("Current matrix size is %d MBs\n", nMbtyes);

    // Malloc
    float* A_host = (float*) malloc(nBytes);
    float* B_host = (float*) malloc(nBytes);
    float* C_host = (float*) malloc(nBytes);
    float* C_from_gpu = (float*) malloc(nBytes);
    //printf("0\n");
    initialData(A_host,nxy);
    initialData(B_host,nxy);
    //printf("1\n");
    // cudaMalloc

    float* A_dev = NULL;
    float* B_dev = NULL;
    float* C_dev = NULL;

    CHECK(cudaMalloc((void**)&A_dev,nBytes));
    CHECK(cudaMalloc((void**)&B_dev,nBytes));
    CHECK(cudaMalloc((void**)&C_dev,nBytes));
    //printf("2\n");

    // copy data
    CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice));

    //printf("3\n");

    // capture kernel size
    int dimx = argc>2 ? atoi(argv[1]) : 32;
    int dimy = argc>2 ? atoi(argv[2]) : 32;

    // CPU compute
    sumMatrix2D_CPU(A_host, B_host, C_host, nx, ny);

    //capture kernel execution time

    cudaEvent_t custart, custop;
    cudaEventCreate(&custart);
    cudaEventCreate(&custop);
    

    // 2d block and 2d grid
    dim3 block(dimx,dimy);
    dim3 grid((nx-1)/block.x+1, (ny-1)/block.y+1);

    cudaEventRecord(custart, 0);
    sumMatrix<<<grid,block>>>(A_dev,B_dev,C_dev,nx,ny);
    cudaEventRecord(custop, 0);
    cudaEventSynchronize(custop);

    float CudaElaps;
    cudaEventElapsedTime(&CudaElaps, custart, custop);

    CHECK(cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost));
    checkResult(C_host,C_from_gpu,nxy);
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)|%f ms\n", grid.x, grid.y, block.x, block.y, CudaElaps);


    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);

    cudaDeviceReset();

    return 0;

}