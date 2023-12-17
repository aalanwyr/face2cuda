#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include "common.h"

void sumArraysCPU(float *A, float *B, float *C, int N) {
    
    for (int i=0; i<N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysGPU(float *A, float *B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main(int argc, char** argv) {

    int len = 1<<24;
    printf("Vector size:%d\n",len);
    int nBytes = len * sizeof(float);

    float *res_h = (float*) malloc(nBytes);
    memset(res_h, 0, nBytes);

    // uniform memory don't need to memcpy or get the device ptr (mapped ptr)
    float *a_d, *b_d, *c_d;
    CHECK(cudaMallocManaged((float**)&a_d, nBytes));
    CHECK(cudaMallocManaged((float**)&b_d, nBytes));
    CHECK(cudaMallocManaged((float**)&c_d, nBytes));

    initialData(a_d, len); //  nBytes = len * sizeof()
    initialData(b_d, len);
    
    // uniform no need memcpy
    //CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));
    //cudaError_t cudaHostGetDevicePointer(void ** pDevice,void * pHost,unsigned flags);

    dim3 block(512);
    dim3 grid((len-1)/block.x+1);

    cudaEvent_t custart, custop;
    cudaEventCreate(&custart);
    cudaEventCreate(&custop);

    cudaEventRecord(custart, 0);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, c_d, len);
    cudaEventRecord(custop, 0);
    cudaEventSynchronize(custop);

    float CudaElaps;
    cudaEventElapsedTime(&CudaElaps, custart, custop);

    sumArraysCPU(a_d, b_d, res_h, len);

    checkResult(res_h, c_d, len);
    printf("GPU Execution %f ms\n", CudaElaps);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(res_h);

    return 0;
}

