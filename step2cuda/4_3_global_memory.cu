#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include "common.h"

void sumArraysCPU(float *A, float *B, float *C, int offset, int N) {
    // cpu 侧的计算也需要带偏移
    for (int i=0, k=offset; k<N; i++,k++) {
        C[i] = A[k] + B[k]; // offset 之前的数值应该不参与计算了
    }
}

__global__ void sumArraysGPU(float *A, float *B, float *C, int offset, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  
    int index = tid + offset;  //添加偏移量 从 gld memory 读取数据带地址偏移 由于带了
    if(index<N) {
        C[tid] = A[index] + B[index];
    }
}
 
int main(int argc, char** argv) {

    int len = 1<<24;
    printf("Vector size:%d\n",len);
    int nBytes = len * sizeof(float);

    int offset = 0;
    if( argc>=2 ) {
        offset = atoi(argv[1]); // get the offset value
    }

    //malloc host buffer
    float *a_h = (float*) malloc(nBytes);
    float *b_h = (float*) malloc(nBytes);
    float *res_h = (float*) malloc(nBytes);
    float *res_gpu = (float*) malloc(nBytes);
    memset(res_h, 0, nBytes);
    memset(res_gpu, 0, nBytes);

    initialData(a_h, len);
    initialData(b_h, len);

    //malloc pinned device buffer
    float *a_d, *b_d, *res_d;
    CHECK(cudaMallocHost((float**) &a_d, nBytes))
    CHECK(cudaMallocHost((float**) &b_d, nBytes))
    CHECK(cudaMallocHost((float**) &res_d, nBytes))
    CHECK(cudaMemset(res_d,0,nBytes));

    CHECK(cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(len/block.x);

    // creat cudaEvent to collect the timestamp
    cudaEvent_t custart, custop;
    cudaEventCreate(&custart);
    cudaEventCreate(&custop);

    cudaEventRecord(custart, 0);
    sumArraysGPU<<<grid,block>>> (a_d, b_d, res_d, offset, len);
    cudaEventRecord(custop, 0);
    cudaEventSynchronize(custop);

    float CudaElaps;
    cudaEventElapsedTime(&CudaElaps, custart, custop);

    sumArraysCPU(a_d, b_d, res_h, offset, len);

    CHECK(cudaMemcpy(res_gpu, res_d, nBytes, cudaMemcpyDeviceToHost));

    //checkResult(res_h, res_gpu, len);
    printf("GPU Execution %f ms\n", CudaElaps);


    free(a_h);
    free(b_h);
    free(res_h);
    free(res_gpu);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);


    return 0;
}
