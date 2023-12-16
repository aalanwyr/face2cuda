#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void recurHelloworld(int size, int depth) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Current dept %d blockIdx %d threadIdx %d tid %d\n", depth, blockIdx.x, threadIdx.x, tid );

    if (size == 1) {
        return;  // 递归退出标志
    }

    int nthread = (size>>1);
    
    if (threadIdx.x==0 && nthread) {
        recurHelloworld<<<1,nthread>>> (nthread,++depth); // 0 号 子线程启动 一般的 子线程
        printf("-----------> nested execution depth: %d\n",depth);
    }



}

int main(int argc, char ** argv) {
    int size = 32;
    // grid = 1 block size =32 only launch a warp to print hello world
    recurHelloworld<<<1,32>>>(size,0);
    cudaDeviceSynchronize();
    return 0;
}