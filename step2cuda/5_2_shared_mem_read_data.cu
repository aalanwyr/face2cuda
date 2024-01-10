#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include "common.h"

/* 本code 主要是测试 共享内存中 二维数组的读写 
 测试行主序 与 列主序 导致的 bank 冲突
 测试 通过 pad 操作来避免 bank 冲突 */

//定义正方形数组的大小
#define BDIMX 32
#define BDIMY 32

//定义矩形数组的大小
#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1  

__global__ void warmup(int* out) {

    __shared__ int tile[BDIMY][BDIMX]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    tile[threadIdx.y][threadIdx.x] = idx; // 往 SMEM 中写数据
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[threadIdx.y][threadIdx.x]; // 行主序

}

__global__ void setRowReadRow(int* out) {

    __shared__ int tile[BDIMY][BDIMX]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    tile[threadIdx.y][threadIdx.x] = idx; // 往 SMEM 中写数据
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[threadIdx.y][threadIdx.x]; // 行主序

}

__global__ void setRowReadCol(int* out) {

    __shared__ int tile[BDIMY][BDIMX]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    tile[threadIdx.y][threadIdx.x] = idx; // 往 SMEM 中写数据
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[threadIdx.x][threadIdx.y]; // 列主序 由于 cuda 是按照 threadId.x 先变化的 所以先读一列数据

}

__global__ void setColReadCol(int* out) {

    __shared__ int tile[BDIMY][BDIMX]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    tile[threadIdx.x][threadIdx.y] = idx; // 往 SMEM 中写数据 列主序 写回数据
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[threadIdx.x][threadIdx.y]; // 列主序 由于 cuda 是按照 threadId.x 先变化的 所以先读一列数据

}

__global__ void setColReadRow(int* out) {

    __shared__ int tile[BDIMY][BDIMX]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    tile[threadIdx.x][threadIdx.y] = idx; // 往 SMEM 中写数据 列主序写回数据
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[threadIdx.y][threadIdx.x]; // 行主序 由于 cuda 是按照 threadId.x 先变化的 所以先读一列数据

}

__global__ void setColReadColPad(int* out) {

    __shared__ int tile[BDIMY][BDIMX+IPAD]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    tile[threadIdx.x][threadIdx.y] = idx; // 往 SMEM 中写数据 列主序写回数据 PAD 之后理论上这个  冲突应该就没有了
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[threadIdx.x][threadIdx.y]; // 行主序 由于 cuda 是按照 threadId.x 先变化的 所以先读一列数据

}
/*------------------------------矩形二维数组----------------------------------------------*/

__global__ void setRowReadRowRect(int* out) {

    __shared__ int tile[BDIMY_RECT][BDIMX_RECT]; // 行主序 存放二维数组 这个 tile share mem 的长度和宽度确定了 感觉不能随意 更换行主序 列主序了
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    unsigned int icol=idx%blockDim.x; // 矩形数组关键在于 索引的映射的修改
    unsigned int irow=idx/blockDim.x;
    tile[threadIdx.y][threadIdx.x] = idx; // 往 SMEM 中写数据 只能行主序写回数据 因为是行主序 16*32 的矩阵 那么 threadIdx.x (0~31) block 32*16
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[irow][icol]; // 行主序  同一个线程写 和 读的位置不一样了 但是依旧保证的数据全部拿回去了

}

__global__ void setRowReadColRect(int* out) {

    __shared__ int tile[BDIMY_RECT][BDIMX_RECT]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx; // 往 SMEM 中写数据
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    //printf("dx%d dy%d id%d icol%d irow%d\n",threadIdx.x,threadIdx.y,idx,icol,irow);
    out[idx]=tile[icol][irow]; // 列主序  这样映射的话 需要  idx%blockDim.y

}


__global__ void setRowReadColPadRect(int* out) {

    __shared__ int tile[BDIMY_RECT][BDIMX_RECT+IPAD]; // 行主序 存放二维数组
    unsigned int idx = threadIdx.y*blockDim.x + threadIdx.x; // 二维矩阵 index 映射到 idx 
    unsigned int icol=idx%blockDim.y;
    unsigned int irow=idx/blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx; // 由于 tile 的大小为 16*32 
    __syncthreads();

    // 读SMEM 数据 写回 全局内存
    out[idx]=tile[icol][irow]; // 列主序 由于 cuda 是按照 threadId.x 先变化的 所以先读一列数据

}

void run(int kernel_index, dim3 block, dim3 grid, int* out) {

    // Use Cuda event to capture the duration
    cudaEvent_t custart, custop;
    cudaEventCreate(&custart);
    cudaEventCreate(&custop);

    cudaEventRecord(custart, 0);
    // according kernel index to choose different kernel
    printf("kernel_index %d launch\n", kernel_index);
    int loop = 100;
    for (int i=0; i<100; i++) {

    switch (kernel_index)
    {
        case 0:
            setRowReadRow<<<grid,block>>>(out);
            break;
        case 1:
            setRowReadCol<<<grid,block>>>(out);
            break;
        case 2:
            setColReadRow<<<grid,block>>>(out);
            break;
        case 3:
            setColReadCol<<<grid,block>>>(out);
            break;
        case 4:
            setColReadColPad<<<grid,block>>>(out);
            break;
        case 5:
            setRowReadRowRect<<<grid,block>>>(out);
            break;
        case 6:
            setRowReadColRect<<<grid,block>>>(out);
            break;
        case 7:
            setRowReadColPadRect<<<grid,block>>>(out);
            break;
        default:
            setRowReadRow<<<grid,block>>>(out);
            break;
    }
    }
    cudaEventRecord(custop, 0);
    cudaEventSynchronize(custop);

    float CudaElaps;
    cudaEventElapsedTime(&CudaElaps, custart, custop);
    printf("Kernel execution GPU time %.6f ms\n", CudaElaps/loop);

}
   

int main(int argc, char** argv) {

    initDevice(0);
    int nElem=BDIMX*BDIMY;
    printf("Vector size:%d\n",nElem);
    int nByte=sizeof(int)*nElem;
    int * out;
    CHECK(cudaMalloc((int**)&out,nByte));

    // 检查当前 GPU share Mem 的 bank 颗粒度 
    cudaSharedMemConfig MemConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&MemConfig));
    printf("--------------------------------------------\n");
    switch (MemConfig) {

      case cudaSharedMemBankSizeFourByte:
        printf("the device is cudaSharedMemBankSizeFourByte: 4-Bytes\n");
      break;
      case cudaSharedMemBankSizeEightByte:
        printf("the device is cudaSharedMemBankSizeEightByte: 8-Bytes\n");
      break;

     }
    printf("--------------------------------------------\n");
    // 设定 block size 以及 grid size
    dim3 block(BDIMY,BDIMX);
    dim3 grid(1,1);
    dim3 block_rect(BDIMX_RECT,BDIMY_RECT);
    dim3 grid_rect(1,1);
    
    // first  warm up
    warmup<<<grid,block>>>(out);
    printf("warmup!\n");

    //
    printf("===launch kernel setRowReadRow=== \n");
    run(0, block, grid, out);
    printf("===launch kernel setRowReadCol=== \n");
    run(1, block, grid, out);
    printf("===launch kernel setColReadRow=== \n");
    run(2, block, grid, out);
    printf("===launch kernel setColReadCol=== \n");
    run(3, block, grid, out);
    printf("===launch kernel setColReadColPad=== \n");
    run(4, block, grid, out);
    printf("===launch kernel setRowReadRowRect=== \n");
    run(5, block_rect, grid_rect, out);
    printf("===launch kernel setRowReadColRect=== \n");
    run(6, block_rect, grid_rect, out);
    printf("===launch kernel setRowReadColPadRect=== \n");
    run(7, block_rect, grid_rect, out);


    cudaFree(out);
    return 0;
}