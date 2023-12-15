#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256
const int N = 32*1024*1024;
const int M = 4;

__device__ void warpReduce(volatile float* cache, unsigned int tid) {
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

// baseline should be the device ptr
__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// warp divergence
__global__ void reduce1(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduce and avoid the wrap divergence 256 threads per block and 8 wraps
    // 尽量避免 但是没法完全避免 最后 一次 if 判断 thread0 thread1 的if 判断产生了不同分支
    // s step 最大到128 就完成所有的 reduce 操作了
    for (unsigned int s=1; s<=blockDim.x/2; s *= 2) {
        int index = tid*2*s;
        if(index<blockDim.x) {
            sdata[index] += sdata[index + s];
        }
         __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];

}

// avoid bank conflict 是 warp 层面 还是 block 里的所有 thread 都必须得避免 bank conflict
// warp 层面才考虑 bank conflict 同一个 warp  中的  thread 是 SIMD ，不要产生 bank conflict
__global__ void reduce2(float *d_in, float *d_out) {

    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // 逆 for 循环 可以完全解决一个 warp 中的 bank conflict 同时保证前几次的 warp 没有分支

    for(unsigned int s = blockDim.x/2; s>0; s>>=1) { // 移位效率高
        if(tid<s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = sdata[0];

}

// 如果 tid 都是在一个 wrap 里面的话 就不需要 sync 了，对于同一个 warp 的thread 如果没有 divergent 那么就有天然的 sync
// 把 s=32 的 if 判断抽取出来 然后直接循环展开操作 或者单写一个 for 循环不带 sync

__global__ void reduce3(float *d_in, float *d_out) {

    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // 逆 for 循环 可以完全解决一个 warp 中的 bank conflict 同时保证前几次的 warp 没有分支

    for(unsigned int s = blockDim.x/2; s>32; s>>=1) { // 多个 warp 操作需要 sync
        if(tid<s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    //  the ans is wrong,res[0]: 256.00 host_out[0] 104.00
    volatile float *cache = sdata;
    for(unsigned int s = 32; s>0; s>>=1) { // 同一个 warp 操作不需要 sync
        if(tid<s) {
            cache[tid] += cache[tid+s]; // 不需要 sync 但是 必须得 volatile 否则会被编译器优化出问题
        }
    }

   /*
   volatile float *cache = sdata; // 必须得 volatile 否则编译器会优化掉这个操作
   if (tid<32) {
      cache[tid]+=cache[tid+32];
      cache[tid]+=cache[tid+16];
      cache[tid]+=cache[tid+8];
      cache[tid]+=cache[tid+4];
      cache[tid]+=cache[tid+2];
      cache[tid]+=cache[tid+1];

   }
   */

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];

}

// thread < 31 的 warp 循环展开 不用 sync
__global__ void reduce4(float *d_in, float *d_out) {

    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // 逆 for 循环 可以完全解决一个 warp 中的 bank conflict 同时保证前几次的 warp 没有分支

    for(unsigned int s = blockDim.x/2; s>32; s>>=1) { // 多个 warp 操作需要 sync
        if(tid<s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    //  the ans is wrong,res[0]: 256.00 host_out[0] 104.00
    if (tid<32) warpReduce(sdata,tid);

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];

}

// 使用调整 Block 个数 以及每个block reduce 的个数
__global__ void reduce5(float *d_in, float *d_out) {

    __shared__ float sdata[THREAD_PER_BLOCK];
    volatile float *cache = sdata;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*M + threadIdx.x;
    
    //sdata[tid] = d_in[i] + d_in[i+blockDim.x];
    cache[tid] = 0; // 必须得初始化
    for (unsigned j=0; j<M; j++) {
        cache[tid] += d_in[i+blockDim.x*j]; // 这种带循环的方式 始终有性能损失 256 个线程 操作1024 个数据 是性能最好的情况 M=4
    }
    
    __syncthreads();

    // 逆 for 循环 可以完全解决一个 warp 中的 bank conflict 同时保证前几次的 warp 没有分支

    for(unsigned int s = blockDim.x/2; s>32; s>>=1) { // 多个 warp 操作需要 sync
        if(tid<s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    //  the ans is wrong,res[0]: 256.00 host_out[0] 104.00
    if (tid<32) warpReduce(sdata,tid);

    if (tid == 0) d_out[blockIdx.x] = sdata[tid];

}

// 完全循环展开


// shuffle 指令



double CpuTimeSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

bool check(float *out, float *res, int n) {
    for(int i=0; i<n; i++) {
        if(res[i]!=out[i]) return false;
    }
    return true;
}

void run(int kernel_index, int block_num, int loop, dim3 Grid,dim3 Block,float *device_a ,float *device_out,float *host_out,float *res) {

    // Use CPU time to capture the duration
    double cpuStart,cpuElaps, bandwidth, Cubandwidth;

    // Use Cuda event to capture the duration
    cudaEvent_t custart, custop;
    cudaEventCreate(&custart);
    cudaEventCreate(&custop);

    cpuStart = CpuTimeSecond();
    cudaEventRecord(custart, 0);

    // according kernel index to choose different kernel
    switch (kernel_index)
    {
    case 0:
        printf("kernel_index %d launch reduce0 baseline\n", kernel_index);
        for ( int i=0; i<loop; i++) {
            reduce0<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    
    case 1:
        printf("kernel_index %d launch reduce1 avoid warp divergent\n", kernel_index);
        for ( int i=0; i<loop; i++) {
            reduce1<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    case 2:
        printf("kernel_index %d launch reduce2 avoid warp bank conflict\n", kernel_index);
        for ( int i=0; i<loop; i++) {
            reduce2<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    case 3:
        printf("kernel_index %d launch reduce3 avoid sync in warp0\n", kernel_index);
        for ( int i=0; i<loop; i++) {
            reduce3<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    case 4:
        printf("kernel_index %d launch reduce4 avoid sync in warp0 loop unrolling  \n", kernel_index);
        for ( int i=0; i<loop; i++) {
            reduce4<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    case 5:
        printf("kernel_index %d launch reduce5 change the block num  \n", kernel_index);
        printf("M %d, %d data per block\n", M, M*THREAD_PER_BLOCK);
        for ( int i=0; i<loop; i++) {
            reduce5<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    
    default:
        printf("launch default reduce0 baseline\n");
        for ( int i=0; i<loop; i++) {
            reduce0<<<Grid,Block>>>(device_a,device_out);
        }
        break;
    }


    cudaEventRecord(custop, 0);
    cudaEventSynchronize(custop);

    float CudaElaps;
    cudaEventElapsedTime(&CudaElaps, custart, custop);
    CudaElaps = CudaElaps / loop;

    cpuElaps = (CpuTimeSecond() - cpuStart) / loop;  // cpu time

    bandwidth = N*sizeof(float) / cpuElaps / 1024 / 1024 / 1024;
    Cubandwidth = N*sizeof(float) / CudaElaps / 1024 / 1024 / 1024 * 1e3;

    cudaMemcpy(host_out,device_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(host_out,res,block_num)) {
        printf("the ans is right host_out[0]:%.2f \n",host_out[0]);
        printf("Kernel execution CPU time %.6f ms BW: %.2f GB/s\n", cpuElaps*1e3, bandwidth);
        printf("Kernel execution GPU time %.6f ms BW: %.2f GB/s\n", CudaElaps, Cubandwidth);
    }
    else{
        printf("the ans is wrong,res[0]: %.2f host_out[0] %.2f \n",res[0], host_out[0]);
    }

}

int main() {

    // input data
    float *host_a = (float*) malloc(N*sizeof(float));
    float *device_a = NULL;
    cudaMalloc((void**)&device_a,N*sizeof(float));

    // output data
    int block_num = N / THREAD_PER_BLOCK;
    float *host_out =  (float*) malloc(block_num*sizeof(float));
    float *device_out = NULL;
    cudaMalloc((void**)&device_out,block_num*sizeof(float));
    float *res = (float*) malloc(block_num*sizeof(float)); // reference res


    // initial host_a
    for(int i=0;i<N;i++) {
        host_a[i] = 1.0;
    }
    
    // calcuate res
    for (int i=0; i<block_num; i++) {
        float tmp = 0;
        for (int j =0; j<THREAD_PER_BLOCK; j++) {
            tmp+=host_a[i*THREAD_PER_BLOCK+j];
        }
        res[i] = tmp;
    }

    // copy host_a to device
    cudaMemcpy(device_a,host_a,N*sizeof(float),cudaMemcpyHostToDevice);

    // Set cuda kernel size
    dim3 Grid(block_num,1);
    dim3 Block(THREAD_PER_BLOCK,1);


    // run kernel
    run(0,block_num, 1000, Grid, Block, device_a, device_out, host_out, res);

    //  clean the ptr
    memset(host_out,0,block_num*sizeof(float));
    cudaMemset(device_out,0,block_num*sizeof(float));
    run(1,block_num, 1000,  Grid, Block, device_a, device_out, host_out, res);


    //  clean the ptr
    memset(host_out,0,block_num*sizeof(float));
    cudaMemset(device_out,0,block_num*sizeof(float));
    run(2,block_num, 1000, Grid, Block, device_a, device_out, host_out, res);

    //  clean the ptr
    memset(host_out,0,block_num*sizeof(float));
    cudaMemset(device_out,0,block_num*sizeof(float));
    run(3,block_num, 1000,  Grid, Block, device_a, device_out, host_out, res);

    //  clean the ptr
    memset(host_out,0,block_num*sizeof(float));
    cudaMemset(device_out,0,block_num*sizeof(float));
    run(4,block_num, 1000,  Grid, Block, device_a, device_out, host_out, res);

   
    // 调整 block num 个数


    // output data
    int NUM_PER_BLOCK = M*THREAD_PER_BLOCK;
    int block_num_ = N / NUM_PER_BLOCK;
    float *host_out_ =  (float*) malloc(block_num_*sizeof(float));
    float *device_out_ = NULL;
    cudaMalloc((void**)&device_out_,block_num_*sizeof(float));
    float *res_ = (float*) malloc(block_num_*sizeof(float)); // reference res


        // calcuate res
    for (int i=0; i<block_num_; i++) {
        float tmp = 0;
        for (int j =0; j<NUM_PER_BLOCK; j++) {
            tmp+=host_a[i*NUM_PER_BLOCK+j];
        }
        res_[i] = tmp;
    }


    // Set cuda kernel size
    dim3 Grid_(block_num_,1);
    dim3 Block_(THREAD_PER_BLOCK,1);
    memset(host_out_,0,block_num_*sizeof(float));
    cudaMemset(device_out_,0,block_num_*sizeof(float));
    run(5,block_num_, 1000, Grid_, Block_, device_a, device_out_, host_out_, res_);





    //release ptr
    free(host_a);
    free(host_out);
    free(res);
    cudaFree(device_a);
    cudaFree(device_out);





}