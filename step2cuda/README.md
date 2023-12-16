According to the  https://face2ai.com/program-blog/

# 2_4_collect_cuda_device.cu
This is a demo to collect CUDA device details.
```
./bin/collect_device 
There are 1 available CUDA devices
Current active Device 0:"NVIDIA A10"
  CUDA Driver Version / Runtime Version         12.2  /  12.2
  CUDA Capability Major/Minor version number:   8.6
  Total amount of global memory:                21.99 GBytes (23609475072 bytes)
  GPU Clock rate:                               1695 MHz (1.70 GHz)
  Memory Bus width:                             384-bits
  L2 Cache Size:                                6291456 bytes
  Max Texture Dimension Size (x,y,z)            1D=(131072),2D=(131072,65536),3D=(16384,16384,16384)
  Max Layered Texture Size (dim) x layers       1D=(32768) x 2048,2D=(32768,32768) x 2048
  Total amount of constant memory               65536 bytes
  Total amount of shared memory per block:      49152 bytes
  Total number of registers available per block:65536
  Threads Wrap size:                            32
  Maximun number of thread per multiprocesser:  1536
  Maximun number of thread per block:           1024
  Maximun size of each dimension of a block:    1024 x 1024 x 64
  Maximun size of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximu memory pitch                           2147483647 bytes
```

# 3_6_recursive_kernel.cu
This is a demo to call a kernel in the cuda kernel recursively
We need to use `cudaDeviceSynchronize()` to sync the thread execution  between Host and GPU;
There is an implicit sync in the recursion thread. So we don't need `__syncthreads()`.
```
./bin/recursive_hello 
Current dept 0 blockIdx 0 threadIdx 0 tid 0
Current dept 0 blockIdx 0 threadIdx 1 tid 1
Current dept 0 blockIdx 0 threadIdx 2 tid 2
Current dept 0 blockIdx 0 threadIdx 3 tid 3
Current dept 0 blockIdx 0 threadIdx 4 tid 4
Current dept 0 blockIdx 0 threadIdx 5 tid 5
Current dept 0 blockIdx 0 threadIdx 6 tid 6
Current dept 0 blockIdx 0 threadIdx 7 tid 7
Current dept 0 blockIdx 0 threadIdx 8 tid 8
Current dept 0 blockIdx 0 threadIdx 9 tid 9
Current dept 0 blockIdx 0 threadIdx 10 tid 10
Current dept 0 blockIdx 0 threadIdx 11 tid 11
Current dept 0 blockIdx 0 threadIdx 12 tid 12
Current dept 0 blockIdx 0 threadIdx 13 tid 13
Current dept 0 blockIdx 0 threadIdx 14 tid 14
Current dept 0 blockIdx 0 threadIdx 15 tid 15
Current dept 0 blockIdx 0 threadIdx 16 tid 16
Current dept 0 blockIdx 0 threadIdx 17 tid 17
Current dept 0 blockIdx 0 threadIdx 18 tid 18
Current dept 0 blockIdx 0 threadIdx 19 tid 19
Current dept 0 blockIdx 0 threadIdx 20 tid 20
Current dept 0 blockIdx 0 threadIdx 21 tid 21
Current dept 0 blockIdx 0 threadIdx 22 tid 22
Current dept 0 blockIdx 0 threadIdx 23 tid 23
Current dept 0 blockIdx 0 threadIdx 24 tid 24
Current dept 0 blockIdx 0 threadIdx 25 tid 25
Current dept 0 blockIdx 0 threadIdx 26 tid 26
Current dept 0 blockIdx 0 threadIdx 27 tid 27
Current dept 0 blockIdx 0 threadIdx 28 tid 28
Current dept 0 blockIdx 0 threadIdx 29 tid 29
Current dept 0 blockIdx 0 threadIdx 30 tid 30
Current dept 0 blockIdx 0 threadIdx 31 tid 31
-----------> nested execution depth: 1
Current dept 1 blockIdx 0 threadIdx 0 tid 0
Current dept 1 blockIdx 0 threadIdx 1 tid 1
Current dept 1 blockIdx 0 threadIdx 2 tid 2
Current dept 1 blockIdx 0 threadIdx 3 tid 3
Current dept 1 blockIdx 0 threadIdx 4 tid 4
Current dept 1 blockIdx 0 threadIdx 5 tid 5
Current dept 1 blockIdx 0 threadIdx 6 tid 6
Current dept 1 blockIdx 0 threadIdx 7 tid 7
Current dept 1 blockIdx 0 threadIdx 8 tid 8
Current dept 1 blockIdx 0 threadIdx 9 tid 9
Current dept 1 blockIdx 0 threadIdx 10 tid 10
Current dept 1 blockIdx 0 threadIdx 11 tid 11
Current dept 1 blockIdx 0 threadIdx 12 tid 12
Current dept 1 blockIdx 0 threadIdx 13 tid 13
Current dept 1 blockIdx 0 threadIdx 14 tid 14
Current dept 1 blockIdx 0 threadIdx 15 tid 15
-----------> nested execution depth: 2
Current dept 2 blockIdx 0 threadIdx 0 tid 0
Current dept 2 blockIdx 0 threadIdx 1 tid 1
Current dept 2 blockIdx 0 threadIdx 2 tid 2
Current dept 2 blockIdx 0 threadIdx 3 tid 3
Current dept 2 blockIdx 0 threadIdx 4 tid 4
Current dept 2 blockIdx 0 threadIdx 5 tid 5
Current dept 2 blockIdx 0 threadIdx 6 tid 6
Current dept 2 blockIdx 0 threadIdx 7 tid 7
-----------> nested execution depth: 3
Current dept 3 blockIdx 0 threadIdx 0 tid 0
Current dept 3 blockIdx 0 threadIdx 1 tid 1
Current dept 3 blockIdx 0 threadIdx 2 tid 2
Current dept 3 blockIdx 0 threadIdx 3 tid 3
-----------> nested execution depth: 4
Current dept 4 blockIdx 0 threadIdx 0 tid 0
Current dept 4 blockIdx 0 threadIdx 1 tid 1
-----------> nested execution depth: 5
Current dept 5 blockIdx 0 threadIdx 0 tid 0
```

