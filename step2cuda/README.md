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

# 3_3_tuning_kernel_size.cu
This is a demo to check the different performance of different kernel size
```
./bin/tuning_kernel_size 32 16
Current matrix size is 256 MBs
Check result success!
GPU Execution configuration<<<(256,512),(32,16)|1.764096 ms

./bin/tuning_kernel_size 16 32
Current matrix size is 256 MBs
Check result success!
GPU Execution configuration<<<(512,256),(16,32)|1.792320 ms

./bin/tuning_kernel_size 16 16
Current matrix size is 256 MBs
Check result success!
GPU Execution configuration<<<(512,512),(16,16)|1.786656 ms

./bin/tuning_kernel_size 16 8
Current matrix size is 256 MBs
Check result success!
GPU Execution configuration<<<(512,1024),(16,8)|1.802656 ms

./bin/tuning_kernel_size 256 4
Current matrix size is 256 MBs
Check result success!
GPU Execution configuration<<<(32,2048),(256,4)|1.773120 ms

./bin/tuning_kernel_size 64 2
Current matrix size is 256 MBs
Check result success!
GPU Execution configuration<<<(128,4096),(64,2)|1.780096 ms

 /bin/tuning_kernel_size 256 8
Current matrix size is 256 MBs
Results don't match!
20.989000(hostRef[0] )!= 0.000000(gpuRef[0])
GPU Execution configuration<<<(32,1024),(256,8)|0.128512 ms
```
## configuration<<<(32,1024),(256,8) failed
    The number of threads in a block should be lower 1024 or the kernel would fail!
    Meanwhile the result check is necessary
## Use nv prof tool to dump details
```
Check active thread:
nvprof --metrics achieved_occupancy ./simple_sum_matrix
===> sudo ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./bin/tuning_kernel_size 256 4
Current matrix size is 256 MBs
==PROF== Connected to process 432610 (/home/gta/yaranwu/face2cuda/step2cuda/bin/tuning_kernel_size)
ERROR: 3_3_tuning_kernel_size.cu:61,code:36,reason:API call is not supported in the installed CUDA driver
==PROF== Disconnected from process 432610
==ERROR== The application returned an error code (1).
==WARNING== No kernels were profiled.
==WARNING== Profiling kernels launched by child processes requires the --target-processes all option.

Check global memory throughput:
nvprof --metrics gld_throughput ./simple_sum_matrix
===> sudo ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second /bin/tuning_kernel_size 256 4

Check global memory efficiency:
nvprof --metrics gld_efficiency ./simple_sum_matrix
===> sudo ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct /bin/tuning_kernel_size 256 4

More details:
sudo ncu --list-metrics

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

