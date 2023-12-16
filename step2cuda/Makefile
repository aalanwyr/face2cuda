all:
	nvcc -o ./bin/collect_device 2_4_collect_cuda_device.cu
	nvcc -o ./bin/recursive_hello 3_6_recursive_kernel.cu -lcudadevrt --relocatable-device-code true