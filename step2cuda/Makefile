all:
	nvcc -o ./bin/collect_device 2_4_collect_cuda_device.cu
	nvcc -o ./bin/tuning_kernel_size 3_3_tuning_kernel_size.cu
	nvcc -o ./bin/recursive_hello 3_6_recursive_kernel.cu -lcudadevrt --relocatable-device-code true