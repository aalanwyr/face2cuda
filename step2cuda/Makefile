all:
	nvcc -o ./bin/collect_device 2_4_collect_cuda_device.cu
	nvcc -o ./bin/tuning_kernel_size 3_3_tuning_kernel_size.cu
	nvcc -o ./bin/recursive_hello 3_6_recursive_kernel.cu -lcudadevrt --relocatable-device-code true
	nvcc -o ./bin/uniform_memory 4_5_uniform_memory.cu
	nvcc -o ./bin/global_memory_disable_l1 -O3 -Xptxas -dlcm=cg 4_3_global_memory.cu
	nvcc -o ./bin/global_memory_enable_l1 -O3 -Xptxas -dlcm=ca 4_3_global_memory.cu
	nvcc -o ./bin/global_memory_enable_l1_aos -O3 -Xptxas -dlcm=ca 4_3_array_of_struct.cu
	nvcc -o ./bin/5_2_shared_mem_read_data 5_2_shared_mem_read_data.cu
