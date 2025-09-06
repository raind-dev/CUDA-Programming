/*
 * This file simply demonstrates how to use one-dimension kernel (8-threads) 
 * to calculate one-dimension array. 	
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void basic_add(float* ga, float* gb, float* gc)
{
	int idx = threadIdx.x;
	gc[idx] = ga[idx] + gb[idx];
}

bool GPU_Memory_Allocation(void** ga, void** gb, void** gc, const int size)
{
	cudaError_t status = cudaMalloc(ga, size * sizeof(float));
	if (status != cudaSuccess) {
		fprintf(stderr, "Allocate pointer ga failed\n");
		return false;
	}

	status = cudaMalloc(gb, size * sizeof(float));
	if (status != cudaSuccess) {
		fprintf(stderr, "Allocate pointer gb failed\n");
		return false;
	}

	status = cudaMalloc(gc, size * sizeof(float));
	if (status != cudaSuccess) {
		fprintf(stderr, "Allocate pointer gc failed\n");
		return false;
	}

	return true;
}

bool CPU_Data_To_GPU(const void* ca, const void* cb, void* ga, void* gb, const int size)
{
	cudaError_t status = cudaMemcpy(ga, ca, size * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "CPU data from ca to ga failed! Error: %d\n", status);
		return false;
	}

	status = cudaMemcpy(gb, cb, size * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "CPU data from cb to gb failed! Error: %d\n", status);
		return false;
	}

	return true;
}

void Free_Memory(void* ga, void* gb, void* gc)
{
	cudaFree(ga);
	cudaFree(gb);
	cudaFree(gc);
}

void Print_Results(float* c, const int size)
{
	for (int i = 0; i < size; i++) {
		std::cout << c[i] << ", ";
	}
}

bool To_Device()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return false;
	}
	return true;
}

int main() {
	const int size = 8;
	const float A[size] = {1.2, 2.3, 3.3, 2.1, 5.2, 6.4, 2.5, 5.2};
	const float B[size] = { 1.2, 2.3, 3.3, 2.1, 5.2, 6.4, 2.5, 5.2 };
	float C[size];
	float* ga = nullptr, * gb = nullptr, *gc = nullptr;
	
	if (!To_Device()) {
		return 0;
	}

	if (!GPU_Memory_Allocation((void**)&ga, (void**)&gb, (void**)&gc, size)) {
		Free_Memory(ga, gb, gc);
		return 0;
	}

	if (!CPU_Data_To_GPU(A, B, ga, gb, size)) {
		Free_Memory(ga, gb, gc);
		return 0;
	}

	dim3 gridDim = 1; 
	dim3 blockDim = size; 
	basic_add <<< gridDim, blockDim >>> (ga, gb, gc);
	cudaDeviceSynchronize();

	cudaMemcpy(C, gc, size*sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << "Start to print the GPU results" << std::endl;
	Print_Results(C, size);
	
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	Free_Memory(ga, gb, gc);

	return 0;
}
