
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void traverse_device_data(int *device) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = x + y * blockDim.x;
    printf("%d ", device[idx]);
}

__global__ void matrix_operations_V1(int* mat, int* mul, int* scl, int* res, int N, int K, int W) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < K; i++) {
        int tmp = 0;
        for (int j = 0; j < N; j++) {
            tmp += mat[x * N + j] * mul[i + j * K];
        }
        tmp += scl[x * W + i];
        res[x * W + i] = tmp;
    }
}

__global__ void matrix_operations_V2(int* mat, int* mul, int* scl, int* res, int N, int K, int W) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tmp = 0;
    for (int i = 0; i < N; i++) {
        tmp += mat[y * N + i] * mul[i * K + x];
    }
    tmp += scl[y * N + x];
    res[y * N + x] = tmp;
}

bool is_cuda_available()
{
    int device = 0;
    cudaError_t status = cudaSetDevice(device); // cudaGetDeviceCount
    if (status != cudaSuccess) {
        fprintf(stderr, "Setting device failed! Please check whether CUDA-Capable GPU installed.\n");
        return false;
    }
    return true;
}

template<size_t C, size_t R>
bool allocate_host_data(int (&map)[R][C], const char* name)
{
    srand(time(NULL));
    printf("%s: \n", name);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            map[i][j] = rand() % 2;
            printf("%d ", map[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    return true;
}

bool allocate_device_data(int** device_buffer, int col, int row)
{
	cudaError_t status = cudaMalloc(device_buffer, sizeof(int) * col * row);
	if (status != cudaSuccess) {
		fprintf(stderr, "Allocate device data failed: %d\n", status);
		return false;
	}
    return true;
}

template<size_t C, size_t R>
bool copy_host_data_to_device(int (&host_buffer)[R][C], int** device_buffer, const char* name)
{
	cudaError_t status = cudaMemcpy(*device_buffer, host_buffer, sizeof(int) * R * C, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
		fprintf(stderr, "Copy host data %s to device failed: %d\n", name, status);
		return false;
    }
    return true;
}

int main()
{
    if (!is_cuda_available()) {
        return -1;
    }
    const int col = 3, row = 3;
    int mat[row][col];
    int mul[row][col];
    int scl[row][col];
    int res[9] = { 0 };
    int* gpu_mat = nullptr;
    int* gpu_mul = nullptr;
    int* gpu_scl = nullptr;
    int* gpu_res = nullptr;
    if (!allocate_host_data(mat, "Matrix") ||
        !allocate_host_data(mul, "Multiplier") ||
        !allocate_host_data(scl, "Scalar")) {
        return -1;
    }
    if (!allocate_device_data(&gpu_mat, row, col) ||
        !allocate_device_data(&gpu_mul, row, col) ||
        !allocate_device_data(&gpu_scl, row, col) ||
        !allocate_device_data(&gpu_res, row, col)) {
        return -1;
    }
	if (!copy_host_data_to_device(mat, &gpu_mat, "Matrix") ||
        !copy_host_data_to_device(mul, &gpu_mul, "Multiplier") ||
        !copy_host_data_to_device(scl, &gpu_scl, "Scalar")) {
        return -1;
    }
    dim3 grid = 1;
    dim3 block_v1 = 3;
	dim3 block_v2(3, 3);
    matrix_operations_V1 << < grid, block_v1 >> > (gpu_mat, gpu_mul, gpu_scl, gpu_res, row, col, col);
    cudaDeviceSynchronize();

    cudaError_t status = cudaMemcpy(res, gpu_res, sizeof(int) * col * col, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "Can not copy the results from device to host\n");
        return -1;
    }
    printf("\n\nThe V1 Results: \n");
    for (int i = 0; i < sizeof(res) / sizeof(int); i++) {
        printf("%d ", res[i]);
        if (i % col == 2) {
            printf("\n");
        }
    }

    matrix_operations_V2 << < grid, block_v2 >> > (gpu_mat, gpu_mul, gpu_scl, gpu_res, row, col, col);
    cudaDeviceSynchronize();

    status = cudaMemcpy(res, gpu_res, sizeof(int) * col * col, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "Can not copy the results from device to host\n");
        return -1;
    }
    printf("\n\nThe V2 Results: \n");
    for (int i = 0; i < sizeof(res) / sizeof(int); i++) {
        printf("%d ", res[i]);
        if (i % col == 2) {
            printf("\n");
        }
    }

    cudaFree(gpu_mat);
	cudaFree(gpu_mul);
	cudaFree(gpu_scl);
    cudaFree(gpu_res);

    return 0;
}
