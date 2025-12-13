
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void matrix_operations(int* mat, int* mul, int* scl, int* res, int col, size_t mat_pitch, size_t mul_pitch, size_t res_pitch) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int mat_pitch_idx = mat_pitch / sizeof(int);
	int mul_pitch_idx = mul_pitch / sizeof(int);
	int res_pitch_idx = res_pitch / sizeof(int);
    int tmp = 0;
    for (int i = 0; i < col; i++) {
        tmp += mat[y * mat_pitch_idx + i] * mul[i * mul_pitch_idx + x];
    }
    tmp += scl[y * res_pitch_idx + x];
    res[y * res_pitch_idx + x] = tmp;
}

__global__ void matrix_operations_with_shared_memory(int* mat, int* mul, int* scl, int* res, int col, size_t mat_pitch, size_t mul_pitch, size_t res_pitch)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int mat_pitch_idx = mat_pitch / sizeof(int);
    int mul_pitch_idx = mul_pitch / sizeof(int);
    int res_pitch_idx = res_pitch / sizeof(int);
	__shared__ int mat_shared[3][3];
	__shared__ int mul_shared[3][3];
	__shared__ int scl_shared[3][3];
    if (x < col && y < col) {
            mat_shared[y][x] = mat[y * mat_pitch_idx + x];
            mul_shared[y][x] = mul[y * mul_pitch_idx + x];
			scl_shared[y][x] = scl[y * res_pitch_idx + x];
    }
	__syncthreads();
	int tmp = 0;
	for (int i = 0; i < col; i++) {
		tmp += mat_shared[y][i] * mul_shared[i][x];
	}
    __syncthreads();
	tmp += scl_shared[y][x];
	res[y * res_pitch_idx + x] = tmp;
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
    printf("%s: \n", name);
	//srand((unsigned)time(NULL));
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

bool allocate_device_data_with_pitch(int** device_buffer, size_t& pitch, int col, int row, const size_t& size)
{
    cudaError_t status = cudaMallocPitch(device_buffer, &pitch, col * size, row);
    if (status != cudaSuccess) {
        fprintf(stderr, "Allocate device data with pitch failed: %d\n", status);
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

template<size_t C, size_t R>
bool copy_host_data_to_device_with_pitch(int(&host_buffer)[R][C], int** device_buffer, size_t pitch, const char* name)
{
	cudaError_t status = cudaMemcpy2D(*device_buffer, pitch, host_buffer, C * sizeof(int), C * sizeof(int), R, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "Copy host data %s to device with pitch failed: %d\n", name, status);
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
    size_t mat_pitch, mul_pitch, scl_pitch, res_pitch;
    if (!allocate_host_data(mat, "Matrix") ||
        !allocate_host_data(mul, "Multiplier") ||
        !allocate_host_data(scl, "Scalar")) {
        return -1;
    }
    if (!allocate_device_data_with_pitch(&gpu_mat, mat_pitch, col, row, sizeof(int)) || 
        !allocate_device_data_with_pitch(&gpu_mul, mul_pitch, col, row, sizeof(int)) ||
        !allocate_device_data_with_pitch(&gpu_scl, scl_pitch, col, row, sizeof(int)) ||
        !allocate_device_data_with_pitch(&gpu_res, res_pitch, col, row, sizeof(int))) {
        return -1;
    }
	if (!copy_host_data_to_device_with_pitch(mat, &gpu_mat, mat_pitch, "Matrix") ||
		!copy_host_data_to_device_with_pitch(mul, &gpu_mul, mul_pitch, "Multiplier") ||
		!copy_host_data_to_device_with_pitch(scl, &gpu_scl, scl_pitch, "Scalar")) {
		return -1;
	}
    //printf("Mat Pitch: %d, Mul Pitch: %d, Scl Pitch: %d, Res Pitch: %d\n", mat_pitch, mul_pitch, scl_pitch, res_pitch);
    dim3 grid = 1;
    dim3 block(3, 3);

    matrix_operations << < grid, block >> > (gpu_mat, gpu_mul, gpu_scl, gpu_res, col, mat_pitch, mul_pitch, res_pitch);
    cudaDeviceSynchronize();

	cudaError_t status = cudaMemcpy2D(res, col * sizeof(int), gpu_res, res_pitch, col * sizeof(int), row, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "Can not copy the results from device to host\n");
        return -1;
    }
    printf("\n\nThe Original Results: \n");
    for (int i = 0; i < sizeof(res) / sizeof(int); i++) {
        printf("%d ", res[i]);
        if (i % col == 2) {
            printf("\n");
        }
    }

    matrix_operations_with_shared_memory << < grid, block >> > (gpu_mat, gpu_mul, gpu_scl, gpu_res, col, mat_pitch, mul_pitch, res_pitch);
    cudaDeviceSynchronize();

    status = cudaMemcpy2D(res, col * sizeof(int), gpu_res, res_pitch, col * sizeof(int), row, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "Can not copy the results from device to host\n");
        return -1;
    }
    printf("\n\nThe Shared Memory Results: \n");
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
