
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

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

__global__ void matrix_operations_with_S3D(int* mat, int* mul, int* scl, int* res, size_t pitch, size_t pitch_slice, int col, int row, int depth)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ int mat_shared[3][3][3];
    __shared__ int mul_shared[3][3][3];
    __shared__ int scl_shared[3][3][3];
    for (int z = 0; z < 3; z++) {
        if (x < col && y < row) {
			char* slice = (char*)mat + z * pitch_slice;
			int* row = (int*)(slice + y * pitch);
            mat_shared[z][y][x] = row[x];
			slice = (char*)mul + z * pitch_slice;
			row = (int*)(slice + y * pitch);
            mul_shared[z][y][x] = row[x];
			slice = (char*)scl + z * pitch_slice;
			row = (int*)(slice + y * pitch);
            scl_shared[z][y][x] = row[x];
        }
    }
    __syncthreads();
    for (int z = 0; z < 3; z++) {
        int tmp = 0;
        for (int i = 0; i < col; i++) {
            tmp += mat_shared[z][y][i] * mul_shared[z][i][x];
        }
        __syncthreads();
        tmp += scl_shared[z][y][x];
		char* slice = (char*)res + z * pitch_slice;
		int* row = (int*)(slice + y * pitch);
        row[x] = tmp;
    }
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

template<size_t C, size_t R, size_t D>
bool allocate_host_data_with_3D(int (&cube)[D][R][C], const char* name)
{
    printf("%s: \n", name);
	//srand((unsigned)time(NULL));
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < D; j++) {
			for (int k = 0; k < C; k++) {
				cube[j][i][k] = rand() % 2;
				printf("%d ", cube[j][i][k]);
			}
			printf("  ");
		}
		printf("\n");
	}
	printf("\n");
	return true;
}

bool allocate_device_data_with_3D(cudaMemcpy3DParms &params, int col, int row, int depth)
{
	cudaExtent extent = make_cudaExtent(col * sizeof(int), row, depth);
	cudaPitchedPtr pitch; // Y, Z axis pitch
	cudaError_t status = cudaMalloc3D(&pitch, extent);
	params.extent = extent;
	params.dstPtr = pitch;
	if (status != cudaSuccess) {
		fprintf(stderr, "Allocate device data with 3D failed: %d\n", status);
		return false;
	}
	return true;
}

template<size_t C, size_t R, size_t D>
bool copy_host_data_to_device_with_3D(int(&host_buffer)[D][R][C], cudaMemcpy3DParms& params, const char* name)
{
	params.srcPtr = make_cudaPitchedPtr((void*)host_buffer, C * sizeof(int), C, R);
	params.kind = cudaMemcpyHostToDevice;
    cudaError_t status = cudaMemcpy3D(&params);
    if (status != cudaSuccess) {
        fprintf(stderr, "Copy host data %s to device with 3D failed: %d\n", name, status);
        return false;
    }
    return true;
}

int main()
{
    if (!is_cuda_available()) {
        return -1;
    }
    const int col = 3, row = 3, depth = 3;
    int mat[depth][row][col];
    int mul[depth][row][col];
    int scl[depth][row][col];
    int res[depth][row][col] = {0};
	cudaMemcpy3DParms mat_params = { 0 }, mul_params = { 0 }, scl_params = { 0 }, res_params = { 0 };
    if (!allocate_host_data_with_3D(mat, "Matrix") ||
        !allocate_host_data_with_3D(mul, "Multiplier") ||
        !allocate_host_data_with_3D(scl, "Scalar")) {
        return -1;
    }
    if (!allocate_device_data_with_3D(mat_params, col, row, depth) ||
        !allocate_device_data_with_3D(mul_params, col, row, depth) ||
        !allocate_device_data_with_3D(scl_params, col, row, depth) ||
        !allocate_device_data_with_3D(res_params, col, row, depth)) {
        return -1;
    }
	if (!copy_host_data_to_device_with_3D(mat, mat_params, "Matrix") ||
		!copy_host_data_to_device_with_3D(mul, mul_params, "Multiplier") ||
		!copy_host_data_to_device_with_3D(scl, scl_params, "Scalar")) {
		return -1;
	}
    dim3 grid = 1;
    dim3 block(3, 3);

    matrix_operations_with_S3D << < grid, block >> > ((int*)mat_params.dstPtr.ptr, (int*)mul_params.dstPtr.ptr, (int*)scl_params.dstPtr.ptr, (int*)res_params.dstPtr.ptr, mat_params.dstPtr.pitch, mat_params.dstPtr.pitch * row, col, row, depth);
    cudaDeviceSynchronize();

	res_params.kind = cudaMemcpyDeviceToHost;
	res_params.srcPtr = res_params.dstPtr;
	res_params.dstPtr = make_cudaPitchedPtr((void*)res, col * sizeof(int), col, row);
    cudaError_t status = cudaMemcpy3D(&res_params);
    printf("\n\nThe Results: \n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < depth; j++) {
            for (int k = 0; k < col; k++) {
                printf("%d ", res[j][i][k]);
            }
            printf("  ");
        }
        printf("\n");
    }
    printf("\n");

    cudaFree(mat_params.dstPtr.ptr);
	cudaFree(mul_params.dstPtr.ptr);
	cudaFree(scl_params.dstPtr.ptr);
    cudaFree(res_params.dstPtr.ptr);

    return 0;
}
