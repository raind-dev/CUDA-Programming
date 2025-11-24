
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void matrix_operations(int* mat, int* mul, int* res, int col, size_t pitch) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int mat_pitch_idx = pitch / sizeof(int);
    int mul_pitch_idx = pitch / sizeof(int);
    int res_pitch_idx = pitch / sizeof(int);
    int tmp = 0;
    for (int i = 0; i < col; i++) {
        tmp += mat[y * mat_pitch_idx + i] * mul[i * mul_pitch_idx + x];
    }
    res[y * res_pitch_idx + x] = tmp;
}

__global__ void matrix_operations_with_shared_memory_and_tiling(int* mat, int* mul, int* res, int col_row, size_t pitch)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    const int tile_dim = 32;
    int pitch_idx = pitch / sizeof(int);
    int tile_number = (col_row + tile_dim - 1) / tile_dim;
    __shared__ int mat_shared[tile_dim][tile_dim];
    __shared__ int mul_shared[tile_dim][tile_dim];
    int tiler_ty = by * tile_dim + y; // Actually, tile_dim = blockDim
    int tiler_tx = bx * tile_dim + x;
    int tmp = 0;
    for (int i = 0; i < tile_number; i++) { // total run = 2
        if (x < col_row && y < col_row) {
            int tiler_shift = i * tile_dim;
            mat_shared[y][x] = mat[tiler_ty * pitch_idx + tiler_shift + x];
            mul_shared[y][x] = mul[tiler_tx + (tiler_shift + y) * pitch_idx];
        }
        __syncthreads();
        for (int j = 0; j < tile_dim; j++) {
            tmp += mat_shared[y][j] * mul_shared[j][x];
        }
        __syncthreads();
    }
    int gx = bx * blockDim.x + x;
    int gy = by * blockDim.y + y;
    res[gy * pitch_idx + gx] = tmp;
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
bool traverse_host_data(int(&map)[R][C], const char* name)
{
    printf("%s: \n", name);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%d ", map[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    return true;
}

template<size_t C, size_t R>
bool allocate_host_data(int(&map)[R][C], const char* name)
{
    srand(time(NULL));
    printf("%s: \n", name);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            map[i][j] = rand() % 2;
            //printf("%d ", map[i][j]);
        }
        //printf("\n");
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
bool copy_host_data_to_device(int(&host_buffer)[R][C], int** device_buffer, const char* name)
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
    const int col = 2048, row = 2048;
    static int mat[row][col] = { 0 };
    static int mul[row][col] = { 0 };
    static int res[row][col] = { 0 };
    int* gpu_mat = nullptr;
    int* gpu_mul = nullptr;
    int* gpu_res = nullptr;
    size_t mat_pitch, mul_pitch, scl_pitch, res_pitch;
    if (!allocate_host_data(mat, "Matrix") ||
        !allocate_host_data(mul, "Multiplier")) {
        return -1;
    }
    if (!allocate_device_data_with_pitch(&gpu_mat, mat_pitch, col, row, sizeof(int)) ||
        !allocate_device_data_with_pitch(&gpu_mul, mul_pitch, col, row, sizeof(int)) ||
        !allocate_device_data_with_pitch(&gpu_res, res_pitch, col, row, sizeof(int))) {
        return -1;
    }
    if (!copy_host_data_to_device_with_pitch(mat, &gpu_mat, mat_pitch, "Matrix") ||
        !copy_host_data_to_device_with_pitch(mul, &gpu_mul, mul_pitch, "Multiplier")) {
        return -1;
    }
    dim3 grid(64, 64);
    dim3 block(32, 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_operations_with_shared_memory_and_tiling << < grid, block >> > (gpu_mat, gpu_mul, gpu_res, col, mat_pitch);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms1 = 0.0f;
    cudaEventElapsedTime(&ms1, start, stop);
    std::cout << "Matrix multiplication with shared memory and tiling elapsed time: " << ms1 << " ms" << std::endl;

    cudaEventRecord(start);
    matrix_operations << < grid, block >> > (gpu_mat, gpu_mul, gpu_res, col, mat_pitch);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms2 = 0.0f;
    cudaEventElapsedTime(&ms2, start, stop);
    std::cout << "Matrix multiplication without shared memory elapsed time: " << ms2 << " ms" << std::endl;

    cudaFree(gpu_mat);
    cudaFree(gpu_mul);
    cudaFree(gpu_res);

    return 0;
}