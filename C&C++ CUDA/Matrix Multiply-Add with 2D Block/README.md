# CUDA Learning Notes – Day 3

In this session, I extended the Day 2 – Matrix Multiply-Add with 1D Block example into a 2D Block design.

Instead of modifying the original kernel, I kept the 1D kernel unchanged and wrote a new kernel dedicated to 2D block. The idea is to validate the correctness of the new 2D kernel by comparing its results with the original 1D kernel.

## Block Design

The block size is 3×3, which fully covers the target output data.

With this design, the kernel algorithm is updated as illustrated below:

```cpp
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
```

![Matrix Multiply-Add Version 2](../img/Matrix%20Multiply%20Addtion%20V2.jpg)

## Motivation

The purpose of this design is to explore how 2D thread blocks can better map to matrix operations, reducing complexity and making the implementation more scalable compared to using only 1D blocks.