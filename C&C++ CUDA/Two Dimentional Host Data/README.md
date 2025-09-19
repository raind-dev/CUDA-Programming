# CUDA Learning Notes – Day 2
Today I practiced a few tasks with CUDA:

Converting a 2D Host matrix into a 1D Device array using cudaMalloc.

Implementing a matrix multiply-and-add operation using a 1D block.

## Host-to-Device Conversion
When transferring a 2D Host matrix to Device memory, CUDA arranges the data row by row into a single linear 1D array:

![Device Data Layout](https://github.com/raind-dev/CUDA-Programming/blob/main/C%26C%2B%2B%20CUDA/img/Device%20Data%20Layout.jpg)

So each row in Host memory is appended sequentially into the Device array.

## Matrix Multiply-and-Add
The computation formula can be expressed as:

![Matrix Multiply and Add](https://github.com/raind-dev/CUDA-Programming/blob/main/C%26C%2B%2B%20CUDA/img/Matrix%20Multiply%20Addtion.jpg)

![Matrix Multiply and Add V2](https://github.com/raind-dev/CUDA-Programming/blob/main/C%26C%2B%2B%20CUDA/img/Matrix%20Multiply%20Addtion%202.jpg)

## Thread Block Strategy
- Each thread is responsible for accessing a set of elements (highlighted by different colors in my diagram).

- The multiplier matrix is shared across all threads (represented in gray).

- The challenge:

    - The row-major layout of Host → Device conversion conflicts with the column access pattern needed during multiplication.

    - To handle this, we compute the proper stride when accessing elements of mul so that the traversal happens column-wise.

## Kernel Implementation
```cpp
__global__ void matrix_operations(int* mat, int* mul, int* scl, int* res, int N, int K, int W) {
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
```

## Threads Running and Outputs
There are some interesting things that I found, if we design the kernel to get the index from y-axis, and check threads running, you will see all of column threads in a row will be submitted in each time. 

![Threads Running Order 1](../img/Thread%20Running%201.jpg) ![Threads Running Order 2](../img/Thread%20Running%202.jpg)

And the second thing is that, if we add some output functions in CUDA kernel, like as below:
```cpp
__global__ void matMulAddKernel(int *res, const int *map, const int *mul, const int *s)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    printf("blockDim x: %d, X: %d , Y: %d\n", blockDim.x, x, y);
    x = x + y * blockDim.x;
    printf("Final x: %d\n", x);
    res[x] = map[x] * mul[x] + s[x];
}
```

The outputs order is not interlaced, it will be output as the first collection of outputs from all of threads, and then the second collection of outputs from all of threads. This is because of the threads running order and output logs from device will be stored in a buffer, once device running is done, then the buffer will be copied to host, and flush to stdout.

## Complexity & Next Steps
- Current time complexity: O(N²), since each thread does nested loops.

- Next goal: reduce complexity to O(N) by using 2D thread blocks.

- Future practice will also include:

    - cudaMallocPitch for 2D memory alignment

    - cudaMalloc3D for 3D memory allocations
