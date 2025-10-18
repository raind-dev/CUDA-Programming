# CUDA Learning Notes – Day 5: Matrix Multiply-Add with Shared Memory Optimization

In the previous example — Matrix Multiply-Add, we observed that each element in the matrix is repeatedly fetched and multiplied during computation.
Besides improving performance through 2D thread blocks, we can further optimize the kernel by utilizing shared memory to reduce access time to global memory.

## Implementation Overview

In this exercise, I implemented a shared-memory-based matrix multiply-add kernel, while still keeping the original kernel for result verification and comparison.

Since the block size is identical to the matrix size, all threads in the block can collectively copy the entire matrix into shared memory.
Each thread is responsible for copying one element at its corresponding position.

Synchronization with __syncthreads()

To ensure correctness, we use the __syncthreads() instruction to synchronize all threads after copying data into shared memory.

For safety, I also inserted __syncthreads() calls after each multiply-add operation on shared memory.
However, strictly speaking, this additional synchronization is unnecessary, since the shared memory data is not overwritten afterward.

👉 Next step: measure performance differences between global memory and shared memory implementations, and analyze how memory coalescing affects execution time.