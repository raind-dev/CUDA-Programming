# CUDA Learning Notes – Day 7: Matrix Multiplication with Shared Memory Tiling

The focus of this practice is implementing matrix multiplication using Shared Memory + Tiling.

From previous exercises, we already know that when a kernel repeatedly reads the same data from Global Memory, we can use Shared Memory to significantly reduce memory access time.

However, when the dataset (matrix) becomes larger than the available shared memory, we must divide the data into smaller tiles and let each kernel process a portion of it.
This approach is known as Shared Memory Tiling.

## Concept Overview

Let’s assume we want to multiply two 6×6 matrices.
If our block size is 3×3, and each block uses a 3×3 shared memory, then the grid size will be 2×2, as illustrated below:

![Shared Memory Tiling](../img/Shared%20Memory%20Tiling.jpg)

This means we only need 4 blocks to cover the entire output matrix.

## Block Movement and Data Loading

To correctly perform matrix multiplication, the block must move across different regions of the input matrices.
If a block always reads data from the same position, it cannot complete the entire multiplication.

Therefore, we design the block to shift its data window during computation — as shown in the figure below:

![Shared Memory Tiling](../img/Shared%20Memory%20Tiling%20Multiply.jpg)

When the block is at grid position (0, 0), it performs multiplication on the two current tiles, storing partial results into a temporary variable.
Then, as the block shifts its position, it continues to perform multiplication on the next tiles, accumulating the results into that temporary variable.

## Calculating Block Shifts

The movement of each block differs between the two matrices being multiplied, so we need to carefully handle the indexing.

We can determine a block’s position in the grid using blockIdx.x and blockIdx.y:

- For the first matrix, blockIdx.x is not directly meaningful, since we shift the block’s column position during each iteration of the multiplication loop.
Only when blockIdx.y changes (from 0 to 1) do we move to a new row.

- For the second matrix, the situation is reversed — blockIdx.x becomes important, since each loop iteration shifts the block’s row position.
Thus, when blockIdx.x changes from 0 to 1, we move to a new row region in the second matrix.

## Summary

This tiling strategy allows each block to reuse data from shared memory efficiently while coordinating movement across both matrices to complete the full multiplication.
It’s a key step toward optimizing large-scale matrix operations in CUDA.