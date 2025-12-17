# CUDA Learning Notes – Day 9

This entry is just a quick snapshot of my notes for future reference.

## CUDA Kernel Design Thought Process

1. How large is the data?
(e.g., the dimensions of the input matrices)

2. What computation is required?
(e.g., matrix multiply-add)

3. If the data size is large than total block size 1024:
- i. Consider how to apply tiling: block size (SM tiling) and shared memory size (GM tiling).
- ii. For example, if the block size is 32 × 32 and the shared memory capacity is 128 × 96 (48 KB on an RTX 4070 Ti), how should the tiling strategy be designed?

4. How should the blocks and grid be configured?
- i. The grid × block configuration must fully cover the entire dataset.
- ii. Consider the relationship between block size and shared memory usage (each block has its own independent shared memory).

## Notes

1. When declaring two dynamically allocated shared memory buffers without indexing them separately, they will initially point to the same memory region. For example:

```cpp
extern __shared__ int sm_mat[];
extern __shared__ int sm_mul[];
```

Both refer to the same base address unless manually offset.

2. Memory Coalescing principles also apply when designing the grid and block layout.
At the hardware level, GPU execution is organized into warps, each dispatching 32 threads.
To maximize efficiency:

- Ensure threads access contiguous memory, and

- Favor block and grid dimensions that align with multiples of 32.

3. Since I evaluated the time consumption of Day 8 example, and found that the kernel with shared memory is slower than the kernel without shared memory, so probably the kernel with shared memory should be fine-tune with a larger shared memory to gain the maximum benifit from using shared memory.  