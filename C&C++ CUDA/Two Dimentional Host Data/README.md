CUDA Learning Notes – Day 2

Today I practiced a few tasks with CUDA:

Converting a 2D Host matrix into a 1D Device array using cudaMalloc.

Implementing a matrix multiply-and-add operation using a 1D block.

Host-to-Device Conversion

When transferring a 2D Host matrix to Device memory, CUDA arranges the data row by row into a single linear 1D array:
