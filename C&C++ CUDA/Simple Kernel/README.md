# CUDA Learning Notes – Day 1
This practice focuses on the basic of CUDA kernel workflow.

## Workflow
The workflow includes the following steps:
1. Check CUDA computation availability.
2. Allocate host data (on the CPU).
3. Allocate device memory (using the CUDA Runtime API).
4. Copy host data to device memory.
5. Configure the block and grid dimensions.
6. Launch the CUDA kernel.
7. Synchronize the host and device.