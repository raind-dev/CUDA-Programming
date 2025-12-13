# CUDA Learning Notes – Day 4

The focus of this practice is on using cudaMallocPitch to allocate 2D GPU memory and exploring how it differs from the standard cudaMalloc allocation.
Additionally, I examined how memory addresses are arranged when allocated with cudaMallocPitch.

## Memory Allocation & Data Transfer

Since this exercise uses cudaMallocPitch, the data transfer between Host and Device is handled with cudaMemcpy2D instead of cudaMemcpy.

For this purpose, I added two helper functions:

```cpp
allocate_device_data_with_pitch();
copy_host_data_to_device_with_pitch();
```

## Address Traversal
I also implemented a utility function to walk through the pitched memory layout:

```cpp
traverse_device_data_address_by_pitch();
```

The printed output (see figure) shows that:

![cudaMallocPitch Log](../img/cudaMallocPitch%20log.jpg)

Each row occupies 512 bytes.

The starting address of each row is exactly 512 bytes (0x200) apart.

## Kernel Considerations

Inside the kernel:

- The device pointer is first cast to char* before adding the pitch.

- This is required because pitch is expressed in size_t units.

- Without the cast, directly using an int* for index arithmetic would produce incorrect address calculations.

## Pitch-Aware Kernel Parameters

Since the Device matrix space is no longer a perfect match with the Host matrix dimensions, we now need to explicitly account for pitch.
Therefore, the kernel parameters must use pitch as the effective matrix width when performing indexing.