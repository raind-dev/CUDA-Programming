# CUDA Learning Notes – Day 6: 3D Memory Allocation with cudaMalloc3D

The focus of this practice is on using cudaMalloc3D to allocate 3D Device memory and implementing a Matrix Multiply-Add kernel based on that allocation.

Since cudaMalloc3D involves more complex memory management, it’s important to understand the related CUDA data structures used in the process — namely cudaMemcpy3DParms and cudaExtent.

## Understanding cudaMemcpy3DParms and cudaExtent

The cudaMemcpy3DParms structure contains the following key members:

- extent → (cudaExtent)

- dstPtr → (cudaPitchedPtr)

- srcPtr → (cudaPitchedPtr)

- kind → (cudaMemcpyKind)

cudaExtent

This structure defines the dimensions of the 3D memory space we want to allocate.
You can create it using the make_cudaExtent() function:

```cpp
cudaExtent extent = make_cudaExtent(width, height, depth);
```

Each argument corresponds to:

- width → number of columns

- height → number of rows

- depth → number of layers (or slices)

The unit size is based on the data type (e.g., sizeof(int) for integers).

## Understanding cudaPitchedPtr

Similar to cudaMallocPitch used for 2D memory allocation, cudaMalloc3D also aligns memory using pitch to improve GPU memory access efficiency through memory coalescing.

Both srcPtr and dstPtr in cudaMemcpy3DParms are of type cudaPitchedPtr, which can be created using make_cudaPitchedPtr():

```cpp
cudaPitchedPtr pitchedPtr = make_cudaPitchedPtr(
    hostBuffer, width, height, depth
);
```

Here:

- hostBuffer → pointer to the Host memory buffer

- width, height, depth → dimensions of the 3D data (in bytes)

## Understanding cudaMemcpyKind

The kind member specifies the direction of the memory transfer:

- cudaMemcpyHostToDevice → copy data from Host to Device

- cudaMemcpyDeviceToHost → copy data from Device to Host

This ensures CUDA knows how to handle srcPtr and dstPtr during the 3D memory copy operation.

👉 Future work: design a 3D Matrix Multiply-Add kernel using this allocation scheme and visualize how data is laid out across the three dimensions in Device memory.