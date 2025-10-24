# CUDA Programming Examples & Practice

This repository collects my personal hands-on experiments and implementations related to **GPU programming** using:

- **C/C++ with CUDA**
- **Python with Numba**

The goal is to deepen understanding of parallel computing, GPU acceleration techniques, and real-world applications.

---

## Directory Structure

```bash
cuda-programming/
│
├── cpp_cuda/                                                   # C/C++ with CUDA
│   ├── Simple Kernel                                           # CUDA Learning Notes: Kernels (Day 1)
│   ├── Matrix Multiply-Add with 1D Block                       # CUDA Lenrning Notes: Kernels (Day 2)
│   ├── Matrix Multiply-Add with 2D Block                       # CUDA Learning Notes: Kernels (Day 3) 
│   ├── Exploring cudaMallocPitch vs cudaMalloc                 # CUDA Learning Notes: Kernels (Day 4)
│   ├── Matrix Multiply-Add with Shared Memory Optimization     # CUDA Learning Notes: Kernels (Day 5)
│   └── 3D Memory Allocation with cudaMalloc3D                  # CUDA Learning Notes: Kernels (Day 6)
│
├── python_numba/            # Python with Numba
│   ├── Mask Map Generator   # Mask Map Generator for 3D Brain Viewer Project
│   ├──                      # 
│   ├──                      # 
│   └──                      # 
```

## Goals
Practice low-level CUDA C/C++ kernel programming

Explore GPU acceleration in Python via Numba

Compare CPU vs GPU performance on various tasks

Serve as a foundation for future deep learning, graphics, or simulation projects

## Requirements
C/C++ CUDA
NVIDIA GPU with CUDA Compute Capability

CUDA Toolkit (tested on CUDA 12.x)

CMake / gcc / nvcc

Python Numba
Python 3.8+

numba, numpy, matplotlib (optional for visualization)

Install dependencies:

pip install -r requirements.txt
## Coming Up
More optimization patterns (shared memory, streams, pinned memory)

Real-time image processing examples

Integration with OpenCV and PyTorch (planned)

## Contact
If you're working on similar topics or want to collaborate, feel free to reach out via GitHub Issues.

