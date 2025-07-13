import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import cuda

@cuda.jit()
def mask_map(img, mask):
    # Using cuda.grid() to get the global thread index, which is equal as below:
    # x_idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # y_idx = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    x_idx, y_idx = cuda.grid(2)
    if x_idx < img.shape[0] and y_idx < img.shape[1]:
        if img[x_idx][y_idx] > 0:
            mask[x_idx][y_idx] = 1
        else:
            mask[x_idx][y_idx] = 0 # must be set to 0, otherwise other threads are not stable to save binary mask.

print("\n\n---------Start to read image: ")
img = cv.imread(f"{os.getcwd()}/src/52.jpg")

if img is None:
    print("Image load failed")
    quit()
print("Successful!")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
results = np.zeros_like(img_gray)
print(f"Image Shape: {img_gray.shape} | Image Type: {type(img_gray)} | Image Data Type: {img_gray.dtype}")

print("\n\n---------Initialize GPU Resources:")
threadsperblock = (16, 16) # according to image width and height
blockspergrid_x = (img_gray.shape[0] + threadsperblock[0] - 1) // threadsperblock[0] # In case the image size is not an integer multiple of the block size, enough blocks are still allocated to process the entire image.
blockspergrid_y = (img_gray.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)
print(f"Allocate threads: {threadsperblock} | Allocate blocks: {blockspergrid}")
print("Copy data to device: ")
d_img = cuda.to_device(img_gray)
d_results = cuda.device_array_like(results)

print("\n\n---------Start to generate Mask Map")
mask_map[blockspergrid, threadsperblock](d_img, d_results)
print("Synchronize GPU")
cuda.synchronize()
print("Copy data to host: ")
d_results.copy_to_host(results)

plt.subplot(1, 2, 1)
plt.imshow(img_gray)
plt.axis('off')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(results)
plt.axis('off')
plt.title("Mask Map")

plt.show()
