# 🖼️ Image Masking with Python Numba + OpenCV

This is a simple example of using **Python + Numba** to accelerate image processing.

We load an image with **OpenCV**, then apply a custom **mask_map function** on the pixel array using **@cuda.jit**, and finally visualize the mask with **Matplotlib**.

---

## 📌 What This Example Demonstrates

- Reading and pre-processing images with `OpenCV`
- Accelerating pixel-wise image operations using `Numba`
- Generating a binary **mask map** based on pixel value conditions (e.g., brightness or color)
- Visualizing results using `Matplotlib`

---

## 🗂️ File Structure
```bash
Mask Map Generator (Simple)/
├── main.py # Main script for loading, masking, and visualization
├── brain.jpg # Sample input image
└── README.md # This file
```

---

## 🔧 Requirements

Install dependencies with pip:
```bash
pip install numpy opencv-python matplotlib numba
```

▶️ How to Run
```bash
python main.py
```
This will:

Load brain.jpg

Apply a Numba-accelerated function to compute a mask

Display both original image and mask map side-by-side

📷 Example Output
Original image vs. binary mask (e.g., pixels brighter than threshold):

[ Original Image ]    [ Mask Map (white = masked) ]
Masking logic can be customized based on brightness, color channels, or other features.

🧪 Why Use Numba?
Numba significantly speeds up the processing of large images by compiling your Python functions to optimized machine code (via LLVM). This is especially helpful for:

Pixel-by-pixel operations

Nested loops

GPU offloading via @cuda.jit (optional extension)


