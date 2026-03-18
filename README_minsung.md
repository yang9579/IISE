# Hole Array Detection (`anchor_blue.py`)

This repository contains a computer vision script designed to automatically detect and draw bounding boxes around fixed-size hole arrays (220x110 pixels) on panel images. It is optimized to handle varying lighting conditions and panel colors using dynamic thresholding.

## How It Works (The Logic)

The script processes images in a step-by-step pipeline to isolate the target arrays while ignoring edge artifacts, shadows, and dust. 

1. **Quadrant Splitting**: The input image is divided into four equal quadrants (top-left, top-right, bottom-left, bottom-right). Each quadrant is processed independently.
2. **"R-Region" Extraction**: Inside each quadrant, the script defines a safe "R-Region" (cropped 20% from the sides, 15% from the top) to avoid the physical curvature of the camera and the center cross of the panel. 
3. **Background Color Calculation**: It calculates a 256-bin histogram of the R-Region to find the "majority background value" (the mode). It explicitly ignores pure black (shadows/edges) and pure white (holes/sealing lines) to ensure an accurate reading of the panel's actual color.
4. **Dynamic Thresholding**: Based on the majority background value, the script calculates a custom threshold offset. Darker panels get a smaller offset to catch faint holes, while lighter panels get a larger offset to avoid thresholding the background itself.
5. **Noise Filtering & Fusing**: 
    * It finds contours in the thresholded binary image and filters out tiny specks (dust) and massive objects.
    * It applies a morphological closing operation (`cv2.morphologyEx`) to visually "fuse" the individual holes together into a single, solid blob.
6. **Fixed Bounding Box Selection**: It searches for the largest fused blob that matches expected aspect ratio and area constraints. Once the center of this blob is found, the script forces a fixed **220x110 pixel** blue bounding box around that center point.
7. **Visualization**: Finally, it plots the results on a grid, drawing the orange R-Region boxes and the blue Array boxes, and saves the final figure to an output directory.

---

## Requirements

Ensure you have Python installed, along with the required computer vision and plotting libraries. You can install the dependencies using `pip`:

```bash
pip install opencv-python numpy matplotlib
