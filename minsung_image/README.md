# Optical Feature Detection Pipeline

This project contains an automated computer vision pipeline designed to robustly detect specific hardware features (bright arrays and dark frames) across varying and uneven lighting conditions. 

## Usage

Run the script by passing your dataset directory and desired output directory via command-line arguments:

```bash
python main.py --input /path/to/dataset --output /path/to/save/figures --samples 10
```

* `--input`: The root folder containing your images, organized by day (e.g., `/dataset/03/`, `/dataset/04/`).
* `--output`: The directory where the diagnostic multi-image grids will be saved.
* `--samples`: (Optional) Number of random images to sample per day directory (defaults to 10).

## Methodology

This algorithm uses a multi-stage, fault-tolerant approach to find features regardless of lighting variations or missing components.

### 1. Image Segmentation (4 Quadrants)
To handle severe lighting gradients, the image is split into 4 independent quadrants (top left, top right, bottom left, bottom right). By processing locally, a bright glare in one corner will not skew the detection logic in a dark corner.

### 2. Primary Target: Detecting Bright Array (0-255 pixel scale)
The algorithm first searches for the bright white array slots against the surrounding background.

* **2-1. Determine Background Value (`bg_val`):** Calculate the dominant pixel intensity of the specific region, ignoring pure black (pixel < 15) and pure white (pixel > 240).
* **2-2. Dynamic Thresholding:** Add a specific offset to `bg_val` to isolate only the bright white holes:
    * If `bg_val` < 60 (very dark): threshold is `bg_val` + 25
    * If `bg_val` < 100 (dark): threshold is `bg_val` + 40
    * If `bg_val` < 150 (medium): threshold is `bg_val` + 55
    * If `bg_val` >= 150 (light): threshold is `bg_val` + 70
* **2-3. Fusion & Filtering:** * Discard small noise (contour area < 500 pixels).
    * Apply a morphological block (25x25 kernel) to fuse the remaining bright holes into one solid shape.
    * Filter the shape to ensure it matches the physical array dimensions: area between 1,000 and 60,000 pixels, with a rectangular aspect ratio (1.0 to 3.5).

### 3. Fallback Method: Detecting Dark Frame Footprint 
If the primary array is missing or obscured, the algorithm falls back to finding the dark rectangular depression.

* **3-1. Local Contrast (Adaptive Thresholding):** Evaluate pixels against their local 151x151 neighborhood. Flag pixels that are significantly darker than their immediate surroundings. This bypasses global lighting issues.
* **3-2. Fusion & Filtering:** * Apply a morphological block (15x15 kernel) to smooth and solidify the dark shape.
    * Filter the shape to match the larger frame footprint: area between 8,000 and 80,000 pixels, aspect ratio of 1.5 to 3.5.

### 4. Outer Frame Detection (Template Matching)
Once the inner target (either the bright array or the dark footprint) is found, the algorithm locks onto the outer boundaries.

* **4-1. Targeted Search Area:** Use the center coordinate of the inner box (found in Step 2 or 3) as an anchor. Define a tight search boundary slightly larger than the expected outer frame.
* **4-2. Feature Extraction:** * If `bg_val` < 85 (dark background): Highlight edges using inverse thresholding at `bg_val` - 15.
    * If `bg_val` >= 85 (light background): Highlight edges using Canny Edge Detection.
* **4-3. Template Matching:** Slide a mathematically perfect 340x180 rectangular outline over the search area to find the exact final alignment.
