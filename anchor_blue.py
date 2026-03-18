import cv2
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- FIXED ARRAY SIZE ---
ARRAY_W, ARRAY_H = 220, 110

def get_majority_bg_value(region_img):
    """Finds the absolute most common pixel value (the mode), ignoring extremes."""
    # Calculate a 256-bin histogram for the region
    hist = cv2.calcHist([region_img], [0], None, [256], [0, 256])
    
    # Zero out pure black (camera edges/shadows) and pure white (sealing lines/holes)
    # so they cannot possibly be selected as the majority background color.
    hist[0:15] = 0
    hist[240:256] = 0
    
    # The index with the highest count is our dominant background color
    majority_val = int(np.argmax(hist))
    return majority_val

def find_hole_array_in_quadrant(quad_img, offset_x, offset_y):
    """Searches using the majority value from a strictly defined inner R-region."""
    qh, qw = quad_img.shape
    
    # 1. Define the R-Region (Matching your orange boxes)
    # We crop 20% from the sides to avoid the camera curve and the center cross
    rx, ry = int(qw * 0.20), int(qh * 0.15)
    rw, rh = int(qw * 0.65), int(qh * 0.75)
    
    # Extract just this safe region
    r_region = quad_img[ry:ry+rh, rx:rx+rw]
    
    # Save global coordinates so we can draw the orange box later
    r_box_global = (offset_x + rx, offset_y + ry, rw, rh)
    
    # 2. Get the True Majority Background Color
    majority_val = get_majority_bg_value(r_region)
    
    # 3. Dynamic Thresholding based on the majority value
    if majority_val < 60:
        offset = 25  # Dark panels need a smaller jump to catch the faint holes
    elif majority_val < 100:
        offset = 40
    elif majority_val < 150:
        offset = 55
    else:
        offset = 70  # Light panels need a bigger jump to avoid thresholding the background itself
        
    dynamic_thresh = min(majority_val + offset, 240) 
    
    # Apply threshold to the whole quadrant
    _, binary = cv2.threshold(quad_img, dynamic_thresh, 255, cv2.THRESH_BINARY)
    
    # 4. Extract and Filter Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 500: # Filter out massive lines and tiny dust
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            
    # 5. Fuse the holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    fused_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    
    # 6. Find the Array Blob
    fused_contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cand = None
    max_area = 0
    
    for cnt in fused_contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 60000: 
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            if 1.0 < aspect_ratio < 3.5:
                if area > max_area:
                    max_area = area
                    
                    # Force the fixed 220x110 size around the center of the blob
                    cx = x + (w // 2)
                    cy = y + (h // 2)
                    fixed_x = offset_x + cx - (ARRAY_W // 2)
                    fixed_y = offset_y + cy - (ARRAY_H // 2)
                    
                    best_cand = (fixed_x, fixed_y, ARRAY_W, ARRAY_H)
                    
    return best_cand, r_box_global, majority_val

def process_and_plot_blue_only(input_dir, test_images, output_dir):
    num_images = len(test_images)
    if num_images == 0:
        print(f"No valid images found in {input_dir}")
        return

    # Dynamically size the plot grid based on the number of images
    cols = 2
    rows = math.ceil(num_images / cols)
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 6 * rows))
    
    # Flatten axes array for easy iteration, handling cases where there's only 1 row or 1 image
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, filename in enumerate(test_images):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None: 
            print(f"Failed to read {filename}. Skipping.")
            continue
            
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        H, W = img.shape
        
        quadrants = [
            (img[0:H//2, 0:W//2], 0, 0),             
            (img[H//2:H, 0:W//2], 0, H//2),          
            (img[0:H//2, W//2:W], W//2, 0),          
            (img[H//2:H, W//2:W], W//2, H//2)        
        ]
        
        for quad_img, offset_x, offset_y in quadrants:
            array_box, r_box, bg_val = find_hole_array_in_quadrant(quad_img, offset_x, offset_y)
            
            # Draw the R-Region Box (Orange)
            rx, ry, rw, rh = r_box
            cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 128, 255), 2)
            
            # Print the calculated Majority Background Value in the corner of the R-box
            cv2.putText(display_img, f"BG: {bg_val}", (rx + 10, ry + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
            
            # Draw the Detected Array Box (Blue)
            if array_box is not None:
                x, y, w, h = array_box
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            else:
                print(f"No array found in a quadrant of {filename} (BG Val: {bg_val}) - Potential DT1")

        axes[idx].imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(filename, fontsize=16)
        axes[idx].axis('off')

    # Turn off axes for any empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "blue_only_plot_output.png")
    
    plt.savefig(output_path, dpi=150)
    print(f"Saved multi-plot to {output_path}")

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect hole arrays in panel images.")
    parser.add_argument("--input_dir", type=str, default="./data/label/", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./outputs/figures/", help="Directory to save the plotted outputs")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process (use 0 for all images)")
    
    args = parser.parse_args()

    # Create input directory if testing out of the box
    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        print("Please create it and add test images, or specify a valid path with --input_dir.")
        exit(1)

    # Filter for standard image formats
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_ext)]
    all_files.sort() # Ensure consistent order

    # Handle number of images to process
    if args.num_images > 0:
        test_images = all_files[:args.num_images]
    else:
        test_images = all_files

    process_and_plot_blue_only(args.input_dir, test_images, args.output_dir)
