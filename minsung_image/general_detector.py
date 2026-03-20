"""
Optical Feature Detection Pipeline
Detects bright hole arrays and dark bounding frames in segmented quadrants using 
dynamic thresholding and template matching. Generates diagnostic heatmap grids.
"""

import cv2
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- FIXED SIZES ---
ARRAY_W, ARRAY_H = 220, 110
FRAME_W, FRAME_H = 340, 180

def get_majority_bg_value(region_img):
    """Calculates the dominant background pixel intensity."""
    hist = cv2.calcHist([region_img], [0], None, [256], [0, 256])
    hist[0:15] = 0
    hist[240:256] = 0
    return int(np.argmax(hist))

def find_dark_box_in_quadrant(quad_img, offset_x, offset_y):
    """Fallback: Detects the dark rectangular footprint using local contrast."""
    blurred = cv2.GaussianBlur(quad_img, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 
        151, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 8000 < area < 80000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)
            if 1.5 < aspect_ratio < 3.5 and area > max_area:
                max_area = area
                cx = x + (w // 2)
                cy = y + (h // 2)
                best_box = (
                    offset_x + cx - (ARRAY_W // 2), 
                    offset_y + cy - (ARRAY_H // 2), 
                    ARRAY_W, ARRAY_H
                )

    return best_box

def find_hole_array_in_quadrant(quad_img, offset_x, offset_y):
    """Primary: Detects the bright slots of the array using dynamic thresholding."""
    qh, qw = quad_img.shape
    rx, ry = int(qw * 0.20), int(qh * 0.15)
    rw, rh = int(qw * 0.65), int(qh * 0.75)
    r_region = quad_img[ry:ry+rh, rx:rx+rw]
    
    majority_val = get_majority_bg_value(r_region)
    
    # Dynamic thresholds based on lighting
    if majority_val < 60: offset = 25
    elif majority_val < 100: offset = 40
    elif majority_val < 150: offset = 55
    else: offset = 70
        
    dynamic_thresh = min(majority_val + offset, 240) 
    _, binary = cv2.threshold(quad_img, dynamic_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_mask = np.zeros_like(binary)
    for cnt in contours:
        if 5 < cv2.contourArea(cnt) < 500: 
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    fused_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    fused_contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    array_box = None
    max_area = 0
    for cnt in fused_contours:
        area = cv2.contourArea(cnt)
        if 1000 < area < 60000: 
            x, y, w, h = cv2.boundingRect(cnt)
            if 1.0 < (float(w) / max(h, 1)) < 3.5 and area > max_area:
                max_area = area
                cx, cy = x + (w // 2), y + (h // 2)
                array_box = (offset_x + cx - (ARRAY_W // 2), offset_y + cy - (ARRAY_H // 2), ARRAY_W, ARRAY_H)
                    
    return array_box, majority_val

def find_frame_around_array_split(quad_img, array_box, bg_val, offset_x, offset_y):
    """Finds the outer frame using template matching anchored to the inner box."""
    ax, ay, aw, ah = array_box
    local_ax = ax - offset_x
    local_ay = ay - offset_y
    
    cx = local_ax + (aw // 2)
    cy = local_ay + (ah // 2)
    
    search_w = FRAME_W + 60
    search_h = FRAME_H + 60
    
    sx1 = max(0, cx - search_w // 2)
    sy1 = max(0, cy - search_h // 2)
    sx2 = min(quad_img.shape[1], cx + search_w // 2)
    sy2 = min(quad_img.shape[0], cy + search_h // 2)
    
    search_crop = quad_img[sy1:sy2, sx1:sx2].copy()
    
    bright_mask = search_crop > min(bg_val + 35, 240)
    kernel = np.ones((5, 5), np.uint8)
    bright_mask_dilated = cv2.dilate(bright_mask.astype(np.uint8), kernel, iterations=1)
    search_crop[bright_mask_dilated == 1] = bg_val
    
    if bg_val < 85:
        dark_threshold = max(5, bg_val - 15) 
        _, match_target = cv2.threshold(search_crop, dark_threshold, 255, cv2.THRESH_BINARY_INV)
        match_target = cv2.dilate(match_target, np.ones((2,2), np.uint8), iterations=1)
        match_thresh = 0.05
    else:
        blurred = cv2.GaussianBlur(search_crop, (5, 5), 0)
        match_target = cv2.Canny(blurred, 15, 60)
        match_thresh = 0.04

    pad = 5
    tw, th = FRAME_W + pad*2, FRAME_H + pad*2
    template = np.zeros((th, tw), dtype=np.uint8)
    cv2.rectangle(template, (pad, pad), (pad + FRAME_W, pad + FRAME_H), 255, 3)
    
    if th > match_target.shape[0] or tw > match_target.shape[1]:
        return None
        
    res = cv2.matchTemplate(match_target, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < match_thresh: return None
        
    fx = offset_x + sx1 + max_loc[0] + pad
    fy = offset_y + sy1 + max_loc[1] + pad
    return (fx, fy, FRAME_W, FRAME_H)

def process_and_plot_day(day_dir, day_id, output_dir, num_samples=10):
    """Randomly samples images and generates a debug grid comparing Original vs Heatmap."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_files = [f for f in os.listdir(day_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_files:
        print(f"Skipping {day_id}: No images found.")
        return
        
    sampled_files = random.sample(all_files, min(len(all_files), num_samples))
    
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(32, 30))
    plot_idx = 0
    
    for filename in sampled_files:
        if plot_idx >= 10: break 
            
        img_path = os.path.join(day_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
            
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        heatmap_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        
        H, W = img.shape
        quadrants = [
            (img[0:H//2, 0:W//2], 0, 0),             
            (img[H//2:H, 0:W//2], 0, H//2),          
            (img[0:H//2, W//2:W], W//2, 0),          
            (img[H//2:H, W//2:W], W//2, H//2)        
        ]
        
        region_count = 0
        array_count = 0
        
        for quad_img, offset_x, offset_y in quadrants:
            array_box, bg_val = find_hole_array_in_quadrant(quad_img, offset_x, offset_y)
            
            if array_box is None:
                array_box = find_dark_box_in_quadrant(quad_img, offset_x, offset_y)
                bg_val = get_majority_bg_value(quad_img)
            
            if array_box:
                array_count += 1
                frame_box = find_frame_around_array_split(quad_img, array_box, bg_val, offset_x, offset_y)
                
                if frame_box:
                    region_count += 1
                    fx, fy, fw, fh = frame_box
                    cv2.rectangle(display_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 4)
                    cv2.rectangle(heatmap_img, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 4)

                ax, ay, aw, ah = array_box
                cv2.rectangle(display_img, (ax, ay), (ax + aw, ay + ah), (255, 0, 0), 4)
                cv2.rectangle(heatmap_img, (ax, ay), (ax + aw, ay + ah), (255, 255, 255), 4)

        row = plot_idx // 2
        col_orig = (plot_idx % 2) * 2
        col_heat = col_orig + 1

        title_str = f"{filename} | Region: {region_count}, Array: {array_count}"

        axes[row, col_orig].imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        axes[row, col_orig].set_title(f"{title_str} (Orig)", fontsize=14)
        axes[row, col_orig].axis('off')
        
        axes[row, col_heat].imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        axes[row, col_heat].set_title(f"{title_str} (Heat)", fontsize=14)
        axes[row, col_heat].axis('off')
        
        plot_idx += 1

    for i in range(plot_idx, 10):
        row = i // 2
        col_orig = (i % 2) * 2
        col_heat = col_orig + 1
        axes[row, col_orig].axis('off')
        axes[row, col_heat].axis('off')

    plt.tight_layout()
    output_name = os.path.join(output_dir, f"debug_results_{day_id}.png")
    plt.savefig(output_name, dpi=150)
    plt.close(fig) 
    print(f"Saved random sample grid for day {day_id} -> {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to detect frames and arrays.")
    parser.add_argument("--input", type=str, required=True, help="Base directory containing daily image folders.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save output visual plots.")
    parser.add_argument("--samples", type=int, default=10, help="Number of random samples to plot per day.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.input):
        folders = [f for f in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, f))]
        
        for day_folder in sorted(folders):
            print(f"\nProcessing folder: {day_folder}...")
            day_path = os.path.join(args.input, day_folder)
            process_and_plot_day(day_path, day_folder, args.output, num_samples=args.samples)
            
        print("\nAll daily samples processed!")
    else:
        print(f"Error: Directory {args.input} does not exist.")