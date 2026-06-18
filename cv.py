from datetime import datetime
import glob
import os
import argparse
import sys

import cv2
import numpy as np
import math

# ==== Configuration Constants ====
BILATERAL_FILTER_CONFIG = {
    "d": 9,
    "sigmaColor": 75,
    "sigmaSpace": 75,
}

MORPH_KERNEL_SIZE = (5, 5)
MORPH_KERNEL_ITERATIONS = 1

# ==== Helper Functions ====
def get_centroid(cnt):
    """Extract centroid from contour with fallback to bounding box center.
    
    Args:
        cnt: OpenCV contour
        
    Returns:
        Tuple (cx, cy) representing contour centroid
    """
    M = cv2.moments(cnt)
    if M.get("m00", 0) != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return cx, cy
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        return x + w / 2.0, y + h / 2.0


def create_comparison_visualization(original_color, highlighted, mask, max_score, mask_background):
    """Create a 3-panel comparison visualization.
    
    Args:
        original_color: Original image in BGR format
        highlighted: Highlighted overlay image
        mask: Binary mask of detected region
        max_score: Area score to display
        mask_background: Background image for the mask panel
    Returns:
        Combined visualization as numpy array
    """
    mask_bool = mask == 255
    
    # Create colored mask visualization (green area)
    mask_color = mask_background.copy()
    mask_color[mask_bool] = (0, 255, 0)
    
    # Add area text annotation
    cv2.putText(
        mask_color,
        f"Area: {max_score:.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    
    return np.hstack((original_color, highlighted, mask_color))

def segment_baker_cyst(
    image_path,
    roi_top=0.15,
    roi_bottom=0.80,
    roi_left=0.05,
    roi_right=0.95,
    area_min=400,
    area_max=40000,
    aspect_ratio_min=0.4,
    aspect_ratio_max=5.0,
    solidity_min=0.65,
    extent_min=0.40,
    circularity_min=0.20,
    bottom_exclusion_margin=3,
    center_weight=0.5,
    upper_half_bonus=0.15,
    detect_offset=-28,
    refine_offset=-5,
    debug=False,
):
    """Two-stage Baker's cyst segmentation.

    Stage 1 (detect): Use a strict threshold (Otsu + detect_offset) to find
    candidate cyst contours with high precision.

    Stage 2 (refine): Around the detected contour, apply a permissive threshold
    (Otsu + refine_offset) to capture the full cyst extent.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    if debug:
        cv2.imwrite("logs/0original.png", img)

    final_mask = np.zeros_like(img)

    h, w = img.shape
    roi_top = int(h * roi_top)
    roi_bottom = int(h * roi_bottom)
    roi_left = int(w * roi_left)
    roi_right = int(w * roi_right)

    roi_mask = np.zeros_like(img)
    cv2.rectangle(roi_mask, (roi_left, roi_top), (roi_right, roi_bottom), 255, -1)

    if debug:
        img_roi = cv2.bitwise_and(img, img, mask=roi_mask)
        cv2.imwrite("logs/1roi_applied.png", img_roi)

    # Denoising
    blurred = cv2.bilateralFilter(img, **BILATERAL_FILTER_CONFIG)
    if debug:
        cv2.imwrite("logs/2blurred.png", blurred)

    # === Stage 1: Detect — strict threshold to locate cyst candidates ===
    t_otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    t_detect = max(t_otsu + detect_offset, 1)
    _, thresh_detect = cv2.threshold(blurred, t_detect, 255, cv2.THRESH_BINARY_INV)
    thresh_detect = cv2.bitwise_and(thresh_detect, thresh_detect, mask=roi_mask)

    if debug:
        cv2.imwrite("logs/3thresholded.png", thresh_detect)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    morph = cv2.morphologyEx(thresh_detect, cv2.MORPH_OPEN, kernel, iterations=MORPH_KERNEL_ITERATIONS)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=MORPH_KERNEL_ITERATIONS)

    if debug:
        cv2.imwrite("logs/4morphological.png", morph)

    # Find and score contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = -1.0
    max_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < area_min or area > area_max:
            if debug:
                print(f"Skipping contour with area {area} (not in range {area_min}-{area_max})")
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
        if h_box <= 0:
            continue

        contour_bottom = y_box + h_box
        if contour_bottom >= (roi_bottom - bottom_exclusion_margin):
            if debug:
                print("Skipping contour overlapping ROI bottom boundary")
            continue

        aspect_ratio = float(w_box) / h_box

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        extent = area / float(w_box * h_box)
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        if (
            aspect_ratio_min < aspect_ratio < aspect_ratio_max
            and solidity >= solidity_min
            and extent >= extent_min
            and circularity >= circularity_min
        ):
            cx, cy = get_centroid(cnt)

            roi_cx = (roi_left + roi_right) / 2.0
            roi_cy = (roi_top + roi_bottom) / 2.0

            dist = math.hypot(cx - roi_cx, cy - roi_cy)
            max_dist = math.hypot(
                max(roi_cx - roi_left, roi_right - roi_cx),
                max(roi_cy - roi_top, roi_bottom - roi_cy),
            )
            if max_dist <= 0:
                max_dist = math.hypot(w / 2.0, h / 2.0)

            distance_score = 1.0 - (dist / (max_dist + 1e-9))
            distance_score = max(0.0, min(1.0, distance_score))

            area_score = (area - area_min) / float(max(1, area_max - area_min))
            area_score = max(0.0, min(1.0, area_score))

            half_bonus = upper_half_bonus if cy < (h / 2.0) else 0.0

            combined_score = (1.0 - center_weight) * area_score + center_weight * distance_score + half_bonus

            if debug:
                print(
                    f"Candidate: Area={area}, AR={aspect_ratio:.2f}, "
                    f"Solidity={solidity:.2f}, Extent={extent:.2f}, Circ={circularity:.2f}, "
                    f"Score={combined_score:.3f}"
                )

            if combined_score > best_score:
                best_score = combined_score
                max_score = area
                best_cnt = cnt

    # === Stage 2: Refine — expand detected region with permissive threshold ===
    cyst_found = best_cnt is not None
    if cyst_found:
        # Get bounding box of detected contour and expand it
        x_box, y_box, w_box, h_box = cv2.boundingRect(best_cnt)
        pad_x = int(w_box * 0.5)
        pad_y = int(h_box * 0.5)
        local_x1 = max(x_box - pad_x, 0)
        local_y1 = max(y_box - pad_y, 0)
        local_x2 = min(x_box + w_box + pad_x, w)
        local_y2 = min(y_box + h_box + pad_y, h)

        # Apply permissive threshold in the local region
        t_refine = max(t_otsu + refine_offset, 1)
        _, thresh_refine = cv2.threshold(blurred, t_refine, 255, cv2.THRESH_BINARY_INV)

        # Restrict to local bounding box
        local_mask = np.zeros_like(img)
        local_mask[local_y1:local_y2, local_x1:local_x2] = 255
        thresh_local = cv2.bitwise_and(thresh_refine, thresh_refine, mask=local_mask)
        thresh_local = cv2.bitwise_and(thresh_local, thresh_local, mask=roi_mask)

        # Light morphology to clean up noise in the local region
        morph_local = cv2.morphologyEx(thresh_local, cv2.MORPH_OPEN, kernel, iterations=1)
        morph_local = cv2.morphologyEx(morph_local, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours in the local refined region and keep those overlapping
        # the originally detected contour
        seed_mask = np.zeros_like(img)
        cv2.drawContours(seed_mask, [best_cnt], -1, 255, thickness=-1)
        if debug:
            cv2.imwrite("logs/5seed_mask.png", seed_mask)
            cv2.imwrite("logs/6thresh_local.png", thresh_local)
            cv2.imwrite("logs/7morph_local.png", morph_local)

        local_contours, _ = cv2.findContours(morph_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for lc in local_contours:
            lc_mask = np.zeros_like(img)
            cv2.drawContours(lc_mask, [lc], -1, 255, thickness=-1)
            overlap = cv2.bitwise_and(lc_mask, seed_mask)
            if overlap.any():
                cv2.drawContours(final_mask, [lc], -1, 255, thickness=-1)

        # If refinement produced nothing (edge case), fall back to the seed contour
        if not final_mask.any():
            cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=-1)

        if debug:
            cv2.imwrite("logs/8final_mask.png", final_mask)
            print(f"Baker's cyst found! Seed area={max_score}, refined area={final_mask.sum()//255}")
    else:
        if debug:
            print("No Baker's cyst found meeting criteria.")

    return final_mask, cyst_found, max_score, morph, img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect Baker's cyst in ultrasound images using classical CV"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Specific image filename to process (without path). If not provided, processes all images.",
    )
    parser.add_argument(
        "--debug", default=True, action="store_true", help="Enable debug mode with intermediate outputs and logs"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="data/processed/annotations/post_trans-baker_cyst/batch_000",
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output-dir", default="logs/output", help="Output directory for results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed processing information"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine image paths to process
    if args.image:
        # Process single specified image
        image_paths = [os.path.join(args.input_dir, f"{args.image}.png")]
        output_dir = "logs"  # Use logs for single image debug
    else:
        # Process all images in directory
        if os.path.isfile(args.input_dir):
            with open(args.input_dir, "r") as f:
                image_paths = [line.strip() for line in f if line.strip()]
        else:
            image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
        output_dir = args.output_dir
    
    # Process images
    cysts_found = 0
    for img_path in image_paths:
        if args.verbose:
            print(f"Processing {img_path}...")
        
        # Run segmentation
        mask, cyst_found, max_score, morph_img, original = segment_baker_cyst(
            img_path, debug=args.debug
        )
        
        if cyst_found:
            cysts_found += 1
        
        # Create visualization
        if original is None:
            print(f"  ❌ Failed to read image: {img_path}")
            continue
        
        # Ensure mask matches original dimensions
        if mask.shape != original.shape:
            mask = cv2.resize(
                mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        
        # Convert grayscale to BGR for color visualization
        original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Create highlighted overlay
        overlay = original_color.copy()
        overlay[mask == 255] = (0, 0, 255)  # BGR red
        highlighted = cv2.addWeighted(original_color, 0.7, overlay, 0.3, 0)
        morph_img_color = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)

        # Create comparison visualization
        combined = create_comparison_visualization(
            original_color, highlighted, mask, max_score, morph_img_color
        )
        
        # Save result
        if cyst_found:
            
            basename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, os.path.basename(args.input_dir), basename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, combined)
        if args.verbose:
            status = "✓ Found" if cyst_found else "✗ Not found"
            print(f"  {status} | Saved to: {output_path}")
    
    # Summary
    total = len(image_paths)
    print(f"\n{'='*50}")
    print(f"Cysts found: {cysts_found}/{total}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")
