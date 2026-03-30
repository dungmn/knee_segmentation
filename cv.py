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

BILATERAL_FILTER_CONFIG = {
    "d": 11,
    "sigmaColor": 100,
    "sigmaSpace": 100,
}

MORPH_KERNEL_SIZE = (7, 7)
MORPH_KERNEL_ITERATIONS = 2

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


def create_comparison_visualization(original_color, highlighted, mask, max_score):
    """Create a 3-panel comparison visualization.
    
    Args:
        original_color: Original image in BGR format
        highlighted: Highlighted overlay image
        mask: Binary mask of detected region
        max_score: Area score to display
        
    Returns:
        Combined visualization as numpy array
    """
    mask_bool = mask == 255
    
    # Create colored mask visualization (green area)
    mask_color = original_color.copy()
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
    roi_top=0.2,
    roi_bottom=0.7,
    roi_left=0.1,
    roi_right=0.9,
    threshold_value=20,
    area_min=500,
    area_max=25000,
    aspect_ratio_min=1.2,
    aspect_ratio_max=4.0,
    solidity_min=0.75,
    extent_min=0.45,
    circularity_min=0.2,
    bottom_exclusion_margin=3,
    center_weight=0.5,  # 0.0 -> prefer largest area only, 1.0 -> prefer most centered only
    upper_half_bonus=0.25,  # additional score bonus if contour is in upper half of image
    debug=False,
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    if debug:
        cv2.imwrite("logs/0original.png", img)  # Save original image for verification

    final_mask = np.zeros_like(img)

    # 2. Crop/Define Region of Interest (ROI)
    # Ultrasound images usually have black borders and text at edges, we only care about the central part
    h, w = img.shape
    roi_top = int(h * roi_top)
    roi_bottom = int(h * roi_bottom)  # Only take the band containing ultrasound image
    roi_left = int(w * roi_left)
    roi_right = int(w * roi_right)
    
    # Create rectangular mask to exclude text and UI at edges
    roi_mask = np.zeros_like(img)
    cv2.rectangle(roi_mask, (roi_left, roi_top), (roi_right, roi_bottom), 255, -1)
    
    # Apply ROI to original image
    img_roi = cv2.bitwise_and(img, img, mask=roi_mask)

    if debug:
        cv2.imwrite("logs/1roi_applied.png", img_roi)  # Save image after ROI for verification

    # 3. Denoising
    # Use Bilateral Filter to blur noise while preserving sharp edges of fluid collections
    blurred = cv2.bilateralFilter(img, **BILATERAL_FILTER_CONFIG)
    # blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    if debug:
        cv2.imwrite("logs/2blurred.png", blurred)




    # 4. Thresholding
    # Baker's cyst appears extremely dark (pixel values near 0).
    # Set threshold value: pixels darker than threshold become white (255), brighter become black (0)
    # _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    t, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    t = t - 30 
    _, thresh = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY_INV)

    # Remove black background outside ultrasound region mistakenly identified as cyst due to THRESH_BINARY_INV
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    if debug:
        cv2.imwrite("logs/3thresholded.png", thresh)



    # 5. Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    
    # Opening: Remove small white noise (blood vessels, shadow artifacts)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=MORPH_KERNEL_ITERATIONS)
    # Closing: Fill black holes inside white cyst regions (if any cloudiness)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=MORPH_KERNEL_ITERATIONS)

    if debug:
        cv2.imwrite("logs/4morphological.png", morph)

    # 6. Find and Filter Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    max_score = -1
    best_score = -1.0  # combined score used for selection when center priority is enabled

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Skip regions too small or too large (may be bone shadow)
        if area < area_min or area > area_max:
            if debug:
                print(f"Skipping contour with area {area} (not in range {area_min}-{area_max})")
            continue
            
        # Find Bounding Box to check ratio and position
        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
        if h_box <= 0:
            if debug:
                print("Skipping contour with invalid bounding box height")
            continue

        # Remove contours touching/overlapping the bottom ROI boundary
        contour_bottom = y_box + h_box
        if contour_bottom >= (roi_bottom - bottom_exclusion_margin):
            if debug:
                print("Skipping contour overlapping ROI bottom boundary")
            continue

        aspect_ratio = float(w_box) / h_box

        # Shape compactness checks to reject irregular shapes (e.g., L-shape)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            if debug:
                print("Skipping contour with invalid perimeter")
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            if debug:
                print("Skipping contour with invalid convex hull area")
            continue

        solidity = area / hull_area
        extent = area / float(w_box * h_box)
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        
        # Baker's cyst in Post-Trans view usually has flattened oval shape
        # Width/height ratio typically > 1.0 (avoid confusion with round artery)
        if (
            aspect_ratio_min < aspect_ratio < aspect_ratio_max
            and solidity >= solidity_min
            and extent >= extent_min
            and circularity >= circularity_min
        ):
            # Compute centering score: prefer contours nearer to image center
            cx, cy = get_centroid(cnt)

            # Use ROI center as reference for centering preference (not full image center)
            roi_cx = (roi_left + roi_right) / 2.0
            roi_cy = (roi_top + roi_bottom) / 2.0

            dist = math.hypot(cx - roi_cx, cy - roi_cy)
            # max possible distance inside ROI (to furthest corner)
            max_dist = math.hypot(max(roi_cx - roi_left, roi_right - roi_cx), max(roi_cy - roi_top, roi_bottom - roi_cy))
            # fallback to image center radius if ROI degenerate
            if max_dist <= 0:
                img_cx = w / 2.0
                img_cy = h / 2.0
                max_dist = math.hypot(img_cx, img_cy)

            # Normalize distance into a score [0,1] where 1 means perfectly centered in ROI
            distance_score = 1.0 - (dist / (max_dist + 1e-9))
            distance_score = max(0.0, min(1.0, distance_score))

            if debug:
                print(
                    f"Contour metrics: Area={area}, AR={aspect_ratio:.2f}, "
                    f"Solidity={solidity:.2f}, Extent={extent:.2f}, Circ={circularity:.2f}"
                )
                print(
                    f"Distance score: {distance_score:.3f} (dist={dist:.1f}, max_dist={max_dist:.1f}), "
                    f"Centroid: ({cx:.1f}, {cy:.1f})"
                )

            # Normalize area into [0,1] based on area_min/area_max bounds
            area_score = (area - area_min) / float(max(1, area_max - area_min))
            area_score = max(0.0, min(1.0, area_score))

            # Prefer contours in the upper half of the image: add small bonus
            half_bonus = upper_half_bonus if cy < (h / 2.0) else 0.0
            if debug and half_bonus > 0:
                print(f"Upper-half bonus applied: {half_bonus:.3f} (cy={cy:.1f}, h/2={h/2.0:.1f})")

            # Combined score mixes area and centering preference, plus optional upper-half bonus
            combined_score = (1.0 - center_weight) * area_score + center_weight * distance_score + half_bonus

            # Select contour by combined score; keep max_score as area for downstream display
            if combined_score > best_score:
                # New best contour found by score
                best_score = combined_score
                max_score = area
                best_cnt = cnt
            elif best_cnt is not None and combined_score == best_score:
                # Tiebreaker: prefer upper-half contour when scores are equal
                _, best_cy = get_centroid(best_cnt)
                if cy < (h / 2.0) and best_cy >= (h / 2.0):
                    best_cnt = cnt

    # 7. Draw Output Mask
    cyst_found = best_cnt is not None
    if cyst_found:
        # Draw selected contour as solid white region (thickness=-1) on black background
        cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=-1)
        if debug:
            print("Baker's cyst found!")
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
        
        # Create comparison visualization
        combined = create_comparison_visualization(
            original_color, highlighted, mask, max_score
        )
        
        # Save result
        basename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, basename)
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
