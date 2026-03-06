import glob
import os

import cv2
import numpy as np

def segment_baker_cyst(
    image_path,
    roi_top=0.2,
    roi_bottom=0.7,
    roi_left=0.1,
    roi_right=0.9,
    threshold_value=25,
    area_min=500,
    area_max=25000,
    aspect_ratio_min=1.2,
    aspect_ratio_max=4.0,
    solidity_min=0.8,
    extent_min=0.45,
    circularity_min=0.2,
    bottom_exclusion_margin=3,
    debug=False,
):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

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
    blurred = cv2.bilateralFilter(img_roi, d=9, sigmaColor=75, sigmaSpace=75)

    if debug:
        cv2.imwrite("logs/2blurred.png", blurred)

    # 4. Thresholding
    # Baker's cyst appears extremely dark (pixel values near 0).
    # Set threshold value: pixels darker than threshold become white (255), brighter become black (0)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Remove black background outside ultrasound region mistakenly identified as cyst due to THRESH_BINARY_INV
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    if debug:
        cv2.imwrite("logs/3thresholded.png", thresh)

    # 5. Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Opening: Remove small white noise (blood vessels, shadow artifacts)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing: Fill black holes inside white cyst regions (if any cloudiness)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        cv2.imwrite("logs/4morphological.png", morph)

    # 6. Find and Filter Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    max_score = -1

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

        if debug:
            print(
                f"Contour area: {area}, Aspect Ratio: {aspect_ratio:.2f}, "
                f"Solidity: {solidity:.2f}, Extent: {extent:.2f}, Circularity: {circularity:.2f}"
            )
        
        # Baker's cyst in Post-Trans view usually has flattened oval shape
        # Width/height ratio typically > 1.0 (avoid confusion with round artery)
        if (
            aspect_ratio_min < aspect_ratio < aspect_ratio_max
            and solidity >= solidity_min
            and extent >= extent_min
            and circularity >= circularity_min
        ):
            # Select contour with largest area satisfying conditions as Baker's cyst
            if area > max_score:
                max_score = area
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

    return final_mask, cyst_found

if __name__ == "__main__":
    image_dir = "data/processed/training/post_trans-baker_cyst-flipped-batch_000/images"
    output_dir = "logs/output"
    debug_mode = False  # Set to True for intermediate image outputs
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    
    # Uncomment the line below to process only a single test image
    output_dir = "logs"
    debug_mode = True
    image_paths = ["data/processed/training/post_trans-baker_cyst-flipped-batch_000/images/72bb1eb6-f020-11ed-b527-0a580a5f736a_17.png"]
    
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        print(f"Processing {img_path}...")
        mask, cyst_found = segment_baker_cyst(img_path, debug=debug_mode)
    
        # visualize results, left is the original image, right is the segmentation mask
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        combined = np.hstack((original, mask))
        cv2.imwrite(os.path.join(output_dir, f"{basename}"), combined)
        print(f"  Cyst found: {cyst_found}")
