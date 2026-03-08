import glob
import os

import cv2
import numpy as np
import math

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
    cv2.imwrite("logs/0original.png", img)  # Save original image for verification
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
    blurred = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # blurred = img
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

    # # write the demo image
    # combined = np.hstack((thresh, morph))
    # cv2.imwrite(os.path.join(output_dir, "viz_morph.png"), combined)

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
            # Use contour centroid if possible
            M = cv2.moments(cnt)
            if M.get("m00", 0) != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                # fallback to bounding box center
                cx = x_box + w_box / 2.0
                cy = y_box + h_box / 2.0

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
                f"Contour area: {area}, Aspect Ratio: {aspect_ratio:.2f}, "
                f"Solidity: {solidity:.2f}, Extent: {extent:.2f}, Circularity: {circularity:.2f}"
                ,f"Distance score (ROI center): {distance_score:.3f} (dist={dist:.1f}, max_dist={max_dist:.1f}, cx={cx:.1f}, cy={cy:.1f})")

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
                # If there is an existing best contour, check deterministic upper-half preference
                should_select = True
                if best_cnt is not None:
                    # compute best_cnt centroid
                    M_best = cv2.moments(best_cnt)
                    if M_best.get("m00", 0) != 0:
                        best_cy = M_best["m01"] / M_best["m00"]
                    else:
                        bx, by, bw, bh = cv2.boundingRect(best_cnt)
                        best_cy = by + bh / 2.0

                    # If current contour is in upper half and best is in lower half, prefer current regardless of score
                    if (cy < (h / 2.0)) and (best_cy >= (h / 2.0)):
                        should_select = True

                if should_select:
                    best_score = combined_score
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

    return final_mask, cyst_found, max_score, morph

if __name__ == "__main__":
    import sys
    name_arg = sys.argv[1] if len(sys.argv) > 1 else None
    image_dir = "data/processed/annotations/post_trans-baker_cyst/batch_000"
    output_dir = "logs/output"
    debug_mode = False  # Set to True for intermediate image outputs
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    
    # Uncomment the line below to process only a single test image
    if name_arg:
        output_dir = "logs"
        debug_mode = True
        image_paths = [f"data/processed/annotations/post_trans-baker_cyst/batch_000/{name_arg}.png"]

    cnt = 0
    for img_path in image_paths:
        basename = os.path.basename(img_path)
        print(f"Processing {img_path}...")
        mask, cyst_found, max_score, morph_img = segment_baker_cyst(img_path, debug=debug_mode)
        if cyst_found:
            cnt += 1
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
        # # visualize results, left is the original image, right is the segmentation mask
        # combined = np.hstack((original, mask))
        # cv2.imwrite(os.path.join(output_dir, f"{basename}"), combined)

        # Also create a color-highlighted overlay (do not replace the existing saved image)
        if original is None:
            print(f"  Failed to read original image {img_path}")
            print(f"  Cyst found: {cyst_found}")
            continue

        # Convert grayscale original to BGR for color visualization
        original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        # Ensure mask is single-channel and same shape
        mask_resized = mask
        if mask_resized.shape != original.shape:
            mask_resized = cv2.resize(mask_resized, (original.shape[1], original.shape[0]))


        mask_bool = mask_resized == 255

        # Create an overlay: red for cyst area
        overlay = original_color.copy()
        overlay[mask_bool] = (0, 0, 255)  # BGR red

        # Blend overlay with original to highlight cyst
        highlighted = cv2.addWeighted(original_color, 0.7, overlay, 0.3, 0)

   

        # Combine original, highlighted, and mask visualization horizontally
        # Ensure `morph_img` is 3-channel and same size as original for np.hstack
        morph_vis = morph_img
        if morph_vis is None:
            morph_vis = np.zeros_like(original_color)
        elif morph_vis.ndim == 2:
            morph_vis = cv2.cvtColor(morph_vis, cv2.COLOR_GRAY2BGR)

        if morph_vis.shape[:2] != original_color.shape[:2]:
            morph_vis = cv2.resize(morph_vis, (original_color.shape[1], original_color.shape[0]))


        # Create a colored mask visualization (green area) for standalone mask view
        mask_color = morph_vis.copy()
        mask_color[mask_bool] = (0, 255, 0)
        # text the max_score on the mask_color image
        cv2.putText(mask_color, f"Area: {max_score:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined_highlight = np.hstack((original_color, highlighted, mask_color))

        # Save the additional highlighted image
        out_path = os.path.join(output_dir, basename)
        cv2.imwrite(out_path, combined_highlight)

        print(f"  Saved highlighted output to: {out_path}")
        print(f"  Cyst found: {cyst_found}")

    print(f"Total cysts found: {cnt}/{len(image_paths)}")
