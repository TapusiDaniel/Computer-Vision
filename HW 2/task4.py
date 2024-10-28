import cv2
import numpy as np
import os

def rgb_to_hsv(r, g, b):
    """
    Convert RGB color to HSV color space.
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    
    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    s = 0 if cmax == 0 else (diff / cmax) * 100
    v = cmax * 100
    
    return h, s, v

def is_in_range(h, s, v, lower, upper):
    """
    Check if a color (h, s, v) is within the specified range.
    """
    return (lower[0] <= h <= upper[0] and
            lower[1] <= s <= upper[1] and
            lower[2] <= v <= upper[2])

def color_filter(init_image, lower, upper):
    """
    Apply color filtering to an image based on the specified HSV range.
    """
    height, width, _ = init_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    filtered_image = np.zeros_like(init_image)

    for y in range(height):
        for x in range(width):
            r, g, b = init_image[y, x]
            h, s, v = rgb_to_hsv(r, g, b)
            
            if is_in_range(h, s, v, lower, upper):
                mask[y, x] = 255
                filtered_image[y, x] = init_image[y, x]

    return filtered_image, mask

def helipad_filter(init_image):
    """
    Apply color filtering to detect the helipad.
    """
    # Define helipad color in RGB format
    helipad_rgb = [212, 139, 143]  # RGB
    helipad_h, helipad_s, helipad_v = rgb_to_hsv(*helipad_rgb)

    # Define HSV range
    lower_helipad = [max(0, helipad_h - 20), 20, 20]
    upper_helipad = [min(360, helipad_h + 20), 100, 100]

    return color_filter(init_image, lower_helipad, upper_helipad)

def blue_filter(init_image):
    """
    Apply color filtering to detect the pool.
    """
    # Define pool color in RGB format
    pool_rgb = [19, 240, 250]  
    pool_h, pool_s, pool_v = rgb_to_hsv(*pool_rgb)

    # Define HSV range 
    lower_blue = [max(0, pool_h - 5), 50, 50]
    upper_blue = [min(360, pool_h + 5), 100, 100]

    return color_filter(init_image, lower_blue, upper_blue)

def orange_filter(init_image):
    """
    Apply color filtering to detect the orange roof.
    """
    # Define roof color in RGB format
    roof_rgb = [206, 124, 94]  
    roof_h, roof_s, roof_v = rgb_to_hsv(*roof_rgb)

    # Define HSV range 
    lower_orange = [max(0, roof_h - 20), 20, 20]
    upper_orange = [min(360, roof_h + 10), 100, 100]

    return color_filter(init_image, lower_orange, upper_orange)

def find_best_contour(mask, init_image, max_contours=10, color=(0, 255, 0), 
                      aspect_ratio_range=None, min_sides=None, max_sides=None, 
                      min_area=0, use_approx=False):
    """
    Find the best contour in the mask based on various criteria.
    """
    # Find all contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_candidate = None
    best_score = 0
    best_square = None

    # Iterate through the top contours
    for i, contour in enumerate(contours[:max_contours]):
        if use_approx:
            # Approximate the contour to reduce the number of points
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            # Check if the number of sides is within the specified range
            if min_sides and max_sides and not (min_sides <= len(approx) <= max_sides):
                continue
        else:
            approx = contour

        # Get the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Calculate aspect ratio and area
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if width and height else 0
        area = cv2.contourArea(approx)

        print(f"Contour {i+1}: Area = {area:.2f}, Aspect ratio = {aspect_ratio:.2f}")

        # Check if the aspect ratio is within the specified range
        if aspect_ratio_range and not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue

        # Check if the area is above the minimum threshold
        if area < min_area:
            continue

        # Create a mask for the current contour
        mask_contour = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(mask_contour, [box], 0, 255, -1)
        
        # Calculate the matching percentage between the contour and the original mask
        matching_pixel_count = cv2.countNonZero(cv2.bitwise_and(mask, mask_contour))
        total_pixel_count = cv2.countNonZero(mask_contour)
        matching_percentage = matching_pixel_count / total_pixel_count if total_pixel_count > 0 else 0

        # Calculate the score based on area and matching percentage
        score = area * matching_percentage

        print(f"Contour {i+1}: Matching percentage = {matching_percentage:.2f}, Score = {score:.2f}")

        # Update the best candidate if the current contour has a higher score
        if score > best_score:
            best_score = score
            best_candidate = box if not use_approx else approx

            # If using approximation, create a square based on the contour's area
            if use_approx:
                side_length = int(np.sqrt(area))
                center, _, _ = rect
                half_side = side_length // 2
                best_square = np.array([
                    [int(center[0] - half_side), int(center[1] - half_side)],
                    [int(center[0] + half_side), int(center[1] - half_side)],
                    [int(center[0] + half_side), int(center[1] + half_side)],
                    [int(center[0] - half_side), int(center[1] + half_side)]
                ], dtype=np.int32)

    # Create a copy of the initial image for drawing the best contour
    final_image = init_image.copy()

    # Draw the best contour found (if any) on the final image
    if best_candidate is not None:
        cv2.drawContours(final_image, [best_square if best_square is not None else best_candidate], 0, color, 3)
        print("Object detected successfully.")
    else:
        print("No object detected in the image.")

    # Return the final image with the drawn contour (or without, if none was found)
    return final_image

def contour_finding_pool(init_image, mask_pool):
    """
    Find the contour of the pool in the image.
    """
    return find_best_contour(mask_pool, init_image, color=(255, 0, 0), aspect_ratio_range=(1, 3), min_area=60)

def contour_finding_helipad(init_image, mask_helipad):
    """
    Find the contour of the helipad in the image.
    """
    return find_best_contour(mask_helipad, init_image, max_contours=20, color=(0, 255, 0),
                             aspect_ratio_range=(0.7, 1.5), min_sides=4, max_sides=8, use_approx=True)

def contour_finding_building(init_image, mask_roof):
    """
    Find the contour of the orange roof in the image.
    """
    return find_best_contour(mask_roof, init_image, color=(0, 0, 255),
                             aspect_ratio_range=(1.3, 2.0), min_area=200)

def process_image(image_path):
    """
    Process the image to detect pool, helipad, and building.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read the image from path: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect pool
    print("Applying pool filter...")
    filtered_pool, mask_pool = blue_filter(image_rgb)
    mask_pool = cv2.medianBlur(mask_pool, 9)
    image_with_pool = contour_finding_pool(image_rgb, mask_pool)

    # Detect helipad
    print("Applying helipad filter...")
    filtered_helipad, mask_helipad = helipad_filter(image_rgb)
    mask_helipad = cv2.medianBlur(mask_helipad, 9)
    image_with_pool_and_helipad = contour_finding_helipad(image_with_pool, mask_helipad)

    # Detect building
    print("Applying building filter...")
    filtered_building, mask_building = orange_filter(image_rgb)
    mask_building = cv2.medianBlur(mask_building, 9)
    final_image = contour_finding_building(image_with_pool_and_helipad, mask_building)

    # Save results
    output_dir = 'results/object_detection'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert final image to BGR 
    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/detected_objects.jpg', final_image_bgr)
    cv2.imwrite(f'{output_dir}/pool_mask.jpg', mask_pool)
    cv2.imwrite(f'{output_dir}/helipad_mask.jpg', mask_helipad)
    cv2.imwrite(f'{output_dir}/building_mask.jpg', mask_building)
    
    print(f"Results have been saved in {output_dir}/")

if __name__ == "__main__":
    image_path = "/home/danyez87/Master AI/CV/HW 2/Gura_Portitei_Scara_100.jpg"  
    process_image(image_path)