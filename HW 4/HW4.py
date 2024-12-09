import cv2
import numpy as np
import gc
import os
from pathlib import Path

def resize_image(img, max_dimension=1200):
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / float(max(height, width))
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def detect_and_match(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize the sift detector
    sift = cv2.SIFT_create(nfeatures=3000)
    
    # Detect the key points and descriptors in each grayscale image
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return None, None, None
    
    # Flann using the Kdtree algorithm to split the space in multiple regions for efficent an search
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # Flann matcher used to find matches between the SIFT descriptors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find the 2 closest neighbours for each point
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    # We keep only the high-quality matchings using Lowe's ratio test
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]
    
    del gray1, gray2, matches
    gc.collect()
    
    return kp1, kp2, good_matches

def stitch_two_images(img1, img2):
    kp1, kp2, matches = detect_and_match(img1, img2)
    
    if matches is None or len(matches) < 4:
        return img1
    
    # Convert key points into coordinates for both images
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute the homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        return img1
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Compute the corners of the image by saving their coordinates in an array
    corners1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
    
    # Apply the homography matrix onto the corners of the first image
    warped_corners1 = cv2.perspectiveTransform(corners1, H)
    # Combine the corners of the first image that has been warped with the corners of the second image
    corners = np.concatenate((warped_corners1, corners2), axis=0)
    
    # Compute the limits 
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    # Translation matrix, that regardless of the values of xmin and ymin, all the coordinates will be positive
    t = [-xmin, -ymin]
    # 3x3 matrix for a homographic transformation
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    # Apply the transformation
    output_size = (min(xmax-xmin, 12000), min(ymax-ymin, 12000))
    result = cv2.warpPerspective(img1, Ht.dot(H), output_size)
    
    # Check if the second image overlaps the first transformed image
    if -ymin < result.shape[0] and -xmin < result.shape[1]: 
        # Coordinates for the first image and use min and max to not get out of bounds
        y_start = max(0, -ymin)
        y_end = min(result.shape[0], -ymin + h2)
        x_start = max(0, -xmin)
        x_end = min(result.shape[1], -xmin + w2)
        
        # Coordinates for the second image
        img2_y_start = max(0, ymin)
        img2_y_end = min(h2, ymin + result.shape[0])
        img2_x_start = max(0, xmin)
        img2_x_end = min(w2, xmin + result.shape[1])
        
        # Gaussian mask
        mask = np.zeros((y_end - y_start, x_end - x_start), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (mask.shape[1], mask.shape[0]), 1, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Combining the 2 images, each rgb channel is treated separately
        for c in range(3):
            result[y_start:y_end, x_start:x_end, c] = \
                (1 - mask) * result[y_start:y_end, x_start:x_end, c] + \
                mask * img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end, c]
    
    return result

def trim_black_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find all non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find the non-black contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        border = 1
        x = max(0, x - border)
        y = max(0, y - border)
        w = min(img.shape[1] - x, w + 2*border)
        h = min(img.shape[0] - y, h + 2*border)
        
        # Crop the image
        return img[y:y+h, x:x+w]
    
    return img

def create_panorama_sequential():
    current_dir = Path.cwd()
    base_path = current_dir / 'HW 4'

    i = 1
    while (base_path / f'ss1_{i}.png').exists():
        i += 1
    total_images = i - 1
    
    # Start with the image in the middle
    middle = total_images // 2
    result = cv2.imread(str(base_path / f'ss1_{middle}.png'))
    result = resize_image(result)
    
    # Process to the right
    for i in range(middle + 1, total_images + 1):
        next_img = cv2.imread(str(base_path / f'ss1_{i}.png'))
        if next_img is not None:
            next_img = resize_image(next_img)
            result = stitch_two_images(result, next_img)
    
    # Process to the left
    for i in range(middle - 1, 0, -1):
        next_img = cv2.imread(str(base_path / f'ss1_{i}.png'))
        if next_img is not None:
            next_img = resize_image(next_img)
            result = stitch_two_images(next_img, result)
    
    # Resize the final result
    final_result = resize_image(result, max_dimension=3000)
    # Remove the black edges
    final_result = trim_black_edges(final_result)
    
    return final_result
        
def main():
    panorama1 = create_panorama_sequential()
    current_dir = Path.cwd()
    base_path = current_dir / 'HW 4'
    output_path1 = base_path / 'panorama1_result.png'
    cv2.imwrite(str(output_path1), panorama1)
    print("Succesfully created the panorama!")

if __name__ == "__main__":
    main()