import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label

def segment_skin(img):
    # Convert to multiple color spaces for more robust detection
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # HSV ranges
    lower_hsv = np.array([0, 20, 50], dtype=np.uint8)
    upper_hsv = np.array([30, 255, 255], dtype=np.uint8)
    
    # YCrCb ranges
    lower_ycrcb = np.array([0, 110, 70], dtype=np.uint8)
    upper_ycrcb = np.array([255, 158, 120], dtype=np.uint8)
    
    # Create masks
    mask_hsv = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb_img, lower_ycrcb, upper_ycrcb)
    
    # Combine masks
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    return mask

def remove_small_components(mask, min_size=3000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            output_mask[labels == i] = 255
    
    return output_mask

def apply_morphology(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply closing to fill small holes
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply opening to remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened

def filter_round_regions(mask, roundness_threshold=0.2):
    label_img = label(mask)
    regions = regionprops(label_img)
    filtered_regions = []
    region_scores = []
    
    total_image_area = mask.shape[0] * mask.shape[1]
    min_face_area = total_image_area * 0.01  # 1% of image
    max_face_area = total_image_area * 0.4   # 40% of image
    
    for idx, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        aspect_ratio = width / height if height != 0 else 0
        
        if region.perimeter == 0:
            continue
            
        roundness = 4 * np.pi * (region.area) / (region.perimeter ** 2)
        
        normalized_area = region.area / total_image_area
        score = normalized_area * (1 + roundness)
        
        print(f"\nRegion {idx+1}:")
        print(f"Area: {region.area} pixels ({(region.area/total_image_area)*100:.2f}% of image)")
        print(f"Roundness: {roundness:.2f}")
        print(f"Aspect ratio: {aspect_ratio:.2f}")
        print(f"Score: {score:.4f}")
        
        # Basic conditions for valid region
        if (region.area >= min_face_area and 
            region.area <= max_face_area and 
            0.4 < aspect_ratio < 2.5):
            filtered_regions.append(region)
            region_scores.append(score)
            print("-> Accepted")
        else:
            print("-> Rejected")
    
    # If we have multiple valid regions, choose the one with highest score
    if filtered_regions:
        best_region_idx = np.argmax(region_scores)
        print(f"\nRegion with best score: Region {best_region_idx+1}")
        print(f"Score: {region_scores[best_region_idx]:.4f}")
        return [filtered_regions[best_region_idx]]
    
    return []

def draw_ellipses(img, regions):
    output_img = img.copy()
    for idx, region in enumerate(regions):
        coords = region.coords
        if len(coords) < 5:
            continue
            
        # Calculate center using region moments
        M = cv2.moments(region.image.astype(np.uint8))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
            
        # Adjust coordinates for actual position in image
        cX += region.bbox[1]
        cY += region.bbox[0]
        
        # Calculate ellipse dimensions
        height = region.bbox[2] - region.bbox[0]
        width = region.bbox[3] - region.bbox[1]
        
        # Adjust vertical position
        cY = cY - height//10

        # Draw ellipse
        angle = 0
        cv2.ellipse(output_img,
                   (cX, cY),
                   (width//2, height//2),
                   angle,
                   0,
                   360,
                   (0, 255, 0),
                   2)
    
    return output_img

def process_and_save_results(image_paths, output_dir='results'):
    # Create results directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")

    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {idx + 1}/{len(image_paths)}: {image_path}")
        img = cv2.imread(image_path)
        
        # Detect skin
        skin_mask = segment_skin(img)
        
        # Remove small components
        clean_mask = remove_small_components(skin_mask)
        
        # Apply morphological operations
        morphed_mask = apply_morphology(clean_mask)
        
        # Filter regions
        regions = filter_round_regions(morphed_mask)
        
        if not regions:
            print("No potential face regions detected.")
            continue
        
        # Draw results
        result_img = draw_ellipses(img, regions)
        
        # Create figure with 4 images
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Skin mask
        plt.subplot(2, 2, 2)
        plt.imshow(skin_mask, cmap='gray')
        plt.title('Skin Mask')
        plt.axis('off')
        
        # Processed mask
        plt.subplot(2, 2, 3)
        plt.imshow(morphed_mask, cmap='gray')
        plt.title('Processed Mask')
        plt.axis('off')
        
        # Final result
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Final Result')
        plt.axis('off')
        
        plt.tight_layout()
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'result_{filename}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results saved to: {output_path}")

def main():
    input_dir = 'input_images'
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.endswith(valid_extensions)]
    
    if not image_paths:
        print(f"No images found in directory {input_dir}")
        print("Please add images with extensions: .jpg, .jpeg, or .png")
        return
    
    process_and_save_results(image_paths, output_dir='results')

if __name__ == "__main__":
    main()