import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy import ndimage

def gaussian_filter(size, sigma):
    """
    Create a Gaussian filter kernel.
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def convolution(image, kernel):
    """
    Perform convolution on an image using scipy's ndimage.
    """
    return ndimage.convolve(image, kernel)

def sobel_filters():
    """
    Define Sobel filters for edge detection.
    """
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def non_max_suppression(img, D):
    """
    Perform non-maximum suppression to thin out edges.
    """
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255
            
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0
    
    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    """
    Apply double thresholding to classify edges as strong, weak, or non-edges.
    """
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.float32)
    
    weak = 25
    strong = 255
    
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res

def hysteresis(img):
    """
    Perform edge tracking by hysteresis to finalize the edge detection.
    """
    M, N = img.shape
    weak = 25
    strong = 255

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    
    return img

def canny_edge_detection(image, sigma=1, low_threshold=0.05, high_threshold=0.15, kernel_type='sobel'):
    """
    Perform Canny edge detection on the input image.
    """
    # Normalize image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Gaussian Smoothing
    smoothed = convolution(image, gaussian_filter(size=5, sigma=sigma))
    
    # Gradient calculation
    if kernel_type == 'sobel':
        sobel_x, sobel_y = sobel_filters()
        gx = convolution(smoothed, sobel_x)
        gy = convolution(smoothed, sobel_y)
    elif kernel_type == 'custom':
        custom_kernel = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
        gx = convolution(smoothed, custom_kernel)
        gy = convolution(smoothed, np.rot90(custom_kernel))
    else:
        raise ValueError("Invalid kernel_type. Choose 'sobel' or 'custom'.")
    
    # Magnitude and orientation
    magnitude = np.hypot(gx, gy)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(gy, gx)
    
    # Non-maximum suppression
    suppressed = non_max_suppression(magnitude, direction)
    
    # Double thresholding
    thresholded = threshold(suppressed, low_threshold, high_threshold)
    
    # Edge tracking by hysteresis
    final_edges = hysteresis(thresholded)
    
    return final_edges

def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale.
    """
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

def main():
    """
    Main function to process the image and apply Canny edge detection.
    """
    image_path = '/home/danyez87/Master AI/CV/HW 2/Gura_Portitei_Scara_100.jpg'
    image = np.array(Image.open(image_path))
    image_gray = rgb_to_grayscale(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Apply Canny edge detection with Sobel kernels
    canny_sobel = canny_edge_detection(image_gray, kernel_type='sobel')
    axes[1].imshow(canny_sobel, cmap='gray')
    axes[1].set_title('Canny Edge Detection (Sobel)')
    axes[1].axis('off')

    # Apply Canny edge detection with custom kernel
    canny_custom = canny_edge_detection(image_gray, kernel_type='custom')
    axes[2].imshow(canny_custom, cmap='gray')
    axes[2].set_title('Canny Edge Detection (Custom)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('results/edge_detection_results/edge_detection_comparison_100.png')
    plt.close()

if __name__ == "__main__":
    main()