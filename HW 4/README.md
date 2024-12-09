# Computer Vision - Homework 4 - Panorama Creation

This project implements panorama creation using feature detection and image stitching in Python. The script processes sequences of overlapping images to create panoramic views by detecting and matching features between adjacent images.

## Prerequisites

The following Python packages are required:
```
opencv-python (cv2)
numpy  
pathlib
```

## Project Structure & Usage

The code is contained in a single Python file that can be run as:
```
python HW4.py
```

To create panoramas:
* Place your image sequences in the `HW 4` directory
* Name your images as:
  * First sequence: ss1_1.png, ss1_2.png, ss1_3.png, etc.
  * Second sequence: ss2_1.png, ss2_2.png, ss2_3.png, etc.
* Run the script
* Results will be saved as:
  * First panorama: panorama1_result.png
  * Second panorama: panorama2_result.png

## Implementation Details

The panorama creation process consists of several key steps:

1. **Image Processing**
   * Resizes images to manageable dimensions
   * Starts from middle image and processes both directions
   * Removes black edges from final result

2. **Feature Detection and Matching**
   * Uses SIFT for feature detection
   * Implements FLANN-based feature matching
   * Applies Lowe's ratio test for match quality

3. **Image Stitching**
   * Computes homography matrix using RANSAC
   * Applies perspective transformation
   * Uses Gaussian blending for seamless transitions

4. **Optimization**
   * Implements memory management with garbage collection
   * Handles edge cases and potential errors
   * Optimizes for large image sequences

## Output Generation

For each sequence of images, the script generates:
* A panoramic view combining all input images
* Results are automatically saved in the `HW 4` directory
* To create panoramas from different sequences, simply change 'ss1' to 'ss2' in the image path strings and update the output filename

The final panoramas are automatically resized and cleaned of black edges for optimal viewing.