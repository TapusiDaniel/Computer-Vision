# Computer Vision - Homework 3 - Daniel-Cristian-Marian Țăpuși

# Face Detection Using Skin Segmentation

This project implements face detection using skin color segmentation and shape analysis in Python. The script processes images to detect faces by identifying skin regions and fitting ellipses around potential face areas.

## Prerequisites

The following Python packages are required:
```
opencv-python (cv2)
numpy  
matplotlib
scikit-image
```

## Project Structure & Usage

The code is contained in a single Python file that can be run as:
```
python HW3.py
```

To use different input images:
* Create an `input_images` directory
* Place your JPG/JPEG/PNG images in this directory
* Run the script
* Results will be saved in the `results` folder

## Implementation Details

The face detection process consists of several key steps:

1. **Skin Color Segmentation**
   * Uses both HSV and YCrCb color spaces
   * Creates masks based on skin color ranges
   * Combines masks for robust detection

2. **Region Processing**
   * Removes small components
   * Applies morphological operations
   * Filters regions based on size and shape

3. **Face Detection**
   * Analyzes remaining regions for face-like properties
   * Fits ellipses around detected face regions

## Output Generation

For each processed image, the script generates visualization plots showing:
* Original input image
* Skin detection mask
* Processed binary mask
* Final result with green ellipses marking detected faces

All results are saved as PNG files in the `results` directory.
