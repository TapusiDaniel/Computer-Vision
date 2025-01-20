# Computer Vision - Homework 6 - Face Recognition using Eigenfaces

This project implements a face recognition system using the Eigenfaces method, based on Principal Component Analysis (PCA) using the methods presented in the paper: “Eigenfaces for Recognition”, M Turk, A Pentland - Journal of cognitive neuroscience, 1991. The goal is to classify face images into one of five categories (persons) using training and test datasets.

## Project Description

The Eigenfaces method is a technique for face recognition that uses PCA to reduce the dimensionality of face images and extract the most important features (Eigenfaces). The system is trained on a set of face images and then used to classify test images into one of five categories (persons).

The project is implemented in Python and consists of the following steps:
1. **Training**: Compute the Eigenfaces and weights for the training images
2. **Testing**: Classify test images using the trained model
3. **Evaluation**: Calculate and display the classification accuracy

## Requirements

To run this project, you need:
- **Python 3.7 or later**.
- The following Python libraries:
  - `numpy`
  - `opencv-python`
  - `pathlib`
- The dataset provided in the `Faces_training_and_testing_images` folder, containing:
  - Training images in `Faces_training_and_testing_images/Train`
  - Test images in `Faces_training_and_testing_images/Test`

  You can install the required libraries using pip:
```bash
pip install numpy opencv-python