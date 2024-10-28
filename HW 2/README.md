Computer Vision - Homework 2 - Daniel-Cristian-Marian Tapusi

The homework is divided into 3 Python files that can be individually run as:

python task2.py
python task3.py
python task4.py

To use a different input image, simply change the image filename in the respective Python script.

I. Gaussian and Box Filters (task2.py)

Gaussian and Box filters were implemented from scratch and tested with kernels of size 3, 5, 7, and 9, and sigma values of 1, 2, and 5 for all image scales. Results can be found in the "result/gaussian_box_results" folder. 

Observations:
    * Larger kernel sizes produce more blurred outputs.
    * The Gaussian filter preserves edges better and provides smoother output compared to the Box filter.
    * Filter effects vary with image scale, being more pronounced at larger scales.

II. Edge Detector (task3.py)

Two edge detectors were implemented: one based on the Sobel operator and a custom one. Both use 3x3 kernels. Results are in the "result/edge_detection_results" folder, comparing both approaches for different image scales, with clearer, more defined results provided by the custom matrix, highlighting edges more prominently.

III. Object Detection Filter (task4.py)

A set of filters was created to highlight three specific structures: the blue pool, the building with an orange roof, and the "H" marked landing site. Results are in the "result/object_detection" folder.

The code implements a color-based object detection system for identifying a pool, helipad, and building with an orange roof in an image. It uses custom RGB to HSV conversion and color filtering functions to create masks for each object. A median blur is applied to each mask to reduce noise. The program then applies contour detection to find the best-fitting shapes for each object, drawing bounding boxes around them. Finally, it saves the resulting image with detected objects as well as individual masks for each object type.