import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def box_filter(size):
    """
    Define a function to create a box filter kernel.
    """
    return np.ones((size, size), dtype=np.float32) / (size * size)

def gaussian_filter(size, sigma):
    """
    Define a function to create a Gaussian filter kernel.
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    # Calculate Gaussian values for each position in the kernel
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel
    return kernel / np.sum(kernel)

def pad_image(image, kernel_size):
    """
    Define a function to pad an image for convolution.
    """
    pad_size = kernel_size // 2
    height, width, channels = image.shape
    padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size, channels), dtype=image.dtype)
    padded_image[pad_size:pad_size+height, pad_size:pad_size+width] = image
    return padded_image

def convolution(image, kernel):
    """
    Define a function to perform convolution on an image.
    """
    h, w, c = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Pad the image
    padded = pad_image(image, k_h)
    result = np.zeros_like(image)

    # Perform convolution
    for i in range(h):
        for j in range(w):
            for k in range(c):
                result[i, j, k] = np.sum(kernel * padded[i:i+k_h, j:j+k_w, k])
    
    return result.astype(np.uint8)

def plot_images(original, gaussian_filtered_list, box_filtered, kernel_size, sigma_values, save=False):
    """
    Define a function to plot and save the filtered images
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Plot original image
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Plot Gaussian filtered images
    for i, (gaussian_filtered, sigma) in enumerate(zip(gaussian_filtered_list, sigma_values), start=1):
        axes[i].imshow(gaussian_filtered)
        axes[i].set_title(f'Gaussian Filter {kernel_size}x{kernel_size}, σ={sigma}')
        axes[i].axis('off')

    # Plot Box filtered image
    axes[4].imshow(box_filtered)
    axes[4].set_title(f'Box Filter {kernel_size}x{kernel_size}')
    axes[4].axis('off')

    plt.tight_layout()
    
    # Save or display the plot
    if save:
        output_dir = 'results/gaussian_box_results'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/Scara_080_comparison_{kernel_size}x{kernel_size}.png')
        plt.close()
    else:
        plt.show()

def main():
    """
    Main function to process the image
    """
    image_path = '/home/danyez87/Master AI/CV/HW 2/Gura_Portitei_Scara_010.jpg'

    # Open and convert the image to RGB
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Define kernel sizes and sigma values for filtering
    kernel_sizes = [3, 5, 7, 9]
    sigma_values = [1, 2, 5]

    # Process the image with different kernel sizes
    for size in kernel_sizes:
        # Create and apply box filter
        box_kernel = box_filter(size)
        box_filtered = convolution(image_np, box_kernel)

        gaussian_filtered_list = []

        print(f"\nApplying filters with kernel size {size}x{size}")

        # Create and apply Gaussian filters with different sigma values
        for sigma in sigma_values:
            print(f"Applying Gaussian filter with sigma={sigma}")

            gaussian_kernel = gaussian_filter(size, sigma)
            print(f"Gaussian Kernel ({size}x{size}), σ={sigma}:\n{gaussian_kernel}")

            gaussian_filtered = convolution(image_np, gaussian_kernel)
            gaussian_filtered_list.append(gaussian_filtered)

        print(f"Box Kernel ({size}x{size}):\n{box_kernel}")

        # Plot and save the results
        plot_images(
            original=image,
            gaussian_filtered_list=gaussian_filtered_list,
            box_filtered=box_filtered,
            kernel_size=size,
            sigma_values=sigma_values,
            save=True  
        )

# Entry point of the script
if __name__ == "__main__":
    main()