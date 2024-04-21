import cv2
import numpy as np
import os

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 5000
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    # print(image)
    # print(gauss)
    noisy_image = image + gauss
    return noisy_image

def process_images(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            img_path = os.path.join(input_directory, filename)
            image = cv2.imread(img_path)
            if image is not None:
                noisy_image = add_gaussian_noise(image)
                # Clip the values to be in the valid range [0, 255] and convert back to uint8
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                
                # Save the noisy image to the output directory
                output_img_path = os.path.join(output_directory, filename)
                cv2.imwrite(output_img_path, noisy_image)

# Set the paths for the input and output directories
input_directory = 'driving_videos/Video_3/images'
output_directory = 'noise/Video_2_3/images'

# Process the images and save the noisy versions to the output directory
process_images(input_directory, output_directory)